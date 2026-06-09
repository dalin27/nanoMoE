"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
# os.environ['NCCL_P2P_DISABLE'] = '1'
# os.environ['NCCL_IGNORE_DISABLED_P2P'] = '1'
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, broadcast

from torch.nn import functional as F

from model import GPTConfig, GPT

from manager import MANAGER

from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
load_dotenv()
out_dir = 'checkpoints'
eval_interval = 500
log_interval = 10
old_log_int = log_interval
wandb_interval = 50
old_wandb_int = wandb_interval
eval_iters = 20
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

# wandb logging
wandb_log = True # False # disabled by default
wandb_project = 'cs-628-moe'
wandb_run_name = 'gpt2-124M-owt' + str(time.time())

# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?

# moe
n_exp = 8 # if n_exp = 1 we just use regular MLP layers
top_k = 2
use_aux_loss = True
use_router_z_loss = True
use_noisy_top_k = True
aux_loss_weight = 0.001
router_z_loss_weight = 0.01
train_capacity = 1.25
eval_capacity = 2.0
min_capacity = 4
stride = 2
use_switch_tfm_init = False
switch_tfm_init_scale = 1.0  # recommended 0.1 for stability (pg.10, https://arxiv.org/abs/2101.03961)
router_use_full_prec = True

# adamw optimizer
optimizer_choice = 'adamw'
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
momentum = 0
nesterov = False

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

gpu_count = 1

# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.

# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging

tokens_iter = int(batch_size * block_size * gradient_accumulation_steps * gpu_count)
tokens_expert = int(tokens_iter * top_k / n_exp)
limit_expert = int(tokens_expert * train_capacity)

config['moe_meta/tokens_iter'] = tokens_iter
config['moe_meta/tokens_expert_ideal'] = tokens_expert
config['moe_meta/limit_expert'] = limit_expert

print(config)
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

shockset = 'tiny-codes'
# poor man's data loader
data_dir = os.path.join('data', dataset)
data_dir_shock = os.path.join('data', shockset)
def get_batch(split, is_shock_phase=False):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    
    current_dir = data_dir_shock if (is_shock_phase and split == 'train') else data_dir
    if split == 'train':
        data = np.memmap(os.path.join(current_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(current_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout, n_exp=n_exp, top_k=top_k,
                  use_aux_loss=use_aux_loss, use_router_z_loss=use_router_z_loss,
                  use_noisy_top_k=use_noisy_top_k, aux_loss_weight=aux_loss_weight,
                  router_z_loss_weight=router_z_loss_weight, train_capacity=train_capacity,
                  eval_capacity=eval_capacity, min_capacity=min_capacity, stride=stride,
                  use_switch_tfm_init=use_switch_tfm_init, switch_tfm_init_scale=switch_tfm_init_scale,
                  router_use_full_prec=router_use_full_prec) # start with model_args from command line
print('\n\n')
print(model_args)
print('\n\n')
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(optimizer_choice,weight_decay, learning_rate, (beta1, beta2), device_type, momentum, nesterov)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        aux_losses = torch.zeros(eval_iters)
        z_losses = torch.zeros(eval_iters)
        running_router_probs = None

        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss, aux, z, _, _, r_probs, _ = model(X, Y)
            losses[k] = loss.item()
            aux_losses[k] = aux.item()
            z_losses[k] = z.item()
            layer_probs = [p.detach().view(-1, p.size(-1)).mean(dim=0) for p in r_probs]

            if running_router_probs is None:
                running_router_probs = layer_probs
            else:
                for i in range(len(layer_probs)):
                    running_router_probs[i] += layer_probs[i]
        out[split] = losses.mean()
        out[f"{split}_aux"] = aux.mean()
        out[f"{split}_z"] = z.mean()
        out[f'{split}_router_probs'] = [p / eval_iters for p in running_router_probs]
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

class TrainingState:
    def __init__(self):
        self.prev_router_weight = None
        self.prev_router_update = None

training_state = TrainingState()

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

#track collapse
collapse_threshold = 2.0
pre_shock_baseline_cv = 0.0 # You will calculate this dynamically before step 5000
step_shock_start = 5000
step_recovery_start = 5100
ema_cv = ema_loss = ema_kl = ema_grad = ema_dropped = None
peak_shock_loss = 0.0
total_excess_loss = 0.0
has_collapsed = False
has_recovered = False




stop_file_path = os.path.join(os.getcwd(), 'STOP')

while True:
    #manual stop
    stop_training = torch.tensor(0, dtype=torch.int32, device=device)
    if master_process:
        if os.path.exists(stop_file_path):
            print(f"\n[MANUAL STOP DETECTED] Fichier '{stop_file_path}' trouvé. Préparation de la sortie...")
            stop_training += 1
            try: os.remove(stop_file_path)
            except OSError: pass

    if ddp:
        # On propage le signal d'arrêt à tous les workers pour éviter un blocage
        broadcast(stop_training, src=0)
        
    if stop_training.item() > 0:
        if master_process:
            print("Sauvegarde du checkpoint d'urgence...")
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt_manual_stop.pt'))
            print("Checkpoint sauvegardé avec succès. Sortie.")
        break
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    #set shock log
    is_in_shock_window = (step_shock_start - 50) <= iter_num <= (step_recovery_start + 200)
    
    if is_in_shock_window:
        wandb_interval = 5  
        log_interval = 5
    else: 
        wandb_interval = old_wandb_int
        log_interval = old_log_int

    if iter_num % eval_interval == 0 and master_process:

        losses = estimate_loss()
        
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            eval_metrics = {
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "val/aux_loss": losses['val_aux'],
                "val/z_loss" : losses['val_z'],
                "charts/lr": lr,
                "charts/mfu": running_mfu*100, # convert to percentage
            }

            # Single unified loop to extract all MoE statistics
            if hasattr(raw_model, 'transformer') and hasattr(raw_model.transformer, 'h'):
                for layer_idx, block in enumerate(raw_model.transformer.h):
                    
                    # 1. Check if this layer has the MoE setup
                    if hasattr(block, 'mlp') and hasattr(block.mlp, 'experts'):
                        
                        # --- SVD Statistics (Magnitude Imbalance) ---
                        top_eigenvalues = block.mlp.experts.compute_svd_stats()
                        mean_eigen = top_eigenvalues.mean().item()
                        eigen_cv = top_eigenvalues.std(unbiased=False).item() / (mean_eigen + 1e-10)
                        
                        eval_metrics[f"experts/layer_{layer_idx}/mean_top_eigen"] = mean_eigen
                        eval_metrics[f"experts/layer_{layer_idx}/eigen_cv"] = eigen_cv
                        #eval_metrics[f"experts/layer_{layer_idx}/eigen_dist"] = wandb.Histogram(top_eigenvalues.cpu().numpy())

                        # --- SVD Statistics (Representational Divergence) ---
                        if hasattr(block.mlp.experts, 'compute_top_singular_vectors'):
                            # Expected shape: (n_exp, hidden_dim)
                            top_vectors = block.mlp.experts.compute_top_singular_vectors() 
                            
                            # Normalize vectors to unit length for cosine similarity
                            vectors_normalized = torch.nn.functional.normalize(top_vectors, p=2, dim=1)
                            
                            # Compute pairwise cosine similarity matrix
                            similarity_matrix = torch.matmul(vectors_normalized, vectors_normalized.t())
                            
                            # Extract off-diagonal elements (ignore self-similarity)
                            n_exp = similarity_matrix.size(0)
                            mask = ~torch.eye(n_exp, dtype=torch.bool, device=similarity_matrix.device)
                            pairwise_similarities = similarity_matrix[mask]
                            
                            # Log metrics
                            eval_metrics[f"experts/layer_{layer_idx}/mean_pairwise_sim"] = pairwise_similarities.mean().item()
                            eval_metrics[f"experts/layer_{layer_idx}/max_pairwise_sim"] = pairwise_similarities.max().item()

            wandb.log(eval_metrics)

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    if iter_num == 0 and eval_only:
        break
    
    running_max_r_l = -float('inf')
    running_mean_r_l = 0
    running_aux = 0.0
    running_z = 0.0 
    loss = 0.0

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss, aux_loss_val, z_loss_val, max_r_l, mean_r_l, router_probs, dropped_tokens= model(X, Y)
            loss = loss / gradient_accumulation_steps 
            running_aux += aux_loss_val.item() / gradient_accumulation_steps
            running_z += z_loss_val.item() / gradient_accumulation_steps
            running_max_r_l = max(running_max_r_l, max_r_l.item())
            running_mean_r_l += mean_r_l.item() / gradient_accumulation_steps
            # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        
        is_shock = step_shock_start <= iter_num < step_recovery_start
        X, Y = get_batch('train',is_shock_phase=is_shock)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()

    actual_model = model.module if ddp else model
    router_weights = [p for n, p in raw_model.named_parameters() if 'w_g.weight' in n]
    
    if len(router_weights) > 0:
        target_router_weight = router_weights[0] # We track layer 0 for momentum drag
        if target_router_weight.grad is not None:
            grad_t = target_router_weight.grad.detach().clone()
            weight_pre_step = target_router_weight.detach().clone()
        else:
            grad_t = None
    else:
        grad_t = None

    # clip the gradient
    total_norm = 0.0
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        total_norm = total_norm.item()

    kl_div = 0
    lossf = 0

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        torch.cuda.empty_cache()
        
        # scale up to undo the division above, approximating the true total loss
        lossf = loss.item() * gradient_accumulation_steps
        
        if local_iter_num >= 5: 
            total_batch_scaled = (batch_size * gradient_accumulation_steps) * ddp_world_size 
            raw_mfu = raw_model.estimate_mfu(total_batch_scaled, dt)
            gpu_mfu = raw_mfu / ddp_world_size
            running_mfu = gpu_mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*gpu_mfu
        
        if wandb_log and iter_num % wandb_interval == 0:
            all_router_probs = torch.cat(router_probs, dim=0)

            expert_assignments = torch.argmax(all_router_probs, dim=-1)
            expert_assignments_flat = expert_assignments.flatten()
            expert_counts = torch.bincount(expert_assignments_flat, minlength=model.module.config.n_exp)            
            dead_experts = (expert_counts == 0).sum().item()

            # Since the router no longer logs kl_div and capacity_cv to MANAGER during the compiled 
            # forward pass, we only pull the cosine similarities here.
            _, router_cos_sim, grad_cos_sim, _ = MANAGER.get_and_reset_collapse_metrics()

            # Initialize tracking accumulators for the metrics harvested from the router self attributes
            total_dropped = 0
            total_kl = 0.0
            total_cv = 0.0
            layers_counted = 0
            
            layer_metrics = {}

            # --- Per-Layer Router Stats, Gradients & Metric Harvesting ---
            if hasattr(raw_model, 'transformer') and hasattr(raw_model.transformer, 'h'):
                if master_process:
                    for layer_idx, block in enumerate(raw_model.transformer.h):
                        
                        if hasattr(block, 'mlp'):

                            if hasattr(block.mlp, 'router'):
                                router = block.mlp.router
                                
                                # Harvest metrics saved during the forward pass
                                if hasattr(router, 'latest_dropped_tokens'):
                                    total_dropped += router.latest_dropped_tokens.item()
                                    total_kl += router.latest_kl_div.item()
                                    total_cv += router.latest_capacity_cv.item()
                                    layers_counted += 1

                                # Get gradients safely (Keeping norm on GPU)
                                if hasattr(router, 'w_g') and router.w_g.weight.grad is not None:
                                    layer_router_norm_tensor = router.w_g.weight.grad.data.norm(2).item()
                                    layer_metrics[f"router/layer_{layer_idx}/grad_norm"] = layer_router_norm_tensor

                            # 1. Track Entropy and Dead Experts (Keeping math on GPU)
                            if hasattr(block.mlp, 'total_tracked_tokens') and block.mlp.total_tracked_tokens > 0:
                                tokens = block.mlp.total_tracked_tokens.item()
                                avg_entropy = block.mlp.running_entropy_sum.item() / tokens
                                expected_value = tokens_expert 
                                threshold = expected_value * 0.05
                                layer_und_experts_tensor = (block.mlp.running_expert_counts < threshold).sum().item()
                                layer_dead_experts_tensor = (block.mlp.running_expert_counts == 0).sum().item()

                                
                                layer_metrics[f"router/layer_{layer_idx}/entropy"] = avg_entropy
                                layer_metrics[f"router/layer_{layer_idx}/dead_experts"] = layer_dead_experts_tensor / model.module.config.n_exp
                                layer_metrics[f"router/layer_{layer_idx}/less_0.05_experts"] = layer_und_experts_tensor / model.module.config.n_exp

                                
                                block.mlp.total_tracked_tokens.zero_()
                                block.mlp.running_entropy_sum.zero_()
                                block.mlp.running_expert_counts.zero_()
                            
            # Calculate global averages from the harvested layers
            avg_kl_div = total_kl / layers_counted if layers_counted > 0 else 0.0
            kl_div = avg_kl_div

            avg_capacity_cv = total_cv / layers_counted if layers_counted > 0 else 0.0

            # Build the final metrics dictionary
            train_metrics = {
                "iter": iter_num,
                "train/loss_instant": lossf, 
                "train/aux_loss": running_aux,
                "train/z_loss": running_z, 
                "charts/lr": lr,
                "charts/mfu": running_mfu * 100,
                "charts/grad_norm": total_norm,
                "global_router/max_logit": running_max_r_l,
                "global_router/avg_logit": running_mean_r_l,
                "global_router/expert_counts": wandb.Histogram(expert_counts.cpu().numpy()),
                "global_router/dead_experts": dead_experts,
                "global_router/dropped_tokens": total_dropped,
                "global_router/kl_divergence_from_uni": avg_kl_div,
                "global_router/router_update_cos_sim": router_cos_sim,
                "global_router/grad_update_cos_sim": grad_cos_sim,
                "global_router/capacity_cv": avg_capacity_cv
            }
            
            # Merge layer metrics into the main payload
            train_metrics.update(layer_metrics)

            wandb.log(train_metrics)
            router_probs.clear()

        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    
    # TCCR tracking
    if hasattr(raw_model, 'transformer') and hasattr(raw_model.transformer, 'h'):
        # Extract router probabilities from the LAST layer
        last_layer_idx = len(raw_model.transformer.h) - 1
        last_block = raw_model.transformer.h[last_layer_idx]
        
        if hasattr(last_block.mlp, 'router') and hasattr(last_block.mlp.router, 'latest_probs'):
            # Calculate CV of expert distribution
            probs = last_block.mlp.router.latest_probs
            expert_assignments = torch.argmax(probs.view(-1, probs.size(-1)), dim=-1)
            expert_counts = torch.bincount(expert_assignments, minlength=n_exp).float()
            
            current_cv = (expert_counts.std(unbiased=False) / (expert_counts.mean() + 1e-10)).item()
            
            # --- 1. Update Exponential Moving Averages (EMA) ---
            alpha = 0.1
            if ema_loss is None:
                ema_cv, ema_loss, ema_kl, ema_grad, ema_dropped = current_cv, lossf, kl_div, total_norm, total_dropped
            else:
                ema_cv = alpha * current_cv + (1 - alpha) * ema_cv
                ema_loss = alpha * lossf + (1 - alpha) * ema_loss
                ema_kl = alpha * kl_div + (1 - alpha) * ema_kl
                ema_grad = alpha * total_norm + (1 - alpha) * ema_grad
                ema_dropped = alpha * total_dropped + (1 - alpha) * ema_dropped

            if master_process and wandb_log:
                wandb.log({
                    "EMA/CV": ema_cv,
                    "EMA/Loss": ema_loss,
                    "EMA/KL_Div": ema_kl,
                    "EMA/Grad_Norm": ema_grad,
                    "EMA/dropped_tokens": ema_dropped,
                }, step=iter_num)

            # --- 2. Capture Pre-Shock Baseline ---
            if iter_num == step_shock_start - 1:
                pre_shock_baseline_cv = ema_cv
                pre_shock_baseline_loss = ema_loss
                pre_shock_baseline_kl = ema_kl
                pre_shock_baseline_grad = ema_grad
                pre_shock_baseline_dropped = ema_dropped
                # Explicitly set collapse flag so the tracking knows the shock has begun
                has_collapsed = True 

            # --- 3. Track the 3 Key Metrics DURING the shock ---
            if has_collapsed and not has_recovered:
                
                # Metric A: Peak Shock Loss (Severity)
                if ema_loss > peak_shock_loss:
                    peak_shock_loss = ema_loss
                    
                # Metric B: Total Wasted Compute (Excess Loss Area)
                # We only accumulate loss that is strictly above our baseline
                if ema_loss > pre_shock_baseline_loss:
                    total_excess_loss += (ema_loss - pre_shock_baseline_loss)

            # --- 4. Recovery Check ---
            if iter_num >= step_recovery_start and has_collapsed and not has_recovered:
                
                # Define health criteria using smoothed metrics against baselines
                is_cv_recovered = ema_cv <= (pre_shock_baseline_cv * 1.10)
                is_loss_recovered = ema_loss <= (pre_shock_baseline_loss * 1.10)
                is_kl_recovered = ema_kl <= (pre_shock_baseline_kl * 1.10)
                is_grad_recovered = ema_grad <= (pre_shock_baseline_grad * 1.15)
                is_dropped_recovered = ema_dropped <= (pre_shock_baseline_dropped * 1.10)
                
                # Check if ALL smoothed metrics meet recovery criteria
                if is_cv_recovered and is_loss_recovered and is_kl_recovered and is_grad_recovered and is_dropped_recovered:
                    has_recovered = True
                    total_shock_duration = iter_num - step_shock_start
                    
                    # Metric C: Average Recovery Rate
                    # Avoid division by zero if it recovers instantly
                    duration_divisor = max(total_shock_duration, 1) 
                    avg_recovery_rate = (peak_shock_loss - pre_shock_baseline_loss) / duration_divisor
                    
                    if master_process:
                        print(f"\n[FULL RECOVERY] Metrics normalized in {total_shock_duration} steps!")
                        print(f"  -> Peak Severity: {peak_shock_loss - pre_shock_baseline_loss:.4f}")
                        print(f"  -> Total Excess Loss: {total_excess_loss:.4f}")
                        
                        if wandb_log:
                            wandb.log({
                                "metrics/Total_Shock_Duration": total_shock_duration,
                                "metrics/Peak_Loss_Severity": peak_shock_loss - pre_shock_baseline_loss,
                                "metrics/Total_Wasted_Loss_Cost": total_excess_loss,
                                "metrics/Average_Recovery_Rate": avg_recovery_rate,
                            }, step=iter_num)


    # =========================================================================
    # PRE-STEP: Capture gradients before the optimizer modifies or erases them
    # =========================================================================
    with torch.no_grad():
        router_layer_target = model.module.transformer.h[-1].mlp.router.w_g if hasattr(model, 'module') else model.transformer.h[-1].mlp.router.w_g
        weight_pre_step = router_layer_target.weight.detach().clone()
        grad_t = router_layer_target.weight.grad.detach().clone() if router_layer_target.weight.grad is not None else None

    # Step the optimizer and scaler
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # =========================================================================
    # POST-STEP: Calculate weight updates and log cosine similarities
    # =========================================================================
    with torch.no_grad():
        current_weight = router_layer_target.weight.detach()
        
        # 1. Gradient vs. Weight Update Cosine Similarity
        if grad_t is not None:
            update_t = current_weight - weight_pre_step
            # Avoid zero-division on flat updates
            if grad_t.norm() > 1e-7 and update_t.norm() > 1e-7:
                cos_sim_grad = F.cosine_similarity(grad_t.flatten(), update_t.flatten(), dim=0)
                MANAGER.add_grad_update_cos_sim(cos_sim_grad.item())

        # 2. Step-to-Step Router Weight Trajectory Tracking
        if iter_num > 0 and getattr(training_state, 'prev_router_weight', None) is not None:
            actual_update = current_weight - training_state.prev_router_weight
            
            if getattr(training_state, 'prev_router_update', None) is not None:
                cos_sim_step = F.cosine_similarity(
                    actual_update.flatten(), 
                    training_state.prev_router_update.flatten(), 
                    dim=0
                )
                MANAGER.add_router_update_cos_sim(cos_sim_step.item())
            
            training_state.prev_router_update = actual_update.clone()
        
        training_state.prev_router_weight = current_weight.clone()

    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()