import time

# config for training GPT-2 (124M) baseline model (one expert) on two RTX 3090 GPUs
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=2 train.py config/train_nano_moe.py

wandb_log = True
init_from = 'scratch'
wandb_project = 'ml-optim'
wandb_run_name ='gpt2-124M-moe-owt ' + time.strftime('%Y-%m-%d %H:%M:%S')

# model/moe settings
n_exp = 64
top_k = 1 #2
use_aux_loss = True
aux_loss_weight = 0.01 #0.05
use_router_z_loss = True
router_z_loss_weight = 0.001 
use_noisy_top_k = False
train_capacity = 1.5
eval_capacity = 2.0
stride = 2
use_switch_tfm_init = True
switch_tfm_init_scale = 1.0  # recommended 0.1 for stability (pg.10, https://arxiv.org/abs/2101.03961)
router_use_full_prec = True

# use smaller GPT model
n_layer = 6
n_head = 6
n_embd = 384

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 32
block_size = 1024
gradient_accumulation_steps = 16
gpu_count = 2

tokens_iter = batch_size * block_size * gradient_accumulation_steps * gpu_count
tokens_expert = tokens_iter * top_k / n_exp
limit_expert = tokens_expert * train_capacity
if True: print(f'tokens, per iter: {tokens_iter}, per expert: {tokens_expert}; limit: {limit_expert}')

# this makes total number of tokens be 25B
max_iters = 5000
lr_decay_iters = 5000

# eval stuff
eval_interval = 500
eval_iters = 200
log_interval = 10

# adamw optimizer
optimizer_choice = 'sgd'
weight_decay = 1e-4     
momentum = 0.9           
nesterov = True
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 500 #2000 # how many steps to warm up for
learning_rate = 1e-2 # max learning rate
min_lr = 1e-3 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla