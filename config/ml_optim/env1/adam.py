import time

# config for training GPT-2 baseline model (moe) on two GPUs
# launch as the following (e.g. in a screen session):
# $ torchrun --standalone --nproc_per_node=2 train.py config/train_moe_adam.py

wandb_log = True
init_from = 'scratch'
wandb_project = 'ml-optim'
wandb_run_name ='gpt2-moe-adam ' + time.strftime('%Y-%m-%d %H:%M:%S')

# model/moe settings
n_exp = 8
top_k = 1 
use_aux_loss = False
aux_loss_weight = 0.0 
use_router_z_loss = False
router_z_loss_weight = 0.0 
use_noisy_top_k = False
train_capacity = 8
eval_capacity = 2.0
stride = 1
use_switch_tfm_init = True
switch_tfm_init_scale = 1.0  
router_use_full_prec = True

# use smaller GPT model
n_layer = 2
n_head = 6
n_embd = 384

# total batch size
batch_size = 32
block_size = 1024
gradient_accumulation_steps = 2 #from 16
gpu_count = 2

tokens_iter = batch_size * block_size * gradient_accumulation_steps * gpu_count
tokens_expert = tokens_iter * top_k / n_exp
limit_expert = tokens_expert * train_capacity
if True: print(f'tokens, per iter: {tokens_iter}, per expert: {tokens_expert}; limit: {limit_expert}')

max_iters = 7500
lr_decay_iters = 30000

# eval stuff
eval_interval = 500
eval_iters = 20
log_interval = 10

# adam optimizer
optimizer_choice = 'adam'
weight_decay = 0.0 # Standard Adam decoupled weight decay is 0
beta1 = 0.9
beta2 = 0.95
momentum = 0
grad_clip = 5.0 # high ghost clip

# learning rate decay settings
decay_lr = True 
warmup_iters = 500 
learning_rate = 6e-4 
min_lr = 6e-5