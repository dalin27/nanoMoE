import time
import os

# config for training GPT-2 Base (MoE) - Scaled for realistic optimizer benchmarking
# launch as: torchrun --standalone --nproc_per_node=4 train.py config/env2.py

wandb_log = True
init_from = 'scratch'
wandb_project = 'ml-optim'

optimizer_choice = 'muon'
wandb_run_name = f'env2_{optimizer_choice}_' + time.strftime('%Y-%m-%d %H:%M:%S')

max_iters = 7000
lr_decay_iters = 30000

eval_interval = 500
eval_iters = 20
log_interval = 10

# model/moe settings - 16 Experts to maintain 1/8 exposure ratio
n_exp = 16
top_k = 2 

# Regularization to simulate realistic training dynamics
use_aux_loss = True
aux_loss_weight = 0.01          
use_router_z_loss = True
router_z_loss_weight = 0.001    
use_noisy_top_k = True          

train_capacity = 1.25           
eval_capacity = 2.0
use_switch_tfm_init = True
switch_tfm_init_scale = 1.0  
router_use_full_prec = True

# Standard GPT-2 Base dimensions but interleaved with MoE
n_layer = 12                    
n_head = 12                     
n_embd = 768                    
stride = 2                      

# Total batch size (unchanged)
batch_size = 32
block_size = 1024
gradient_accumulation_steps = 8
gpu_count = 4

tokens_iter = batch_size * block_size * gradient_accumulation_steps * gpu_count
tokens_expert = tokens_iter * top_k / n_exp
limit_expert = tokens_expert * train_capacity
if True: print(f'tokens, per iter: {tokens_iter}, per expert: {tokens_expert}; limit: {limit_expert}')

# Homogeneous Data Shock Parameters
step_shock_start = 5000
step_recovery_start = 5500 

# Learning rate decay settings
decay_lr = True 
warmup_iters = 500 

# ==========================================
# OPTIMIZER SPECIFIC CONFIGURATIONS
# ==========================================

muon_ns_steps = 5

learning_rate = 0.02   # Massive LR for 2D orthogonal updates
min_lr = 0.002
adam_lr = 3e-4         # Fallback LR for embeddings, biases, and layernorms
weight_decay = 0.01 
momentum = 0.95        # Nesterov momentum
muon_ns_steps = 5           
grad_clip = 1.0