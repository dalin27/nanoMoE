import time
import os

# config for training GPT-2 Base (MoE) - Scaled for realistic optimizer benchmarking
# launch as: torchrun --standalone --nproc_per_node=4 train.py config/env2.py

wandb_log = True
init_from = 'scratch'
wandb_project = 'ml-optim'
optimizer_choice = 'lion'

wandb_run_name = f'env2_{optimizer_choice}_' + time.strftime('%Y-%m-%d %H:%M:%S')

max_iters = 4000
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
batch_size = 16
block_size = 1024
gradient_accumulation_steps = 16
gpu_count = 4

tokens_iter = batch_size * block_size * gradient_accumulation_steps * gpu_count
tokens_expert = tokens_iter * top_k / n_exp
limit_expert = tokens_expert * train_capacity
if True: print(f'tokens, per iter: {tokens_iter}, per expert: {tokens_expert}; limit: {limit_expert}')

# Learning rate decay settings
decay_lr = True 
warmup_iters = 500 

learning_rate = 1e-4  
min_lr = 1e-5
weight_decay = 1.0    
beta1 = 0.9
beta2 = 0.99          
momentum = 0.0       
grad_clip = 1.0     

# Homogeneous Data Shock Parameters
step_shock_start = 2500
step_recovery_start = 3000 