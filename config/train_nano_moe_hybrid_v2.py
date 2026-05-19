import time

# config for training GPT-2 (124M) baseline model (one expert) on two RTX 3090 GPUs
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=2 train.py config/train_nano_moe.py

wandb_log = True
init_from = 'scratch'
wandb_project = 'cs-628-moe'
wandb_run_name ='gpt2-124M-moe-owt ' + time.strftime('%Y-%m-%d %H:%M:%S')


eval_capacity = 2.0
stride = 2

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
block_size = 1024

# this makes total number of tokens be 25B
max_iters = 50000
lr_decay_iters = 50000

# eval stuff
eval_interval = 500
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# 1. THE GOLDILOCKS CORE (Deep enough to explode, wide enough to be fast)
n_layer = 128     # 16 is enough sequential routers to cause a cascade failure
n_head = 4
n_embd = 128     # Slightly wider to keep Tensor Cores fed
block_size = 1024

# 2. H200 THROUGHPUT (The Speed Fix)
# Thicker batch size = High GPU utilization. 
batch_size = 16  
# Low accumulation = Lightning fast weight updates (watch it crash quickly!)
gradient_accumulation_steps = 8 

# 3. MOE CHAOS 
n_exp = 64       
top_k = 1 
train_capacity = 1.0 #10.0 

# 4. SAFETY OFF
use_aux_loss = False
use_router_z_loss = False
use_switch_tfm_init = False
use_noisy_top_k = True

# KEEP LEARNING RATE HIGH TO FORCE THE ISSUE
learning_rate = 6e-4
grad_clip = 10.0