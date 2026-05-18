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

# this makes total number of tokens be 25B
max_iters = 50000
lr_decay_iters = 50000

# eval stuff
eval_interval = 500
eval_iters = 200
log_interval = 10

# weight decay
weight_decay = 1e-1

# 1. small core
n_layer = 48     
n_head = 2
n_embd = 64

# 2. 
batch_size = 2
gradient_accumulation_steps = 256
block_size = 1024


# 3.
n_exp = 64       
top_k = 1 
train_capacity = 10.0  #slack 

# 4. 
use_aux_loss = False
use_router_z_loss = False
use_switch_tfm_init = False
use_noisy_top_k = True