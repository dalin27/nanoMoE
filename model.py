"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.nn import functional as F

from manager import MANAGER

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Router(nn.Module):
    def __init__(self, config):
        super().__init__()

        # router settings
        self.top_k = config.top_k
        self.n_exp = config.n_exp
        assert self.top_k >= 1 and self.top_k <= config.n_exp
        self.use_noisy_top_k = config.use_noisy_top_k
        self.train_capacity = config.train_capacity
        self.eval_capacity = config.eval_capacity
        self.min_capacity = config.min_capacity
        self.router_use_full_prec = config.router_use_full_prec

        # auxiliary / load balancing loss settings
        self.use_aux_loss = config.use_aux_loss
        self.use_router_z_loss = config.use_router_z_loss

        # linear projection for (noisy) softmax gating
        # no bias is used, see page 4 eq (4) in (https://arxiv.org/abs/1701.06538)
        self.w_g = nn.Linear(config.n_embd, config.n_exp, bias=False)
        self.w_noise = nn.Linear(config.n_embd, config.n_exp, bias=False) if self.use_noisy_top_k else None

        #gradient update tracking 
        self.prev_weight = None
        self.prev_update = None

    def forward(self, x):
        # optionally run the router in full precision to avoid instability during training
        # see discussion on pg. 9 here: https://arxiv.org/abs/2101.03961
        # setting enabled to False in autocast automatically puts everything in float32
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu' # for later use in torch.autocast
        ctx = nullcontext() if not self.router_use_full_prec else torch.amp.autocast(device_type=device_type, enabled=False)

        with ctx:
            B, T, _ = x.size()
            num_tokens = B * T

            # eq (4) in (https://arxiv.org/abs/1701.06538)
            logits = self.w_g(x)  # [B, T, n_exp]
            if self.use_noisy_top_k:
                # optionally add noise into the router
                noise = F.softplus(self.w_noise(x))
                noise *= torch.randn_like(noise)
                logits += noise

            # router z loss, computed on logits (before softmax)
            if self.use_router_z_loss:
                z_loss = self.compute_router_z_loss(logits)
                # Saved to self WITHOUT detach() so the training loop can add it to the main loss
                self.z_loss = z_loss 
            else:
                self.z_loss = 0.0

            # find top k experts for each token
            top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1) # [B, T, k]

            # normalize expert probabilities via softmax
            full_router_probs = F.softmax(logits, dim=-1)

            # Extract metrics for the training loop's logging block
            self.latest_probs = full_router_probs.detach() 

            mean_router_probs = full_router_probs.mean(dim=(0, 1)) # [n_exp]
            uniform_prob = 1.0 / self.n_exp
            
            # D_KL(P || U) = sum(P * log(P / U))
            kl_div = torch.sum(
                mean_router_probs * (torch.log(mean_router_probs + 1e-10) - math.log(uniform_prob))
            )
            self.latest_kl_div = kl_div.detach()

            router_probs = torch.full_like(logits, float('-inf'))  # [B, T, n_exp]
            router_probs.scatter_(-1, top_k_indices, top_k_logits)

            max_router_logit = top_k_logits.max().detach()
            avg_router_logit = top_k_logits.mean().detach()

            self.latest_max_logit = max_router_logit
            self.latest_avg_logit = avg_router_logit

            router_probs = F.softmax(router_probs, dim=-1)

            # compute auxiliary load balancing loss
            if self.use_aux_loss:
                aux_loss = self.compute_aux_loss(router_probs, top_k_indices)
                # Saved to self WITHOUT detach() so the training loop can use it for gradients
                self.aux_loss = aux_loss
            else:
                self.aux_loss = 0.0

            # compute expert capacity
            exp_capacity = self.get_capacity(num_tokens)

            # make a multi-hot mask of chosen experts, size [B, T, n_exp]
            exp_mask = F.one_hot(top_k_indices, num_classes=self.n_exp)  # [B, T, k, n_exp]
            exp_mask = exp_mask.view(num_tokens, self.top_k, self.n_exp)  # [B * T, k, n_exp]
            exp_mask = exp_mask.permute(1, 0, 2) # [k, B * T, n_exp]

            # compute cumulative sum of each token over experts
            exp_rank = exp_mask.reshape(self.top_k * num_tokens, self.n_exp)  # [k * B * T, n_exp]
            exp_rank = torch.cumsum(exp_rank, dim=0) - 1  # cumulative sum of expert selections [k * B * T, n_exp]
            exp_rank = exp_rank.reshape(self.top_k, num_tokens, self.n_exp)  # [k, B * T, n_exp]

            # mask out (set to zero) entries that go beyond expert capacity
            exp_mask *= torch.lt(exp_rank, exp_capacity) # [k, B * T, n_exp]
            used_capacity = torch.sum(exp_mask, dim=(0, 1)) # [n_exp]

            total_routing_requests = num_tokens * self.top_k
            total_accepted = used_capacity.sum()
            dropped_tokens = total_routing_requests - total_accepted
            
            self.latest_dropped_tokens = dropped_tokens.detach()

            # CV 
            used_capacity_float = used_capacity.float()
            cv_load = used_capacity_float.std(unbiased=False) / (used_capacity_float.mean() + 1e-10)
            
            self.latest_capacity_cv = cv_load.detach()

            # mask rank to only include tokens that are selected
            exp_rank = torch.sum(exp_mask * exp_rank, dim=-1)  # [k, B * T]

            # mask probabilities to only include selected experts
            select_probs = router_probs.view(num_tokens, self.n_exp)[None, :] # [1, B * T, n_exp]
            exp_weights = exp_mask * select_probs # [k, B * T, n_exp]

            # convert rank into one-hot vectors over the available capacity
            exp_rank_sc = F.one_hot(exp_rank, num_classes=exp_capacity) # [k, B * T, exp_capacity]

            # create a vector that stores weight parameters per token location placement
            cb_weight = torch.sum(exp_weights.unsqueeze(3) * exp_rank_sc.unsqueeze(2), dim=0)
            sec_mask = cb_weight.bool() 
            
            return used_capacity, cb_weight, sec_mask
    
    def compute_aux_loss(self, expert_probs: torch.Tensor, indices: torch.Tensor):
        """
        Computes Switch Transformer auxiliary loss (https://arxiv.org/abs/2101.03961)
        See equations (4)-(6) on page 7
        """

        # equation (5): compute ratio of tokens allocated to each expert
        # total number of tokens is defined as total tokens in batch * k
        # (k = 1) for the Switch Transformer
        with torch.no_grad():
            one_hot_indices = F.one_hot(indices, num_classes=self.n_exp)  # [B, T, k, n_exp]
            one_hot_indices = torch.sum(one_hot_indices.float(), dim=2)  # [B, T, n_exp] (sum over k dimension)
            tokens_per_expert = torch.mean(one_hot_indices.float(), dim=(0, 1))

        # equation (6): compute ratio of router probability allocated to each expert
        prob_per_expert = torch.mean(expert_probs.float(), dim=(0, 1))

        # equation (4): take a scaled dot product between prob/token allocation vectors
        # multiply the result by the number of experts
        return self.n_exp * torch.sum(prob_per_expert * tokens_per_expert)
    
    def compute_router_z_loss(self, logits: torch.Tensor):
        """
        Computes ST-MoE router z loss (https://arxiv.org/abs/2202.08906)
        See equation (5) on page 7
        """
    
        # exponentiate logits, sum logits of each expert, take log, and square
        # code below is the same as:
        # > z_loss = torch.exp(logits)
        # > z_loss = torch.sum(z_loss, dim=-1)
        # > z_loss = torch.log(z_loss) ** 2.0
        z_loss = torch.logsumexp(logits, dim=-1) ** 2.0  # [B, T, n_exp]

        # sum over all tokens and divide by total number of tokens
        return torch.mean(z_loss)

    def get_capacity(self, tokens_per_batch):
        # expert capacity is given by (tokens_per_batch / num_experts) * capacity_factor
        # see eq (3) in Switch Transformer (https://arxiv.org/abs/2101.03961)
        capacity_factor = self.train_capacity if self.training else self.eval_capacity
        capacity = math.floor(self.top_k * capacity_factor * tokens_per_batch / self.n_exp)
        capacity += capacity % 2 # make sure capacity is an even number
        capacity = max(capacity, self.min_capacity) # use min capacity
        assert capacity > 0
        return int(capacity)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class MLPExperts(nn.Module):
    """
    implementation of multiple MLP-based experts that can process input
    in batch -- based upon ColossalAI OpenMoE but simple, has optional bias, and
    uses a bmm instead of a loop over a mm for each expert to improve efficiency
    link: https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/moe/experts.py
    """
    def __init__(self, config):
        # TODO: add param init
        super().__init__()
        self.bias = config.bias

        self.c_fc = nn.Parameter(torch.empty(config.n_exp, config.n_embd, 4 * config.n_embd))
        self.c_proj = nn.Parameter(torch.empty(config.n_exp, 4 * config.n_embd, config.n_embd))
        self.fc_bias = nn.Parameter(torch.empty(config.n_exp, 1, 4 * config.n_embd)) if self.bias else None
        self.proj_bias = nn.Parameter(torch.empty(config.n_exp, 1, config.n_embd)) if self.bias else None
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
    

    def forward(self, x):
        x = torch.bmm(x, self.c_fc)
        if self.bias:
            x += self.fc_bias
        x = self.gelu(x)
        x = torch.bmm(x, self.c_proj)
        if self.bias:
            x += self.proj_bias
        x = self.dropout(x)
        return x
    
    @torch.no_grad()
    def compute_svd_stats(self):
        """
        Computes the largest singular value for each expert matrix to monitor 
        rank collapse and representation drift.
        """
        # torch.linalg.svdvals operates on the trailing two dimensions
        # svd_fc shape: [n_exp, min(n_embd, 4 * n_embd)] -> [n_exp, n_embd]
        svd_fc = torch.linalg.svdvals(self.c_fc.detach())
        svd_proj = torch.linalg.svdvals(self.c_proj.detach())
        
        # Extract the largest singular value (index 0) for each expert
        top_svd_fc = svd_fc[:, 0]      # Shape: [n_exp]
        top_svd_proj = svd_proj[:, 0]  # Shape: [n_exp]
        
        # Combine them (e.g., average the top singular value across both matrices per expert)
        top_singular_values = (top_svd_fc + top_svd_proj) / 2.0
        
        return top_singular_values
    
    @torch.no_grad()
    def compute_top_singular_vectors(self):
        """
        Computes top right-singular vector for batched expert weights.
        self.c_fc shape: (n_exp, n_embd, 4 * n_embd)
        """
        # 1. Transpose the weights to (n_exp, 4 * n_embd, n_embd) 
        # so we perform SVD on the transformation matrix (input_dim -> output_dim)
        W = self.c_fc.transpose(1, 2).to(dtype=torch.float32)
        
        # 2. PyTorch linalg.svd processes batch dimension automatically
        # W shape: (n_exp, output_dim, input_dim)
        # U: (n_exp, out, out), S: (n_exp, min_dim), Vh: (n_exp, min_dim, input_dim)
        _, _, Vh = torch.linalg.svd(W, full_matrices=False)
        
        # 3. Vh rows are the right-singular vectors. 
        # The top vector for each expert is the first row of Vh.
        return Vh[:, 0, :] # Returns (n_exp, n_embd)

class MOELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.router = Router(config) # (noisy) top k router
        self.experts = MLPExperts(config) # group of MLPs (experts)

        # --- Tracking State ---
        # --- Tracking State ---
        self.register_buffer('running_expert_counts', torch.zeros(config.n_exp, dtype=torch.long))
        self.register_buffer('running_entropy_sum', torch.zeros(1, dtype=torch.float32))
        self.register_buffer('total_tracked_tokens', torch.zeros(1, dtype=torch.long))

    def forward(self, x: torch.Tensor):
        B, T, n_embd = x.size() 
        num_tokens = (B * T)

        # 1. Get routing outputs
        used_capacity, exp_weight, exp_mask = self.router(x)
        
        # 2. Cache weights for your stats loop tracking
        # Assumes exp_weight or a derivative represents the assignment probabilities
    
        with torch.no_grad():
            # Reconstruct the true dimensions: (num_tokens, n_exp, capacity)
            # The -1 dynamically handles whatever capacity limit the router applies
            mask_3d = exp_mask.view(num_tokens, self.config.n_exp, -1)
            weights_3d = exp_weight.view(num_tokens, self.config.n_exp, -1)
            
            # 1. Track Expert Assignments (Exact Routing)
            # Sum across the token dimension (0) and capacity dimension (2)
            # This returns exactly the number of active tokens assigned to each of the 8 experts
            batch_counts = mask_3d.sum(dim=(0, 2))
            
            # 2. Track Entropy (True Distribution)
            # Sum the weights across the capacity dimension to get (num_tokens, n_exp)
            raw_probs = weights_3d.sum(dim=-1) 
            
            # Normalize to ensure valid probabilities (critical for dropped/padded tokens)
            probs = raw_probs / (raw_probs.sum(dim=-1, keepdim=True) + 1e-10)
            
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            
            self.total_tracked_tokens += num_tokens
            self.running_entropy_sum += self.router.latest_kl_div.detach()
            self.running_expert_counts += batch_counts

        # ... rest of your forward path processing ...
        x = x.view(num_tokens, n_embd)
        exp_batches = exp_mask.permute(1, 2, 0).type_as(x) @ x
        exp_out = self.experts(exp_batches) 
        
        exp_weight_flat = exp_weight.view(num_tokens, -1) 
        exp_out_flat = exp_out.view(-1, n_embd) 
        output = exp_weight_flat @ exp_out_flat 
        
        return output.view(B, T, n_embd)
    


class Block(nn.Module):

    def __init__(self, config, use_moe=False):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        if use_moe:
            self.mlp = MOELayer(config)
        else:
            self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    # MoE-related configs 
    n_exp: int = 1 # if n_exp = 1 we just use regular MLP layers
    top_k: int = 2
    use_aux_loss: bool = False # apply auxiliary loss (from Switch Transformer) in router
    use_router_z_loss: bool = False # apply router z loss (from ST-MoE)
    use_noisy_top_k: bool = False
    aux_loss_weight: float = 0.01 # default setting from Switch Transformer (see top of page 8)
    router_z_loss_weight: float = 0.001 # default setting from ST-MoE (see page 8 eq. 6)
    train_capacity: float = 1.25  # default setting from ST-MoE (see top of page 6)
    eval_capacity: float = 2.0
    min_capacity: int = 4  # minimum batch size to send to any single expert
    stride: int = 2 # one in every stride layers are converted to an MoE
    use_switch_tfm_init: bool = False  # use weight init scheme from Switch Transformer
    switch_tfm_init_scale: float = 1.0
    router_use_full_prec: bool = False  # use float32 precision in the router


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        if config.n_exp == 1:
            # create normal transformer blocks
            blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        else:
            # create transformer blocks, placing an MoE block every <stride> layers
            blocks = []
            for i in range(config.n_layer):
                # TODO: how to implement this?
                # should we change below to i + 1 ?
                use_moe = (i % config.stride) == 0
                blocks.append(Block(config, use_moe=use_moe))
            blocks = nn.ModuleList(blocks)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = blocks,
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        # optionall use switch transformer special init scheme for experts
        # See pg. 10 here: https://arxiv.org/abs/2101.03961
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('experts.c_proj'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    @torch.no_grad()
    def _init_weights(self, module):
        # optionally use switch transformer-style initialization
        # see page 10 for switch init explanation: https://arxiv.org/abs/2101.03961
        if isinstance(module, nn.Linear):
            if self.config.use_switch_tfm_init:
                scale = self.config.switch_tfm_init_scale

                # linear layers have flipped dimensions in torch
                # size of weights is [out_dim, in_dim] 
                w_fan_in = module.weight.shape[-1]
                w_std = (scale / w_fan_in) ** 0.5
                torch.nn.init.trunc_normal_(
                    module.weight,
                    mean=0.0,
                    std=w_std,
                    a=-2*w_std,
                    b=2*w_std,
                )
            else:
                # perform standard (normal) initialization of weights
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            # always initialize bias to zero
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, MLPExperts):
            # we have to init expert weights manually because
            # nn.Parameter is not a type of module in torch
            if self.config.use_switch_tfm_init:
                scale = self.config.switch_tfm_init_scale

                c_fc_fan_in = module.c_fc.shape[-2]
                c_fc_std = (scale / c_fc_fan_in) ** 0.5
                torch.nn.init.trunc_normal_(
                    module.c_fc,
                    mean=0.0,
                    std=c_fc_std,
                    a=-2*c_fc_std,
                    b=2*c_fc_std,
                )

                c_proj_fan_in = module.c_proj.shape[-2]
                c_proj_std = (scale / c_proj_fan_in) ** 0.5
                torch.nn.init.trunc_normal_(
                    module.c_proj,
                    mean=0.0,
                    std=c_proj_std,
                    a=-2*c_proj_std,
                    b=2*c_proj_std,
                )
            else:
                # perform standard (normal) initialization of weights
                torch.nn.init.normal_(module.c_fc, mean=0.0, std=0.02)
                torch.nn.init.normal_(module.c_proj, mean=0.0, std=0.02)

            # bias is always initialized to zero
            if module.fc_bias is not None:
                torch.nn.init.zeros_(module.fc_bias)
                torch.nn.init.zeros_(module.proj_bias)
        elif isinstance(module, nn.Embedding):
            # just use standard initialization scheme for embedding always
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # 1. Explicitly initialize x before the loop
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        # 2. Use a standard loop that is easier for Dynamo to track
        for i in range(len(self.transformer.h)):
            x = self.transformer.h[i](x)
            
        x = self.transformer.ln_f(x)

        # --- HARVEST LOSSES AND METRICS FROM LAYERS ---
        aux_loss_val = torch.tensor(0.0, device=device)
        z_loss_val = torch.tensor(0.0, device=device)
        
        # Lists to gather router stats across layers
        router_probs_list = []
        max_logits = []
        avg_logits = []
        total_dropped = 0

        if self.config.n_exp > 1:
            for block in self.transformer.h:
                if hasattr(block, 'mlp') and hasattr(block.mlp, 'router'):
                    router = block.mlp.router
                    
                    # 1. Harvest Losses
                    aux_loss_val += router.aux_loss
                    z_loss_val += router.z_loss
                    
                    # 2. Harvest Metrics
                    router_probs_list.append(router.latest_probs)
                    max_logits.append(router.latest_max_logit)
                    avg_logits.append(router.latest_avg_logit)
                    total_dropped += router.latest_dropped_tokens

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
            # Combine harvested losses
            loss += self.config.aux_loss_weight * aux_loss_val
            loss += self.config.router_z_loss_weight * z_loss_val
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        # Compute averages for logging
        max_logit = torch.stack(max_logits).mean() if max_logits else torch.tensor(0.0, device=device)
        avg_logit = torch.stack(avg_logits).mean() if avg_logits else torch.tensor(0.0, device=device)
        
        return logits, loss, aux_loss_val, z_loss_val, max_logit, avg_logit, router_probs_list, total_dropped

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert not 'moe' in model_type, "Pretrained checkpoints not available for MoE"
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, optimizer_choice, weight_decay, learning_rate, betas, device_type, momentum, nesterov):
        # TODO: add expert config
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        # add an extra check for "bias" string to account for bias terms in MoE layers
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 ]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        optimizer_classes = {
            'adamw': torch.optim.AdamW,
            'adam_vanilla': torch.optim.Adam,
            'sgd': torch.optim.SGD, 
            'adafactor': torch.optim.Adafactor, # Note: Ensure this is correctly imported/patched 
            'adagrad': torch.optim.Adagrad,
            'sparse_adam': torch.optim.SparseAdam   # PyTorch native LazyAdam implementation
        }

        if optimizer_choice not in optimizer_classes:
            raise ValueError(f"Unrecognized optimizer_choice: {optimizer_choice}")
        
        optim_class = optimizer_classes[optimizer_choice]

        fused_available = 'fused' in inspect.signature(optim_class).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        print(f"using fused {optimizer_choice}: {use_fused}")

        # In your training loop initialization:
        if optimizer_choice == 'adamw':
            optimizer = optim_class(
                optim_groups, 
                lr=learning_rate, 
                betas=betas, 
                **extra_args
            )

        elif optimizer_choice == 'adam_vanilla':
            optimizer = optim_class(
                optim_groups, 
                lr=learning_rate, 
                betas=betas, 
                **extra_args
            )

        elif optimizer_choice == 'sgd':

            sgd_args = {k: v for k, v in extra_args.items() if k in ['fused']}
            optimizer = optim_class(
                optim_groups, 
                lr=learning_rate, 
                momentum=momentum,
                nesterov=nesterov,
                **sgd_args
            )

        elif optimizer_choice == 'adafactor': 
            optimizer = optim_class(
                optim_groups, 
                lr=learning_rate, 
                betas=betas, 
                **extra_args
            )
            
        elif optimizer_choice == 'adagrad':
            optimizer = optim_class(
                optim_groups,
                lr=learning_rate,
                **extra_args
            )
            
        elif optimizer_choice == 'sparse_adam':
            optimizer = optim_class(
                optim_groups,
                lr=learning_rate,
                betas=betas,
                **extra_args
            )
        
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx   
