"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import itertools
import math
import inspect
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from manager import MANAGER

# Instrumentation definitions for future loss-free balancing experiments:
# - expert usage = token-expert assignments per token
# - neuron usage = positive preactivation rate before GELU
# - EMA buffers are local-process instrumentation under DDP
# - future bias hooks would naturally attach to router logits and pre-GELU activations

def _safe_coefficient_of_variation(values: torch.Tensor, eps: float = 1e-9):
    values = values.float()
    return values.std(unbiased=False) / values.mean().abs().clamp_min(eps)

def _normalized_entropy(values: torch.Tensor, eps: float = 1e-9):
    values = values.float().clamp_min(0.0)
    total = values.sum()
    if total.item() <= 0:
        return values.new_tensor(0.0)
    probs = values / total
    if probs.numel() <= 1:
        return probs.new_tensor(1.0)
    entropy = -(probs * torch.log(probs.clamp_min(eps))).sum()
    return entropy / math.log(probs.numel())

def _min_to_max_ratio(values: torch.Tensor, eps: float = 1e-9):
    values = values.float()
    return values.min() / values.max().clamp_min(eps)

def _normalized_l1_distance(values_a: torch.Tensor, values_b: torch.Tensor, eps: float = 1e-9):
    values_a = values_a.float()
    values_b = values_b.float()
    probs_a = values_a / values_a.sum().clamp_min(eps)
    probs_b = values_b / values_b.sum().clamp_min(eps)
    return torch.abs(probs_a - probs_b).sum()

def _update_ema_buffer(buffer: torch.Tensor, value: torch.Tensor, beta: float):
    buffer.mul_(beta).add_(value.detach().to(buffer.dtype), alpha=1.0 - beta)

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
        self.router_selection = config.router_selection
        assert self.router_selection in {'topk', 'sample_without_replacement'}
        self.sample_routing_eval = config.sample_routing_eval
        self.use_noisy_top_k = config.use_noisy_top_k
        self.train_capacity = config.train_capacity
        self.eval_capacity = config.eval_capacity
        self.min_capacity = config.min_capacity
        self.router_use_full_prec = config.router_use_full_prec
        self.usage_ema_beta = config.usage_ema_beta
        self.use_loss_free_expert_balance = config.use_loss_free_expert_balance
        self.expert_balance_beta = config.expert_balance_beta
        self.expert_balance_eta = config.expert_balance_eta
        self.expert_balance_lambda_min = config.expert_balance_lambda_min
        self.expert_balance_lambda_max = config.expert_balance_lambda_max
        self.expert_balance_warmup_steps = config.expert_balance_warmup_steps
        self.expert_balance_use_accepted_usage = config.expert_balance_use_accepted_usage
        self.expert_balance_target = config.expert_balance_target
        self.layer_name = None

        # auxiliary / load balancing loss settings
        self.use_aux_loss = config.use_aux_loss
        self.use_router_z_loss = config.use_router_z_loss
        self.use_reinforce_routing = config.use_reinforce_routing
        self.use_straight_through_routing = config.use_straight_through_routing
        self.register_buffer("expert_usage_ema", torch.zeros(self.n_exp))
        self.register_buffer("lambda_expert", torch.zeros(self.n_exp))
        self.register_buffer("balance_step", torch.zeros((), dtype=torch.long))
        self.collect_balance_metrics = True

        # linear projection for (noisy) softmax gating
        # no bias is used, see page 4 eq (4) in (https://arxiv.org/abs/1701.06538)
        self.w_g = nn.Linear(config.n_embd, config.n_exp, bias=False)
        # Noisy routing only applies to deterministic top-k routing.
        self.w_noise = nn.Linear(config.n_embd, config.n_exp, bias=False) if (
            self.use_noisy_top_k and self.router_selection == 'topk'
        ) else None
        if self.use_reinforce_routing and self.use_straight_through_routing:
            raise ValueError("REINFORCE routing and straight-through routing are mutually exclusive")
        if self.use_straight_through_routing and self.router_selection != 'sample_without_replacement':
            raise ValueError("Straight-through routing is only supported with sample_without_replacement routing")
        if self.use_reinforce_routing:
            if self.router_selection != 'sample_without_replacement':
                raise ValueError("REINFORCE routing is only supported with sample_without_replacement routing")
            if self.top_k > 3:
                raise ValueError("Exact REINFORCE routing is only supported for top_k <= 3")
            self.reinforce_permutations = tuple(itertools.permutations(range(self.top_k)))
        else:
            self.reinforce_permutations = ()
    
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
            routing_logits = logits - self.lambda_expert if self.use_loss_free_expert_balance else logits
            if self.w_noise is not None:
                # optionally add noise into the router
                noise = F.softplus(self.w_noise(x))
                noise *= torch.randn_like(noise)
                logits += noise
                routing_logits += noise

            # router z loss, computed on logits (before softmax)
            # this loss prevents router logits from becoming too large
            if self.use_router_z_loss:
                z_loss = self.compute_router_z_loss(logits)
                MANAGER.add_router_z_loss(z_loss)

            route_mode = self.router_selection
            if route_mode == 'sample_without_replacement' and not self.training and not self.sample_routing_eval:
                route_mode = 'topk'
            use_straight_through = self.use_straight_through_routing and route_mode == 'sample_without_replacement'

            if route_mode == 'topk':
                # find top k experts for each token
                top_k_logits, top_k_indices = routing_logits.topk(self.top_k, dim=-1) # [B, T, k]

                # Shazeer et al (https://arxiv.org/abs/1701.06538) does only top-k
                dispatch_probs = torch.full_like(routing_logits, float('-inf'))  # [B, T, n_exp]
                dispatch_probs.scatter_(-1, top_k_indices, top_k_logits)
                dispatch_probs = F.softmax(dispatch_probs, dim=-1)
                aux_probs = dispatch_probs
                chosen_indices = top_k_indices
            else:
                # Sample k distinct experts without replacement from the full router distribution.
                full_probs = F.softmax(routing_logits, dim=-1)
                flat_probs = full_probs.reshape(-1, self.n_exp).float()
                chosen_indices = torch.multinomial(flat_probs, num_samples=self.top_k, replacement=False)
                chosen_indices = chosen_indices.view(B, T, self.top_k)

                # Capacity overflow uses slot order, so sort sampled experts into a canonical action.
                chosen_indices, chosen_probs = self.canonicalize_sampled_experts(full_probs, chosen_indices)
                if self.use_reinforce_routing and self.training:
                    MANAGER.add_reinforce_log_prob(
                        self.compute_sorted_action_log_prob(full_probs, chosen_indices)
                    )
                chosen_probs = chosen_probs / chosen_probs.sum(dim=-1, keepdim=True).clamp_min(
                    torch.finfo(chosen_probs.dtype).eps
                )

                dispatch_probs = torch.zeros_like(full_probs)
                dispatch_probs.scatter_(-1, chosen_indices, chosen_probs)
                aux_probs = full_probs

            # compute auxiliary load balancing loss
            # this loss encourages equal probability assigned to each expert
            # and equal load balancing of tokens assigned to each expert
            if self.use_aux_loss:
                aux_loss = self.compute_aux_loss(aux_probs, chosen_indices)
                MANAGER.add_aux_loss(aux_loss)

            # compute expert capacity
            exp_capacity = self.get_capacity(num_tokens)

            # make a multi-hot mask of chosen experts, size [B, T, n_exp]
            # entries are 0 if expert not chosen and 1 if expert chosen
            exp_mask = F.one_hot(chosen_indices, num_classes=self.n_exp)  # [B, T, k, n_exp]
            exp_mask = exp_mask.view(num_tokens, self.top_k, self.n_exp)  # [B * T, k, n_exp]
            exp_mask = exp_mask.permute(1, 0, 2) # [k, B * T, n_exp]
            attempted_assignments = exp_mask.sum(dim=(0, 1))

            # compute cumulative sum of each token over experts, this stores
            # the index of each token within the batch of each expert
            # NOTE: cumsum should count all top-1 first, top-2 second, etc.
            # so that we prioritize top experts when dropping tokens (this is
            # done by putting k dimension first for the reshape operation)
            exp_rank = exp_mask.reshape(self.top_k * num_tokens, self.n_exp)  # [k * B * T, n_exp]
            exp_rank = torch.cumsum(exp_rank, dim=0) - 1  # cumulative sum of expert selections [k * B * T, n_exp]
            exp_rank = exp_rank.reshape(self.top_k, num_tokens, self.n_exp)  # [k, B * T, n_exp]

            # mask out (set to zero) entries that go beyond expert capacity
            # compute amount of used capacity by taking a sum over mask
            exp_mask *= torch.lt(exp_rank, exp_capacity) # [k, B * T, n_exp]
            used_capacity = torch.sum(exp_mask, dim=(0, 1)) # [n_exp]
            capacity_util = used_capacity.float() / exp_capacity
            MANAGER.add_capacity_stats({
                'frac_at_limit': (used_capacity == exp_capacity).float().mean().detach(),
                'mean_utilization': capacity_util.mean().detach(),
                'max_utilization': capacity_util.max().detach(),
                'dropped_fraction': (1.0 - used_capacity.sum().float() / (self.top_k * num_tokens)).detach(),
            })
            with torch.no_grad():
                expert_usage_attempted, expert_usage_current, expert_usage_target, expert_balance_metrics = self.build_expert_balance_metrics(
                    attempted_assignments=attempted_assignments,
                    accepted_assignments=used_capacity,
                    exp_capacity=exp_capacity,
                    num_tokens=num_tokens,
                )
                if self.collect_balance_metrics and self.layer_name is not None:
                    MANAGER.add_expert_balance_metrics(self.layer_name, expert_balance_metrics)

            # mask rank to only include tokens that are selected
            # perform a sum so each row only contains index of token
            # for the expert that is selected in that row
            # result is a matrix that contains the position of each token
            # in the batch of its corresponding expert
            exp_rank = torch.sum(exp_mask * exp_rank, dim=-1)  # [k, B * T]

            # mask probabilities to only include selected experts
            dispatch_probs = dispatch_probs.view(num_tokens, self.n_exp)
            exp_mask = exp_mask.to(dispatch_probs.dtype)
            exp_weights_hard = exp_mask * dispatch_probs[None, :] # [k, B * T, n_exp]
            if use_straight_through:
                kept_mask = exp_mask.sum(dim=0) # [B * T, n_exp]
                dispatch_soft_selected = full_probs.reshape(num_tokens, self.n_exp) * kept_mask
                dispatch_soft_selected = dispatch_soft_selected / dispatch_soft_selected.sum(
                    dim=-1, keepdim=True
                ).clamp_min(torch.finfo(dispatch_soft_selected.dtype).eps)
                exp_weights_soft = exp_mask * dispatch_soft_selected.unsqueeze(0)
                exp_weights = exp_weights_hard.detach() - exp_weights_soft.detach() + exp_weights_soft
            else:
                exp_weights = exp_weights_hard

            # convert rank into one-hot vectors over the available capacity
            # stores the position of each token within the capacity of the selected expert
            exp_rank_sc = F.one_hot(exp_rank, num_classes=exp_capacity) # [k, B * T, exp_capacity]

            # create a vector that stores, for each token, the weight of selected
            # experts at token's position in the capacity of that expert
            # size of tensor is [B * T, n_exp, exp_capacity]
            cb_weight = torch.sum(exp_weights.unsqueeze(3) * exp_rank_sc.unsqueeze(2), dim=0)
            sec_mask = cb_weight.bool() # binary mask of selected experts for each token
            return used_capacity, cb_weight, sec_mask

    def canonicalize_sampled_experts(self, full_probs: torch.Tensor, chosen_indices: torch.Tensor):
        chosen_probs = full_probs.gather(-1, chosen_indices)
        chosen_indices, index_order = chosen_indices.sort(dim=-1)
        chosen_probs = chosen_probs.gather(-1, index_order)
        chosen_probs, prob_order = torch.sort(chosen_probs, dim=-1, descending=True, stable=True)
        chosen_indices = chosen_indices.gather(-1, prob_order)
        return chosen_indices, chosen_probs

    def compute_sorted_action_log_prob(self, full_probs: torch.Tensor, chosen_indices: torch.Tensor):
        flat_probs = full_probs.reshape(-1, self.n_exp).float()
        flat_indices = chosen_indices.reshape(-1, self.top_k)
        eps = torch.finfo(flat_probs.dtype).eps

        perm_log_probs = []
        for perm in self.reinforce_permutations:
            permuted_indices = flat_indices[:, list(perm)]
            permuted_probs = flat_probs.gather(1, permuted_indices)
            remaining = torch.ones(flat_probs.size(0), device=flat_probs.device, dtype=flat_probs.dtype)
            log_q = torch.zeros_like(remaining)
            for step in range(self.top_k):
                p_step = permuted_probs[:, step].clamp_min(eps)
                log_q = log_q + torch.log(p_step) - torch.log(remaining.clamp_min(eps))
                remaining = remaining - p_step
            perm_log_probs.append(log_q)

        log_q = torch.logsumexp(torch.stack(perm_log_probs, dim=0), dim=0)
        return log_q.view(*chosen_indices.shape[:2])

    def build_expert_balance_metrics(self, attempted_assignments, accepted_assignments, exp_capacity, num_tokens):
        attempted_assignments = attempted_assignments.float()
        accepted_assignments = accepted_assignments.float()
        expert_usage_attempted = attempted_assignments / num_tokens
        expert_usage_accepted = accepted_assignments / num_tokens
        expert_usage_target_value = self.expert_balance_target if self.expert_balance_target is not None else self.top_k / self.n_exp
        expert_usage_target = expert_usage_attempted.new_full((self.n_exp,), expert_usage_target_value)
        expert_usage_current = expert_usage_accepted if self.expert_balance_use_accepted_usage else expert_usage_attempted

        if self.training:
            if self.use_loss_free_expert_balance:
                if int(self.balance_step.item()) >= self.expert_balance_warmup_steps:
                    self.expert_usage_ema.mul_(1.0 - self.expert_balance_beta).add_(
                        expert_usage_current.detach(),
                        alpha=self.expert_balance_beta,
                    )
                    lambda_update = self.lambda_expert + self.expert_balance_eta * (self.expert_usage_ema - expert_usage_target)
                    self.lambda_expert.copy_(lambda_update.clamp(self.expert_balance_lambda_min, self.expert_balance_lambda_max))
                self.balance_step.add_(1)
            elif self.collect_balance_metrics:
                _update_ema_buffer(self.expert_usage_ema, expert_usage_attempted, self.usage_ema_beta)

        mean_attempted = expert_usage_attempted.mean()
        mean_accepted = expert_usage_accepted.mean()
        fraction_experts_at_capacity = (accepted_assignments == exp_capacity).float().mean()
        assignment_drop_fraction = 1.0 - accepted_assignments.sum() / (self.top_k * num_tokens)

        metrics = {
            "expert_usage_attempted": expert_usage_attempted.detach(),
            "expert_usage_accepted": expert_usage_accepted.detach(),
            "expert_usage_current": expert_usage_current.detach(),
            "expert_usage_target": expert_usage_target.mean().detach(),
            "expert_usage_ema": self.expert_usage_ema.detach(),
            "lambda_expert": self.lambda_expert.detach(),
            "lambda_expert_min": self.lambda_expert.min().detach(),
            "lambda_expert_mean": self.lambda_expert.mean().detach(),
            "lambda_expert_max": self.lambda_expert.max().detach(),
            "lambda_expert_std": self.lambda_expert.std(unbiased=False).detach(),
            "expert_usage_ema_mean": self.expert_usage_ema.mean().detach(),
            "expert_usage_ema_std": self.expert_usage_ema.std(unbiased=False).detach(),
            "l1_expert_usage_ema_vs_target": torch.abs(self.expert_usage_ema - expert_usage_target).sum().detach(),
            "l1_expert_usage_current_vs_target": torch.abs(expert_usage_current - expert_usage_target).sum().detach(),
            "fraction_experts_at_capacity": fraction_experts_at_capacity.detach(),
            "assignment_drop_fraction": assignment_drop_fraction.detach(),
            "max_expert_usage_attempted_over_mean": (
                expert_usage_attempted.max() / mean_attempted.clamp_min(1e-9)
            ).detach(),
            "max_expert_usage_accepted_over_mean": (
                expert_usage_accepted.max() / mean_accepted.clamp_min(1e-9)
            ).detach(),
            "min_to_max_expert_usage_attempted": _min_to_max_ratio(expert_usage_attempted).detach(),
            "min_to_max_expert_usage_accepted": _min_to_max_ratio(expert_usage_accepted).detach(),
            "cv_expert_usage_attempted": _safe_coefficient_of_variation(expert_usage_attempted).detach(),
            "cv_expert_usage_accepted": _safe_coefficient_of_variation(expert_usage_accepted).detach(),
            "entropy_expert_usage_attempted_normalized": _normalized_entropy(expert_usage_attempted).detach(),
            "entropy_expert_usage_accepted_normalized": _normalized_entropy(expert_usage_accepted).detach(),
            "l1_expert_usage_attempted_vs_accepted": _normalized_l1_distance(
                expert_usage_attempted,
                expert_usage_accepted,
            ).detach(),
        }
        return expert_usage_attempted, expert_usage_current, expert_usage_target, metrics
    
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
        self.usage_ema_beta = config.usage_ema_beta
        self.layer_name = None
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("neuron_usage_ema", torch.zeros(4 * config.n_embd))
        self.collect_balance_metrics = True

    def forward(self, x):
        preact = self.c_fc(x)
        with torch.no_grad():
            if self.collect_balance_metrics:
                neuron_usage_rate = (preact > 0).float().mean(dim=(0, 1))
                if self.training:
                    _update_ema_buffer(self.neuron_usage_ema, neuron_usage_rate, self.usage_ema_beta)
                if self.layer_name is not None:
                    MANAGER.add_neuron_balance_metrics(self.layer_name, {
                        "neuron_usage_rate": neuron_usage_rate.detach(),
                        "neuron_usage_target": neuron_usage_rate.new_tensor(0.5).detach(),
                        "neuron_usage_ema": self.neuron_usage_ema.detach(),
                        "mean_neuron_usage_rate": neuron_usage_rate.mean().detach(),
                        "std_neuron_usage_rate": neuron_usage_rate.std(unbiased=False).detach(),
                        "min_neuron_usage_rate": neuron_usage_rate.min().detach(),
                        "median_neuron_usage_rate": neuron_usage_rate.median().detach(),
                        "max_neuron_usage_rate": neuron_usage_rate.max().detach(),
                        "fraction_neurons_at_zero": (neuron_usage_rate < 1e-5).float().mean().detach(),
                        "cv_neuron_usage_rate": _safe_coefficient_of_variation(neuron_usage_rate).detach(),
                        "entropy_neuron_usage_rate_normalized": _normalized_entropy(neuron_usage_rate).detach(),
                    })
        x = self.gelu(preact)
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
        self.usage_ema_beta = config.usage_ema_beta
        self.layer_name = None

        self.c_fc = nn.Parameter(torch.empty(config.n_exp, config.n_embd, 4 * config.n_embd))
        self.c_proj = nn.Parameter(torch.empty(config.n_exp, 4 * config.n_embd, config.n_embd))
        self.fc_bias = nn.Parameter(torch.empty(config.n_exp, 1, 4 * config.n_embd)) if self.bias else None
        self.proj_bias = nn.Parameter(torch.empty(config.n_exp, 1, config.n_embd)) if self.bias else None
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("neuron_usage_ema", torch.zeros(config.n_exp, 4 * config.n_embd))
        self.collect_balance_metrics = True
    

    def forward(self, x, used_capacity=None):
        preact = torch.bmm(x, self.c_fc)
        if self.bias:
            preact += self.fc_bias
        with torch.no_grad():
            if self.collect_balance_metrics and used_capacity is not None:
                exp_capacity = preact.size(1)
                slot_idx = torch.arange(exp_capacity, device=preact.device).unsqueeze(0)
                valid_slots = slot_idx < used_capacity.unsqueeze(1)
                active = (preact > 0).float()
                valid_mask = valid_slots.unsqueeze(-1).float()
                valid_counts = valid_slots.sum(dim=1, keepdim=True).float().clamp_min(1.0)
                neuron_usage_rate = (active * valid_mask).sum(dim=1) / valid_counts
                if self.training:
                    _update_ema_buffer(self.neuron_usage_ema, neuron_usage_rate, self.usage_ema_beta)
                if self.layer_name is not None:
                    per_expert_mean = neuron_usage_rate.mean(dim=-1)
                    flat_usage = neuron_usage_rate.reshape(-1)
                    MANAGER.add_neuron_balance_metrics(self.layer_name, {
                        "neuron_usage_rate": neuron_usage_rate.detach(),
                        "neuron_usage_target": neuron_usage_rate.new_tensor(0.5).detach(),
                        "neuron_usage_ema": self.neuron_usage_ema.detach(),
                        "mean_neuron_usage_rate": flat_usage.mean().detach(),
                        "std_neuron_usage_rate": flat_usage.std(unbiased=False).detach(),
                        "min_neuron_usage_rate": flat_usage.min().detach(),
                        "median_neuron_usage_rate": flat_usage.median().detach(),
                        "max_neuron_usage_rate": flat_usage.max().detach(),
                        "fraction_neurons_at_zero": (flat_usage < 1e-5).float().mean().detach(),
                        "cv_neuron_usage_rate": _safe_coefficient_of_variation(flat_usage).detach(),
                        "entropy_neuron_usage_rate_normalized": _normalized_entropy(flat_usage).detach(),
                        "per_expert_mean_neuron_usage_rate": per_expert_mean.detach(),
                        "min_expert_mean_neuron_usage_rate": per_expert_mean.min().detach(),
                        "median_expert_mean_neuron_usage_rate": per_expert_mean.median().detach(),
                        "max_expert_mean_neuron_usage_rate": per_expert_mean.max().detach(),
                    })
        x = self.gelu(preact)
        x = torch.bmm(x, self.c_proj)
        if self.bias:
            x += self.proj_bias
        x = self.dropout(x)
        return x

class MOELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = Router(config) # (noisy) top k router
        self.experts = MLPExperts(config) # group of MLPs (experts)

    def forward(self, x: torch.Tensor):
        B, T, n_embd = x.size() # track original shape of input
        num_tokens = (B * T)

        # pass each token through the router
        used_capacity, exp_weight, exp_mask = self.router(x)

        # flatten out the input
        x = x.view(num_tokens, n_embd)

        # reshape tokens into batches for each expert
        # [n_exp, exp_capacity, B * T] * [B * T, n_embd] -> [n_exp, exp_capacity, n_embd]
        exp_batches = exp_mask.permute(1, 2, 0).type_as(x) @ x

        # compute expert output
        exp_out = self.experts(exp_batches, used_capacity=used_capacity) # [n_exp, exp_capacity, n_embd]

        # aggregate expert outputs based on router weights
        # eq (2) on page 4 of ST-MoE (https://arxiv.org/abs/2202.08906)
        # similar equations are used for other MoE papers
        exp_weight = exp_weight.view(num_tokens, -1) # [B * T, n_exp * exp_capacity]
        exp_out = exp_out.view(-1, n_embd) # [n_exp * exp_capacity, n_embd] 
        output = exp_weight @ exp_out # [B * T, n_embd]
        
        # resize output before return
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
    router_selection: str = 'topk'
    sample_routing_eval: bool = False
    use_aux_loss: bool = False # apply auxiliary loss (from Switch Transformer) in router
    use_router_z_loss: bool = False # apply router z loss (from ST-MoE)
    use_noisy_top_k: bool = False # only used when router_selection == 'topk'
    use_reinforce_routing: bool = False # add an exact REINFORCE term for sorted sampled routing
    use_straight_through_routing: bool = False # use a support-restricted straight-through gradient for sampled routing
    use_loss_free_expert_balance: bool = False # expert-only loss-free balancing controller
    expert_balance_beta: float = 0.99
    expert_balance_eta: float = 0.1
    expert_balance_lambda_min: float = -5.0
    expert_balance_lambda_max: float = 5.0
    expert_balance_warmup_steps: int = 0
    expert_balance_use_accepted_usage: bool = False
    expert_balance_target: Optional[float] = None
    aux_loss_weight: float = 0.01 # default setting from Switch Transformer (see top of page 8)
    router_z_loss_weight: float = 0.001 # default setting from ST-MoE (see page 8 eq. 6)
    reinforce_loss_weight: float = 1.0
    train_capacity: float = 1.25  # default setting from ST-MoE (see top of page 6)
    eval_capacity: float = 2.0
    min_capacity: int = 4  # minimum batch size to send to any single expert
    stride: int = 2 # one in every stride layers are converted to an MoE
    use_switch_tfm_init: bool = False  # use weight init scheme from Switch Transformer
    switch_tfm_init_scale: float = 1.0
    router_use_full_prec: bool = False  # use float32 precision in the router
    usage_ema_beta: float = 0.99  # EMA beta for local-process instrumentation only


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        self.last_capacity_stats = None
        self.last_expert_balance_metrics = None
        self.last_neuron_balance_metrics = None
        self.collect_balance_metrics = True

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

        for i, block in enumerate(blocks):
            if isinstance(block.mlp, MOELayer):
                block.mlp.router.layer_name = f"block_{i}.router"
                block.mlp.experts.layer_name = f"block_{i}.expert_mlp"
            else:
                block.mlp.layer_name = f"block_{i}.dense_mlp"

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
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            token_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction='none',
            ).view(b, t)
            valid_mask = targets != -1
            ce_mean = token_loss[valid_mask].mean()
            loss = ce_mean

            if self.config.n_exp > 1 and self.config.use_reinforce_routing and self.training:
                log_q_total = MANAGER.aggregate_reinforce_log_prob()
                if log_q_total is None:
                    raise RuntimeError("Expected sorted-action log probabilities for REINFORCE routing")
                reinforce_loss = (((token_loss - ce_mean).detach()) * log_q_total)[valid_mask].mean()
                loss = loss + self.config.reinforce_loss_weight * reinforce_loss
                MANAGER.reset_reinforce_log_prob()

            # add the auxiliary load balancing loss and router z loss to the main loss
            if self.config.n_exp > 1 and self.config.use_aux_loss:
                loss = loss + self.config.aux_loss_weight * MANAGER.aggregate_aux_loss()
                MANAGER.reset_aux_loss()
            if self.config.n_exp > 1 and self.config.use_router_z_loss:
                loss = loss + self.config.router_z_loss_weight * MANAGER.aggregate_router_z_loss()
                MANAGER.reset_router_z_loss()
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        self.last_capacity_stats = MANAGER.aggregate_capacity_stats()
        MANAGER.reset_capacity_stats()
        if self.collect_balance_metrics:
            self.last_expert_balance_metrics = MANAGER.get_expert_balance_metrics()
            self.last_neuron_balance_metrics = MANAGER.get_neuron_balance_metrics()
        else:
            self.last_expert_balance_metrics = None
            self.last_neuron_balance_metrics = None
        MANAGER.reset_expert_balance_metrics()
        MANAGER.reset_neuron_balance_metrics()

        return logits, loss

    def set_balance_metric_tracking(self, enabled: bool):
        self.collect_balance_metrics = enabled
        for block in self.transformer.h:
            if isinstance(block.mlp, MOELayer):
                block.mlp.router.collect_balance_metrics = enabled
                block.mlp.experts.collect_balance_metrics = enabled
            else:
                block.mlp.collect_balance_metrics = enabled

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

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # TODO: add expert config
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        # add an extra check for "bias" string to account for bias terms in MoE layers
        decay_params = [p for n, p in param_dict.items() if (p.dim() >= 2 and not n.endswith('bias'))]
        nodecay_params = [p for n, p in param_dict.items() if (p.dim() < 2 or n.endswith('bias'))]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

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
