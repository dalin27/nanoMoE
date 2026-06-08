import torch
class MOEManager:
    """
    basic wrapper class for tracking, storing, and aggregating auxiliary
    losses across multiple MoE layers in the model
    """

    def __init__(self):
        self.aux_loss = []
        self.router_z_loss = []
        self.max_router_logits = []
        self.mean_router_logits = []
        self.mean_router_logits = []
        self.router_probs = []
        self.dropped_tokens = []
        self.capacity_cv = []
        self.kl_divergence = []
        self.grad_update_cos_sim = []
        self.router_update_cos_sim = []

    #reset
    
    def reset_aux_loss(self):
        self.aux_loss = []
    
    def reset_router_z_loss(self):
        self.router_z_loss = []

    #add 

    def add_max_router_stats(self, logits):
        self.max_router_logits.append(logits.detach())
    
    def add_mean_router_stats(self, logits):
        self.mean_router_logits.append(logits.detach())

    def add_router_probs(self, probs):
        self.router_probs.append(probs.detach())

    def add_dropped_tokens(self, tokens):
        self.dropped_tokens.append(tokens.detach())
    
    @torch._dynamo.disable
    def add_kl_divergence(self, val):
        self.kl_divergence.append(val)

    def add_router_update_cos_sim(self, val):
        self.router_update_cos_sim.append(val)

    def add_grad_update_cos_sim(self, val):
        self.grad_update_cos_sim.append(val)
    
    def add_aux_loss(self, loss):
        self.aux_loss.append(loss)
    
    def add_router_z_loss(self, loss):
        self.router_z_loss.append(loss)

    def add_capacity_cv(self, cv):
        self.capacity_cv.append(cv)

    ### get
    
    def aggregate_aux_loss(self):
        return sum(self.aux_loss)

    def aggregate_router_z_loss(self):
        return sum(self.router_z_loss)
    
    def get_router_stats(self):
        if not self.max_router_logits:
            return 0.0, 0.0
        
        overall_max = max(self.max_router_logits)
        overall_mean = sum(self.mean_router_logits) / len(self.mean_router_logits)

        router_probs = self.router_probs

        if hasattr(self, 'dropped_tokens') and len(self.dropped_tokens) > 0:
            dropped_tokens = sum(self.dropped_tokens).item()
        else:
            dropped_tokens = 0
        
        # Reset for the next forward pass
        self.max_router_logits = []
        self.mean_router_logits = []
        self.router_probs = [] 
        self.dropped_tokens = []
        return overall_max, overall_mean, router_probs, dropped_tokens
    
    def get_and_reset_collapse_metrics(self):
        """
        Retrieves the mean of the new collapse metrics across the macro-batch
        and resets the buffers. Returns 0.0 if no data was logged.
        """
        def get_mean(lst):
            return sum(lst) / len(lst) if lst else 0.0
        
        kl_div = get_mean(self.kl_divergence)
        router_cos_sim = get_mean(self.router_update_cos_sim)
        grad_cos_sim = get_mean(self.grad_update_cos_sim)
        mean_cv = get_mean(self.capacity_cv)

        # Reset buffers
        self.kl_divergence = []
        self.router_update_cos_sim = []
        self.grad_update_cos_sim = []
        self.capacity_cv = []

        return kl_div, router_cos_sim, grad_cos_sim, mean_cv

MANAGER = MOEManager()