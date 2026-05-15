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

    
    def reset_aux_loss(self):
        self.aux_loss = []
    
    def reset_router_z_loss(self):
        self.router_z_loss = []

    def add_max_router_stats(self, logits):
        self.max_router_logits.append(logits)
    
    def add_mean_router_stats(self, logits):
        self.mean_router_logits.append(logits)

    def get_router_stats(self):
        if not self.max_router_logits:
            return 0.0, 0.0
        
        overall_max = max(self.max_router_logits)
        overall_mean = sum(self.mean_router_logits) / len(self.mean_router_logits)
        
        # Reset for the next forward pass
        self.max_router_logits = []
        self.mean_router_logits = []
        return overall_max, overall_mean
    
    def add_aux_loss(self, loss):
        self.aux_loss.append(loss)
    
    def add_router_z_loss(self, loss):
        self.router_z_loss.append(loss)
    
    def aggregate_aux_loss(self):
        return sum(self.aux_loss)

    def aggregate_router_z_loss(self):
        return sum(self.router_z_loss)

MANAGER = MOEManager()