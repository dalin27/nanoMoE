class MOEManager:
    """
    basic wrapper class for tracking, storing, and aggregating MoE-related
    losses and routing statistics across multiple MoE layers in the model
    """

    def __init__(self):
        self.aux_loss = []
        self.router_z_loss = []
        self.reinforce_log_prob = []
        self.capacity_stats = []
    
    def reset_aux_loss(self):
        self.aux_loss = []
    
    def reset_router_z_loss(self):
        self.router_z_loss = []

    def reset_reinforce_log_prob(self):
        self.reinforce_log_prob = []

    def reset_capacity_stats(self):
        self.capacity_stats = []
    
    def add_aux_loss(self, loss):
        self.aux_loss.append(loss)
    
    def add_router_z_loss(self, loss):
        self.router_z_loss.append(loss)

    def add_reinforce_log_prob(self, log_prob):
        self.reinforce_log_prob.append(log_prob)

    def add_capacity_stats(self, stats):
        self.capacity_stats.append(stats)
    
    def aggregate_aux_loss(self):
        return sum(self.aux_loss)

    def aggregate_router_z_loss(self):
        return sum(self.router_z_loss)

    def aggregate_reinforce_log_prob(self):
        if not self.reinforce_log_prob:
            return None
        return sum(self.reinforce_log_prob)

    def aggregate_capacity_stats(self):
        if not self.capacity_stats:
            return None
        keys = self.capacity_stats[0].keys()
        return {key: sum(stats[key] for stats in self.capacity_stats) / len(self.capacity_stats) for key in keys}

MANAGER = MOEManager()
