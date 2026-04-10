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
        self.expert_balance_metrics = []
        self.neuron_balance_metrics = []
    
    def reset_aux_loss(self):
        self.aux_loss = []
    
    def reset_router_z_loss(self):
        self.router_z_loss = []

    def reset_reinforce_log_prob(self):
        self.reinforce_log_prob = []

    def reset_capacity_stats(self):
        self.capacity_stats = []

    def reset_expert_balance_metrics(self):
        self.expert_balance_metrics = []

    def reset_neuron_balance_metrics(self):
        self.neuron_balance_metrics = []
    
    def add_aux_loss(self, loss):
        self.aux_loss.append(loss)
    
    def add_router_z_loss(self, loss):
        self.router_z_loss.append(loss)

    def add_reinforce_log_prob(self, log_prob):
        self.reinforce_log_prob.append(log_prob)

    def add_capacity_stats(self, stats):
        self.capacity_stats.append(stats)

    def add_expert_balance_metrics(self, layer_name, metrics_dict):
        self.expert_balance_metrics.append((layer_name, metrics_dict))

    def add_neuron_balance_metrics(self, layer_name, metrics_dict):
        self.neuron_balance_metrics.append((layer_name, metrics_dict))
    
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

    def get_expert_balance_metrics(self):
        return self._get_balance_metrics(self.expert_balance_metrics)

    def get_neuron_balance_metrics(self):
        return self._get_balance_metrics(self.neuron_balance_metrics)

    def _get_balance_metrics(self, entries):
        if not entries:
            return None

        layerwise = {}
        summary = {}
        counts = {}
        for layer_name, metrics_dict in entries:
            layer_type = self._layer_type_from_name(layer_name)
            layerwise[layer_name] = {}
            for key, value in metrics_dict.items():
                cpu_value = self._to_cpu_value(value)
                layerwise[layer_name][key] = cpu_value
                if self._is_scalar_value(cpu_value):
                    summary.setdefault(layer_type, {})
                    counts.setdefault(layer_type, {})
                    summary[layer_type][key] = summary[layer_type].get(key, 0.0) + float(cpu_value)
                    counts[layer_type][key] = counts[layer_type].get(key, 0) + 1

        for layer_type, metrics_dict in summary.items():
            for key in metrics_dict:
                metrics_dict[key] /= counts[layer_type][key]

        return {
            "summary": summary,
            "layerwise": layerwise,
        }

    @staticmethod
    def _layer_type_from_name(layer_name):
        return layer_name.split(".")[-1]

    @staticmethod
    def _to_cpu_value(value):
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "dim"):
            if value.dim() == 0:
                return value.item()
            return value.cpu()
        return value

    @staticmethod
    def _is_scalar_value(value):
        return isinstance(value, (int, float, bool))

MANAGER = MOEManager()
