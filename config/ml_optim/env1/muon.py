optimizer_choice = 'muon'
weight_decay = 0.01 
momentum = 0.95        # Muon utilise le Nesterov momentum
ns_steps = 5           # Étapes de Newton-Schulz (paramètre standard de Muon)
grad_clip = 5.0 

# learning rate decay settings
decay_lr = True 
warmup_iters = 500 

# Paramètres séparés selon les tenseurs
learning_rate = 0.02   # Le LR de Muon (pour le 2D) est massivement plus élevé
min_lr = 0.002
adam_lr = 3e-4