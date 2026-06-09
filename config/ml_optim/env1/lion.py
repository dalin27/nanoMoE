optimizer_choice = 'lion'
weight_decay = 1.0    # Lion nécessite un WD beaucoup plus fort (AdamW: 0.1 -> Lion: 1.0)
beta1 = 0.9
beta2 = 0.99          # 0.99 est le standard recommandé pour Lion
momentum = 0
grad_clip = 1.0       # Généralement rabaissé pour Lion

# learning rate decay settings
decay_lr = True 
warmup_iters = 500 
learning_rate = 1e-4  # Lion nécessite un LR plus faible (AdamW: 6e-4 -> Lion: 1e-4)
min_lr = 1e-5