import torch

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                
                state = self.state[p]
                if len(state) == 0:
                    state['momentum'] = torch.zeros_like(g)

                buf = state['momentum']
                buf.mul_(momentum).add_(g)

                # Newton-Schulz orthogonalization logic for 2D matrices
                X = buf.clone()
                if X.size(0) < X.size(1):
                    X = X.t()
                
                for _ in range(ns_steps):
                    A = torch.matmul(X, X.t())
                    X = 1.5 * X - 0.5 * torch.matmul(A, X)
                
                if buf.size(0) < buf.size(1):
                    X = X.t()

                p.data.add_(X, alpha=-lr)

        return loss