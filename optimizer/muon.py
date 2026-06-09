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

                # FIX 3: Skip orthogonalization for 1D tensors (biases, layernorms)
                if p.ndim < 2:
                    p.data.add_(buf, alpha=-lr)
                    continue

                # Newton-Schulz orthogonalization logic for >= 2D matrices
                X = buf.clone()
                
                # Ensure compatibility if weights are > 2D (e.g., Conv2D)
                if X.ndim > 2:
                    X = X.view(X.size(0), -1)

                # FIX 2: Force X to be WIDE to compute the smallest possible Gram matrix
                transposed = False
                if X.size(0) > X.size(1):
                    X = X.t()
                    transposed = True
                
                # FIX 1: Normalize to ensure convergence during the iteration
                X = X / (X.norm() + 1e-7)
                
                for _ in range(ns_steps):
                    A = torch.matmul(X, X.t())
                    X = 1.5 * X - 0.5 * torch.matmul(A, X)
                
                if transposed:
                    X = X.t()
                    
                # Restore original shape if flattened
                X = X.view_as(buf)

                p.data.add_(X, alpha=-lr)

        return loss