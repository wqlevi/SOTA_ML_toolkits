import torch
class FixPointLayer(nn.Module):
    def __init__(self, out_features, tol=1e-4, max_iter = 50):
        super().__init__()
        self.linear = nn.Linear(out_features, out_features, bias = False)
        self.tol = tol
        self.max_iter = max_iter
    def forward(self, x):
        z = torch.zeros_like(x)
        self.iteration = 0

        while self.iteration < self.max_iter:
            z_next = torch.tanh(self.linear(z) + x)
            self.err = torch.norm(z_next-z) # end if for given random X, the fix point is found as to Z
            z = z_next
            self.iteration += 1
            if self.err < self.tol:
                break
        return z
