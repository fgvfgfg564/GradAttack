import torch
import torch.nn as nn

def dct(x: torch.Tensor, dim=0):
    device = x.device
    N = x.size(dim)
    n = torch.arange(N, device=device)
    k = n.clone()
    n, k = torch.meshgrid(n, k)
    w = torch.cos(torch.pi / N * (n + 0.5) * k) # (N(n), N(k))
    y = x.transpose(dim, 0)
    shape = y.shape
    y = y.reshape(N, -1).transpose(0, 1).unsqueeze(1)
    y = torch.matmul(y, w)
    y = torch.sum(y, dim=1)
    f = y.transpose(0, 1).reshape(shape).transpose(dim, 0)
    return f

def dct2(x):
    # (B, C, H, W)
    x = dct(x, 2)
    x = dct(x, 3)
    return x

if __name__ == '__main__':
    u = torch.rand((1, 3, 8,12), device='cpu')
    f1 = dct2(u)
    f2 = dct2(u.transpose(2,3)).transpose(2,3)
    print(f2-f1)