import torch
import numpy as np
import os
from model import G_rightfunc_class
import utils

print('cwd', os.getcwd())

def run_test():
    device = torch.device('cpu')
    torch.manual_seed(0)
    N = 50
    n = 3
    dtype = torch.float32
    # create small A,B,Theta
    A = torch.rand(n, N, requires_grad=True, device=device, dtype=dtype)
    B = torch.rand(N, n, requires_grad=True, device=device, dtype=dtype)
    Theta = torch.rand(N, 1, requires_grad=True, device=device, dtype=dtype)
    tau = torch.nn.Parameter(1e4 + torch.zeros((1, N+n), device=device, dtype=dtype))

    # create x0 and hidden
    x0 = torch.tensor([0.1, -0.2, 0.3], dtype=dtype, device=device)
    hidden = torch.matmul(B, x0) + Theta.squeeze()
    init_x0 = torch.cat((x0, hidden))

    model = G_rightfunc_class(A, B, tau, N, n)

    # run fols_Fun_tensor for small steps
    X, t = utils.fols_Fun_tensor(0.99, init_x0, 0, 0.01, 10, N+n, False, model)
    print('Returned X shape', X.shape, 't len', len(t))

    pre_x = X
    print('pre_x requires_grad:', pre_x.requires_grad)
    key_pre = pre_x[:, :n]

    # simple loss to test backward
    loss = (key_pre**2).mean()
    print('loss', loss.item())
    loss.backward()

    print('A.grad norm:', None if A.grad is None else A.grad.norm().item())
    print('B.grad norm:', None if B.grad is None else B.grad.norm().item())
    print('Theta.grad norm:', None if Theta.grad is None else Theta.grad.norm().item())
    print('tau.grad norm:', None if tau.grad is None else tau.grad.norm().item())

if __name__ == '__main__':
    run_test()
