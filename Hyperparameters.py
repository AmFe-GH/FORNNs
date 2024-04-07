import torch
N_of_func_right = 300
n_of_func_right = 3
dtype = torch.float32
device = torch.device('cuda:0')
N_step = 500
test_step = 100
alpha = 0.996
width_step = 0.01
x0 = torch.tensor([-8., 7, 27], dtype=dtype, device=device)
random_seed = 490
