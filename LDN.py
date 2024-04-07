
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import sche_lr, draw3d, fols_Fun, detect_exceptions, X_rightfunc, X_rightfunc_numpy
from model import G_rightfunc_class, Linear_model, Model_for_init_param, additional_model, Mixed_loss
import torch.optim as optim
import torch
from utils import draw3d_train_test
import os
from Hyperparameters import *
train_state = int(input("1:train from checkpoint;0:train from zero"))

log_interval = 10
min_loss = 1e10
lr_init = 4e-3
div = 10
width_step_pre = width_step/div
N_step_pre = N_step*div
test_step_pre = test_step*div
path = './trained_model/init_param_overflow.pth'

if train_state == 1:
    assert os.path.exists(path)
    [A, B, Theta, optimizer_param, min_loss] = torch.load(path)
    [A, B, Theta] = A.to(device), B.to(device), Theta.to(device)
    assert A.shape[-1] == N_of_func_right
    optimizer = optim.Adam([A, B, Theta], lr=lr_init)
    optimizer.load_state_dict(optimizer_param)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_init
    print(min_loss)
elif train_state == 0:
    A = torch.rand(n_of_func_right, N_of_func_right,
                   requires_grad=True, device=device)
    B = torch.rand(N_of_func_right, n_of_func_right,
                   requires_grad=True, device=device)
    Theta = torch.rand(N_of_func_right, 1, requires_grad=True, device=device)
    min_loss = 1e10
    optimizer = optim.Adam([A, B, Theta], lr=lr_init)
else:
    raise TypeError

assert A.device == device
Model_init = Model_for_init_param(A, B, Theta)
criterion = torch.nn.MSELoss()
if not os.path.exists("./Saved_data/data4LDN"):
    real_x_pre, real_t_pre = fols_Fun(alpha, x0.cpu().unsqueeze(0).numpy(
    ), 0, width_step_pre, N_step_pre+test_step_pre, n_of_func_right, False, X_rightfunc_numpy)

    real_x_pre = torch.tensor(
        real_x_pre, dtype=dtype, device=device, requires_grad=False)
    real_t_pre = torch.tensor(
        real_t_pre, dtype=dtype, device=device, requires_grad=False)
    labels = X_rightfunc(real_x_pre)
    torch.save([real_x_pre, real_t_pre, labels], "./Saved_data/data4LDN")
elif os.path.exists("./Saved_data/data4LDN"):
    real_x_pre, real_t_pre, labels = torch.load(
        "./Saved_data/data4LDN", map_location=device)

assert torch.isnan(real_x_pre).any() == False
assert real_x_pre.requires_grad == False
loss_history = []
epsilon_history = []
pbar = tqdm(range(1000))
for epoch_num, _ in enumerate(pbar):
    train_data = real_x_pre + torch.rand_like(real_x_pre)*(real_t_pre.max())
    train_labels = X_rightfunc(train_data)
    output = Model_init(train_data)
    loss = criterion(output[:N_step_pre], train_labels[:N_step_pre])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch_num % log_interval == 0:
        epsl = torch.sqrt(
            torch.sum((output[:N_step_pre]-train_labels[:N_step_pre])**2, dim=-1)).max()
        test_loss = criterion(output[-test_step_pre:],
                              train_labels[-test_step_pre:])
        loss_history.append([loss.item(), test_loss.item()])
        epsilon_history.append(epsl.item())
        pbar.set_postfix(loss=loss.item(),
                         test_loss=test_loss.item(), epsilon=epsl.item())
        if loss.item() < min_loss:
            torch.save([A, B, Theta, optimizer.state_dict(),
                        min_loss], path)
            min_loss = loss.item()
# draw3d_train_test(labels, output, real_t_pre, N_step_pre,
#                   test_step_pre, fig_path=None)
plt.plot(loss_history)
plt.show()
plt.plot(epsilon_history)
plt.show()
print(min_loss)
print(epsilon_history[-4:])
