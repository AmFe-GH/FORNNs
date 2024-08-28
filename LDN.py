
from Hyperparameters import *
import os
from utils import draw3d_train_test
import torch
import torch.optim as optim
from model import G_rightfunc_class, Linear_model, Model_for_init_param, additional_model, Mixed_loss
from utils import sche_lr, draw3d, fols_Fun, detect_exceptions, X_rightfunc_Lorenz, X_rightfunc_Chua, X_rightfunc_Lorenz_numpy, X_rightfunc_Chua_numpy
import matplotlib.pyplot as plt
from tqdm import tqdm
train_state = int(input("1:train from checkpoint;0:train from zero"))

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
os.chdir(current_dir)
os.makedirs('./Figure', exist_ok=True)

Simulation_system = 'Chua'
N_of_func_right, n_of_func_right, dtype, device, N_step, \
    test_step, alpha, width_step, x0, random_seed\
    = set_Hyperparameters(Simulation_system)

if Simulation_system == 'Lorenz':
    X_rightfunc = X_rightfunc_Lorenz
    X_rightfunc_numpy = X_rightfunc_Lorenz_numpy
elif Simulation_system == 'Chua':
    X_rightfunc = X_rightfunc_Chua
    X_rightfunc_numpy = X_rightfunc_Chua_numpy
else:
    raise TypeError

log_interval = 100
min_loss = 1e10
lr_init = 1e-3
div = 10
width_step_pre = width_step/div
N_step_pre = N_step*div
test_step_pre = test_step*div
model_path = f'./trained_model/init_param_LDN_{Simulation_system}_alpha{alpha}.pth'
data_path = f"./Saved_data/data4LDN_{Simulation_system}_div{div}_alpha{alpha}"
os.makedirs('./trained_model', exist_ok=True)
if train_state == 1:
    assert os.path.exists(model_path)
    [A, B, Theta, optimizer_param, min_loss] = torch.load(model_path)
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
    optimizer = optim.Adam([A, B, Theta], lr=lr_init, weight_decay=0)
else:
    raise TypeError

assert A.device == device
Model_init = Model_for_init_param(A, B, Theta)
criterion = torch.nn.MSELoss()
if not os.path.exists(data_path):
    real_x_pre, real_t_pre = fols_Fun(alpha, x0.cpu().unsqueeze(0).numpy(
    ), 0, width_step_pre, N_step_pre+test_step_pre, n_of_func_right, False, X_rightfunc_numpy)
    real_x_pre = torch.tensor(
        real_x_pre, dtype=dtype, device=device, requires_grad=False)
    real_t_pre = torch.tensor(
        real_t_pre, dtype=dtype, device=device, requires_grad=False)

    torch.save([real_x_pre, real_t_pre], data_path)
elif os.path.exists(data_path):
    real_x_pre, real_t_pre = torch.load(data_path, map_location=device)


assert torch.isnan(real_x_pre).any() == False
assert real_x_pre.requires_grad == False
loss_history = []
epsilon_history = []
pbar = tqdm(range(10000))
for epoch_num, _ in enumerate(pbar):
    train_data = real_x_pre + \
        torch.rand_like(real_x_pre)*(real_x_pre.max())
    # train_data = real_x_pre

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
                        min_loss], model_path)
            min_loss = loss.item()
# draw3d_train_test(labels, output, real_t_pre, N_step_pre,
#                   test_step_pre, fig_path=None)
plt.plot(loss_history)
plt.savefig(
    f'./Figure/LDN_loss_history_{Simulation_system}_div{div}_alpha{alpha}.png')
plt.savefig(
    f'./Figure/LDN_loss_history_{Simulation_system}_div{div}_alpha{alpha}.eps')

plt.show()
plt.close()
plt.plot(epsilon_history)
plt.savefig(
    f'./Figure/LDN_epsl_history_{Simulation_system}_div{div}_alpha{alpha}.png')
plt.savefig(
    f'./Figure/LDN_epsl_history_{Simulation_system}_div{div}_alpha{alpha}.eps')

plt.show()

print(min_loss)
