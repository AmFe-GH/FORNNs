
from torchkeras import VLog
import timeit
import numpy as np
import Hyperparameters
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import sche_lr, draw3d, fols_Fun, detect_exceptions, generate_bits
from model import G_rightfunc_class, Linear_model, Model_for_init_param, additional_model, Mixed_loss
import torch.optim as optim
import torch
from utils import draw3d_train_test
import os
from Hyperparameters import *
import utils
# torch.autograd.set_detect_anomaly(True)
Simulation_system = "XOR"
train_state = int(
    input("Training Process:  1:train from checkpoint;   0:train from zero"))
# train_state = 0
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
os.chdir(current_dir)
os.makedirs('./Figure', exist_ok=True)

N_of_func_right, n_of_func_right, dtype, device, N_step, \
    _, alpha, _, epoch_number, _, x0, random_seed, lr, _, _\
    = set_Hyperparameters(Simulation_system)


os.makedirs('./trained_model', exist_ok=True)
torch.manual_seed(random_seed)
np.random.seed(random_seed)


history = {"loss": []}


save_fig_path = f'./Figure/{Simulation_system}/'
os.makedirs(save_fig_path, exist_ok=True)
log_interval = 100
fols_func = utils.fols_Fun_tensor_with_inputs(
    feature_size=N_of_func_right+n_of_func_right,
    N=N_step,
    device=device)
proj_input = torch.nn.Sequential(torch.nn.Linear(1+N_of_func_right+n_of_func_right, 30),
                                 torch.nn.LeakyReLU(),
                                 torch.nn.Linear(30, 10),
                                 torch.nn.LeakyReLU(),
                                 torch.nn.Linear(10, N_of_func_right+n_of_func_right)).to(device)
if train_state == 1:
    A, B, Theta, tau, optimizer_param, poj_param, backbone_param, max_acc = torch.load(
        f"./trained_model/FORNN_{Simulation_system}.pth", map_location=device)
    optimizer = torch.optim.Adam(
        [A, B, Theta, tau], lr=lr)
    optimizer_poj = torch.optim.Adam(proj_input.parameters(), lr=lr)
    optimizer_backbone = torch.optim.Adam(
        fols_func.backbone.parameters(), lr=lr)
    optimizer.load_state_dict(optimizer_param)
    optimizer_poj.load_state_dict(poj_param)
    optimizer_backbone.load_state_dict(backbone_param)

elif train_state == 0:
    A = torch.rand(n_of_func_right, N_of_func_right,
                   requires_grad=True, device=device)
    B = torch.rand(N_of_func_right, n_of_func_right,
                   requires_grad=True, device=device)
    Theta = torch.rand(N_of_func_right, 1, requires_grad=True, device=device)
    tau = torch.nn.Parameter(
        1e2 + torch.zeros(size=(1, n_of_func_right+N_of_func_right), dtype=dtype, device=device))
    optimizer = torch.optim.Adam(
        [A, B, Theta, tau], lr=lr)
    optimizer_poj = torch.optim.Adam(proj_input.parameters(), lr=lr)
    optimizer_backbone = torch.optim.Adam(
        fols_func.backbone.parameters(), lr=lr)
    max_acc = 0.
else:
    raise TypeError
criterion = torch.nn.BCELoss()
history = {"loss": [], "epsilon": [], 'accuracy': []}
print("max_acc", max_acc)
vlog = VLog(epoch_number//log_interval,
            monitor_metric='val_loss', monitor_mode='min')
# vlog.log_start()

start_time = timeit.default_timer()

predict_list = []


for epoch in range(epoch_number):
    inputs, label = generate_bits(N_step, device=device)

    hidden_state_x0 = torch.matmul(B, x0)+Theta.squeeze()
    init_x0 = torch.cat((x0, hidden_state_x0.squeeze()))
    model = G_rightfunc_class(
        A, B, tau, N_of_func_right, n_of_func_right, proj_input=proj_input)
    pre_x = fols_func(
        alpha, init_x0, 0, 1e1, len(inputs), N_of_func_right+n_of_func_right, False, model, inputs)
    assert pre_x.requires_grad
    # key_pre_x = torch.sigmoid(pre_x[:n_of_func_right])
    key_pre_x = pre_x[:n_of_func_right]
    key_real_x = label.unsqueeze(0)
    assert key_pre_x.shape == key_real_x.shape, [
        key_pre_x.shape, key_real_x.shape]
    loss = criterion(key_pre_x, key_real_x)
    right_predic = ((key_pre_x > 0.5).float() ==
                    key_real_x).cpu().squeeze().float()
    predict_list.append(right_predic)
    if len(predict_list) < 50:
        acc = np.mean(predict_list)
    else:
        acc = np.mean(predict_list[-50:])
    optimizer.zero_grad()
    optimizer_poj.zero_grad()
    optimizer_backbone.zero_grad()
    loss.backward()

    # assert A.grad is not None and B.grad is not None and \
    #     Theta.grad is not None and tau.grad is not None, \
    #     [A.grad, B.grad, Theta.grad, tau.grad]
    # torch.nn.utils.clip_grad_norm_([A, B, Theta, tau], max_norm=1.0)
    if epoch % log_interval == 0:
        # print("mean grad:", (A.grad.mean().item(), B.grad.mean().item(),
        #                      Theta.grad.mean().item(), tau.grad.mean().item()))
        print(inputs.cpu().tolist(), '\n', key_real_x.cpu().tolist(), '\n',
              key_pre_x.cpu().tolist())
        # print(proj_input[0].weight)
        pass
    optimizer.step()
    optimizer_poj.step()
    optimizer_backbone.step()
    if epoch % log_interval == 0:
        history['loss'].append(loss.item())
        history['accuracy'].append(acc.item())

        # draw3d(key_real_x, key_pre_x, show=False,
        #        save_path=save_fig_path+str(epoch)+'th_')
        # torch.save(key_pre_x, './Saved_data/key_pre_x_'+str(epoch)+'.pth')
        # vlog.log_epoch({'val_loss': loss.item(),
        #                 'train_loss': loss.item()})
        # print("alpha", alpha, "lr:", lr,
        #       "loss:", loss.item(), "min_loss ", min_loss, "loss_output",
        #       loss_output.item(), "loss_param",  loss_param.item())
        epsilon = torch.sqrt(
            torch.sum((key_pre_x-key_real_x)**2, dim=-1)).max()
        history['epsilon'].append(epsilon.item())
        print("loss acc epsilon", loss.item(), acc.item(), epsilon.item())
        if acc.item() > max_acc:
            torch.save(
                [A, B, Theta, tau, optimizer.state_dict(), optimizer_poj.state_dict(), optimizer_backbone.state_dict(), acc.item()], f"./trained_model/FORNN_{Simulation_system}.pth")
            max_acc = acc.item()

    # print(A.requires_grad)
    # print(A.grad)
    # assert A.grad==B.grad==Theta.grad==None
# vlog.log_end()
print(history)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Loss', color='blue')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 绘制准确率
plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label='Accuracy', color='green')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
