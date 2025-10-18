# from torchkeras import VLog
import timeit
import numpy as np
import FORNNs_hyperparameters
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import (
    sche_lr,
    draw3d,
    fols_Fun,
    detect_exceptions,
    X_rightfunc_Lorenz,
    X_rightfunc_Chua,
    X_rightfunc_Lorenz_numpy,
    X_rightfunc_Chua_numpy,
)
from model import (
    G_rightfunc_class,
    Linear_model,
    Model_for_init_param,
    additional_model,
    Mixed_loss,
)
import torch.optim as optim
import torch
from utils import draw3d_train_test
import os
from FORNNs_hyperparameters import *
import utils

Simulation_system = str(input("Simulation system:  Lorenz or Chua \n"))

train_state = int(
    input(
        "LDN Process options \n 2:refuse new train, use previous checkpoint; \n 1:new train from previous checkpoint; \n 0:train from scratch \n"
    )
)
##################################### start of parameter loading  ##############################################

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
os.chdir(current_dir)
os.makedirs("./Figure", exist_ok=True)


(
    N_of_func_right,
    n_of_func_right,
    dtype,
    device,
    N_step,
    test_step,
    alpha,
    beta,
    epoch_number,
    width_step,
    x0,
    random_seed,
    lr,
    LDN_lr,
    epochs_LDN,
) = set_Hyperparameters(Simulation_system)

if Simulation_system == "Lorenz":
    X_rightfunc = X_rightfunc_Lorenz
    X_rightfunc_numpy = X_rightfunc_Lorenz_numpy
elif Simulation_system == "Chua":
    X_rightfunc = X_rightfunc_Chua
    X_rightfunc_numpy = X_rightfunc_Chua_numpy
else:
    raise TypeError(
        "Simulation system must be Lorenz or Chua,your input is %s".format(Simulation_system)
    )

log_interval = 100
min_loss = 1e10
div = 10
width_step_pre = width_step / div
N_step_pre = N_step * div
test_step_pre = test_step * div
LDN_model_path = f"./trained_model/init_param_LDN_{Simulation_system}_alpha{alpha}.pth"
LDN_data_path = f"./Saved_data/data4LDN_{Simulation_system}_div{div}_alpha{alpha}"
os.makedirs("./trained_model", exist_ok=True)
os.makedirs("./Saved_data", exist_ok=True)
##################################### end of parameter loading  ##############################################


##################################### start of  LDN Process  ##############################################
print("start LDN Process")

if train_state == 1:
    assert os.path.exists(LDN_model_path), "checkpoint not found!"
    [A, B, Theta, optimizer_param, _] = torch.load(LDN_model_path)
    [A, B, Theta] = A.to(device), B.to(device), Theta.to(device)
    assert A.shape[-1] == N_of_func_right
    optimizer = optim.Adam([A, B, Theta], lr=LDN_lr)
    optimizer.load_state_dict(optimizer_param)
    for param_group in optimizer.param_groups:
        param_group["lr"] = LDN_lr
elif train_state == 0:
    A = torch.rand(n_of_func_right, N_of_func_right, requires_grad=True, device=device, dtype=dtype)
    B = torch.rand(N_of_func_right, n_of_func_right, requires_grad=True, device=device, dtype=dtype)
    Theta = torch.rand(N_of_func_right, 1, requires_grad=True, device=device, dtype=dtype)
    optimizer = optim.Adam([A, B, Theta], lr=LDN_lr, weight_decay=0)
elif train_state == 2:
    print("No new LDN process")
else:
    raise TypeError

if train_state != 2:
    Model_init = Model_for_init_param(A, B, Theta)
    criterion = torch.nn.MSELoss()
    if not os.path.exists(LDN_data_path):
        print("Generate training data for LDN process")
        real_x_pre, real_t_pre = fols_Fun(
            alpha,
            x0.cpu().unsqueeze(0).numpy(),
            0,
            width_step_pre,
            N_step_pre + test_step_pre,
            n_of_func_right,
            False,
            X_rightfunc_numpy,
        )
        print("Done!")

        real_x_pre = torch.tensor(real_x_pre, dtype=dtype, device=device, requires_grad=False)
        real_t_pre = torch.tensor(real_t_pre, dtype=dtype, device=device, requires_grad=False)

        torch.save([real_x_pre, real_t_pre], LDN_data_path)
    elif os.path.exists(LDN_data_path):
        print("Load existing training data for LDN process")
        real_x_pre, real_t_pre = torch.load(LDN_data_path, map_location=device)
        # real_x_pre = real_x_pre.to(dtype)
        # real_t_pre = real_t_pre.to(dtype)
        print("Done!")

    assert torch.isnan(real_x_pre).any() == False
    assert real_x_pre.requires_grad == False
    loss_history = []
    print("start LDN training process")
    pbar = tqdm(range(epochs_LDN))

    for epoch_num, _ in enumerate(pbar):
        # train_data = real_x_pre + \
        #     torch.rand_like(real_x_pre)*(real_x_pre.max())
        train_data = real_x_pre + torch.rand_like(real_x_pre, dtype=dtype) * 10
        # train_data = real_x_pre

        train_labels = X_rightfunc(train_data)
        output = Model_init(train_data)
        assert torch.isnan(output).any() == False
        loss = criterion(output[:N_step_pre], train_labels[:N_step_pre])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch_num % log_interval == 0:
            test_loss = criterion(output[-test_step_pre:], train_labels[-test_step_pre:])
            loss_history.append([loss.item(), test_loss.item()])
            pbar.set_postfix(loss=loss.item(), test_loss=test_loss.item())
            if loss.item() < min_loss:
                torch.save([A, B, Theta, optimizer.state_dict(), min_loss], LDN_model_path)
                min_loss = loss.item()
    # draw3d_train_test(labels, output, real_t_pre, N_step_pre,
    #                   test_step_pre, fig_path=None)
    plt.plot(loss_history)
    plt.title(f"LDN loss curve of {Simulation_system} system")
    plt.savefig(f"./Figure/LDN_loss_history_{Simulation_system}_div{div}_alpha{alpha}.png")
    plt.savefig(f"./Figure/LDN_loss_history_{Simulation_system}_div{div}_alpha{alpha}.svg")

    plt.show()
    plt.close()
##################################### end of LDN Process       ##############################################

##################################### start of model training  ##############################################
torch.manual_seed(random_seed)
np.random.seed(random_seed)

criterion = torch.nn.MSELoss(reduction="mean")
history = {"loss": []}
# x0 = torch.randn(size=[n_of_func_right], requires_grad=False)

print("Generate data for model training")
real_x, real_t = fols_Fun(
    alpha,
    x0.cpu().unsqueeze(0).numpy(),
    0,
    width_step,
    N_step + test_step,
    n_of_func_right,
    False,
    X_rightfunc_numpy,
)
print("Done!")

real_x = torch.tensor(real_x, dtype=dtype, device=device, requires_grad=False)
real_t = torch.tensor(real_t, dtype=dtype, device=device, requires_grad=False)
# draw3d(real_x, real_x)


save_fig_path = "./Figure/" + Simulation_system + "/" + "OursFORNNs" + "/"
os.makedirs(save_fig_path, exist_ok=True)
train_state = int(
    input("Training Process option: \n 1:train from checkpoint;\n 0: train from scratch \n")
)

log_interval = 1

if train_state == 1:
    A, B, Theta, tau, optimizer_param, min_loss = torch.load(
        f"./trained_model/FORNN_{Simulation_system}.pth", map_location=device
    )
    optimizer_Adam = torch.optim.Adagrad([A, B, Theta, tau], lr=lr)
    optimizer_Adam.load_state_dict(optimizer_param)
    for param_group in optimizer_Adam.param_groups:
        param_group["lr"] = lr

    original_A, original_B, original_Theta, _, _ = torch.load(LDN_model_path, map_location=device)
    criterion = Mixed_loss(original_A, original_B, original_Theta, beta=beta)
    min_loss = 1e10
elif train_state == 0:
    A, B, Theta, _, _ = torch.load(LDN_model_path, map_location=device)
    A.requires_grad_(True)
    B.requires_grad_(True)
    Theta.requires_grad_(True)
    tau = torch.nn.Parameter(
        1e4 + torch.zeros(size=(1, n_of_func_right + N_of_func_right), dtype=dtype, device=device)
    )
    optimizer_Adam = torch.optim.Adagrad([A, B, Theta, tau], lr=lr)
    criterion = Mixed_loss(A, B, Theta, beta=beta)
    min_loss = 1e10
else:
    raise TypeError
history = {"loss": [], "epsilon_train": [], "epsilon_test": []}
# vlog = VLog(epoch_number//log_interval,
#             monitor_metric='val_loss', monitor_mode='min')
# vlog.log_start()

start_time = timeit.default_timer()

for epoch in range(epoch_number):
    hidden_state_x0 = torch.matmul(B, x0.squeeze()) + Theta.squeeze()
    init_x0 = torch.cat((x0, hidden_state_x0.squeeze()))
    model = G_rightfunc_class(A, B, tau, N_of_func_right, n_of_func_right)
    pre_x, _ = utils.fols_Fun_tensor(
        alpha,
        init_x0,
        0,
        width_step,
        N_step + test_step,
        N_of_func_right + n_of_func_right,
        False,
        model,
    )
    assert len(pre_x) == N_step + test_step + 1

    assert pre_x.requires_grad
    pre_x = pre_x[:, :n_of_func_right]
    # key_pre_x = pre_x[: N_step+1, :]

    # key_real_x = real_x[: N_step + 1, :]
    # assert key_pre_x.shape == key_real_x.shape, [key_pre_x.shape, key_real_x.shape]
    loss, loss_output, loss_param = criterion(
        real_x[: N_step + 1, :], pre_x[: N_step + 1, :], A, B, Theta
    )

    optimizer_Adam.zero_grad()
    loss.backward()

    assert (
        A.grad is not None
        and B.grad is not None
        and Theta.grad is not None
        and tau.grad is not None
    ), [A.grad, B.grad, Theta.grad, tau.grad]

    # if epoch % log_interval == 0:
    #     print("mean grad:", (A.grad.mean().item(), B.grad.mean().item(),
    #                          Theta.grad.mean().item(), tau.grad.mean().item()))

    optimizer_Adam.step()

    if epoch % log_interval == 0:
        history["loss"].append(loss.item())

        draw3d(real_x, pre_x, show=False, save_path=save_fig_path + str(epoch) + "th_")
        # torch.save(key_pre_x, "./Saved_data/key_pre_x_" + str(epoch) + ".pth")
        # vlog.log_epoch({'val_loss': loss.item(),
        #                 'train_loss': loss.item()})

        epsilon_train = torch.sqrt(
            torch.sum((pre_x[: N_step + 1, :] - real_x[: N_step + 1, :]) ** 2, dim=-1)
        ).max()
        epsilon_test = torch.sqrt(
            torch.sum((pre_x[-test_step:, :] - real_x[-test_step:, :]) ** 2, dim=-1)
        ).max()
        history["epsilon_train"].append(epsilon_train.item())
        history["epsilon_test"].append(epsilon_test.item())

        print(
            "epoch:",
            epoch,
            "epsilon train:",
            epsilon_train.item(),
            "epsilon test:",
            epsilon_test.item(),
            " loss:",
            loss.item(),
            "\n alpha",
            alpha,
            " lr:",
            lr,
            " min_loss:",
            min_loss,
            " loss_output:",
            loss_output.item(),
            " loss_param:",
            f"{loss_param.item():.14f}",
        )

        if loss.item() < min_loss:
            torch.save(
                [A, B, Theta, tau, optimizer_Adam.state_dict(), loss.item()],
                f"./trained_model/FORNN_{Simulation_system}.pth",
            )
            min_loss = loss.item()
    # print(A.requires_grad)
    # print(A.grad)
    # print(B.grad)
    # assert A.grad==B.grad==Theta.grad==None
# vlog.log_end()
print(history)
##################################### end of model training  ##############################################
