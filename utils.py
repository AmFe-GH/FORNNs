import matplotlib.ticker as ticker
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
import torch
from torchkeras import VLog
from tqdm import tqdm
import seaborn as sns


class sche_lr:
    def __init__(self, begin_lr, times_beta, beta, epoch_num, begin_epoch=0, ifplot=False):
        self.begin_lr = begin_lr
        self.beta = beta
        self.begin_epoch = begin_epoch
        self.times_beta = times_beta
        self.epoch_now = 0
        self.epoch_num = epoch_num
        self.ifplot = ifplot

    def __call__(self):

        if self.epoch_now < self.begin_epoch:
            new_lr = self.begin_lr
        else:
            progress_rate = (self.epoch_now-self.begin_epoch) / \
                (self.epoch_num-self.begin_epoch)

            new_lr = self.begin_lr * \
                (self.beta)**(np.ceil(progress_rate*self.times_beta))
        if self.ifplot:
            print("lr:", new_lr)
        self.epoch_now += 1
        return float(new_lr)


def X_rightfunc_Lorenz(x):
    if len(x.shape) == 1:
        x = x.unsqueeze(0)
    elif len(x.shape) != 2:
        raise ValueError
    # result = torch.zeros_like(x, requires_grad=True)
    # result[:, 0] = x[:, 0] +2/ x[:, 0]
    # result[:, 1] = x[:, 1] + 2 / x[:, 1]
    # result[:, 2] = x[:, 2] + 2 / x[:, 2]
    result = torch.zeros_like(x, requires_grad=False)
    result[:, 0] = 10*(x[:, 1]-x[:, 0])
    result[:, 1] = 28*x[:, 0] - x[:, 1]-x[:, 0]*x[:, 2]
    result[:, 2] = x[:, 0]*x[:, 1]-8/3*x[:, 2]

    assert len(result.shape) == 2
    return result


def X_rightfunc_Lorenz_numpy(x):
    val = np.array([[10*(x[0, 1]-x[0, 0]), 28*x[0, 0]-x[0, 1] -
                   x[0, 0]*x[0, 2], x[0, 0]*x[0, 1]-8/3*x[0, 2]]])
    return val


def X_rightfunc_Chua(x):
    def g_function_of_Chua(x0):
        val = -0.1 * x0 + 1/2 * (-3.9)*(torch.abs(x0+1)-torch.abs(x0-1))
        return val
    if len(x.shape) == 1:
        x = x.unsqueeze(0)
    elif len(x.shape) != 2:
        raise ValueError
    result = torch.zeros_like(x, requires_grad=False)

    result[:, 0] = 10*(0.7*(x[:, 1] - x[:, 0])-g_function_of_Chua(x[:, 0]))
    result[:, 1] = 0.5*(0.7*(x[:, 0]-x[:, 1])+x[:, 2])
    result[:, 2] = -7*x[:, 1]

    assert len(result.shape) == 2
    return result


def X_rightfunc_Chua_numpy(x):
    #
    def g_function_of_Chua(x0):
        val = -0.1 * x0 + 1/2 * (-3.9)*(np.abs(x0+1)-np.abs(x0-1))

        return val

    val = np.array([[10*(0.7*(x[0, 1] - x[0, 0])-g_function_of_Chua(x[0, 0])),
                    0.5*(0.7*(x[0, 0]-x[0, 1])+x[0, 2]),
                    -7*x[0, 1]]])
    return val


def generate_bits(seq_len, device):
    sequences = []

    # seq_len = torch.randint(
    #     low=seq_len_range[0], high=seq_len_range[1], size=(1,)).item()
    sequences = torch.randint(0, 2, (seq_len, 1)).float()
    xor_value = sequences.sum() % 2
    labels = xor_value
    return sequences.to(device), labels.to(device)


def detect_exceptions(arr):
    """
    检测NumPy数组或PyTorch张量中的异常数值。

    参数：
    - arr: NumPy数组或PyTorch张量

    返回：
    - 包含异常数值的字典，包括NaN、Inf和溢出的索引
    """

    result = {}

    if isinstance(arr, np.ndarray):
        # 对于 NumPy 数组
        nan_mask = np.isnan(arr)
        inf_mask = np.isinf(arr)
        overflow_mask = np.isneginf(arr) | np.isposinf(arr)

        nan_indices = np.where(nan_mask)[0]
        inf_indices = np.where(inf_mask)[0]
        overflow_indices = np.where(overflow_mask)[0]

        result['NaN_indices'] = nan_indices.tolist()
        result['Inf_indices'] = inf_indices.tolist()
        result['Overflow_indices'] = overflow_indices.tolist()

    elif isinstance(arr, torch.Tensor):
        # 对于 PyTorch 张量
        nan_mask = torch.isnan(arr)
        inf_mask = torch.isinf(arr)
        finite_mask = torch.isfinite(arr)

        nan_indices = torch.where(nan_mask)[0]
        inf_indices = torch.where(inf_mask)[0]
        overflow_indices = torch.where(~finite_mask)[0]

        result['NaN_indices'] = nan_indices.tolist()
        result['Inf_indices'] = inf_indices.tolist()
        result['Overflow_indices'] = overflow_indices.tolist()

    else:
        raise ValueError("不支持的数据类型")
    if any(result.values()):
        print(result)
        print("=========")
        print(arr)
        raise ValueError("存在异常值")


def draw3d_train_test(real_x, pre_x, real_t, N_step, test_step, fig_path=None):

    real_x = real_x.to('cpu').detach().numpy()
    pre_x = pre_x.to('cpu').detach().numpy()
    real_t = real_t.to('cpu').detach().numpy()

    alpha = 1.0
    fig, axs = plt.subplots(3, 1, figsize=(8, 6))
    for index in range(3):
        # First subplot
        axs[index].plot(real_t, real_x[:, index],
                        linewidth=2.2, color="steelblue")
        axs[index].plot(real_t[:N_step], pre_x[:N_step, index], linewidth=2.2,
                        color="firebrick", dashes=[4, 3])
        axs[index].plot(real_t[-test_step:], pre_x[-test_step:, index], linewidth=2.2,
                        color="peru", dashes=[4, 3])
        axs[2].set_xlabel("t (sec)")
    # Adjust layout to prevent overlap
    plt.tight_layout()

    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Save the figure if fig_path is provided
    if fig_path:
        plt.savefig(fig_path + "1D.test.eps", format="eps")
        plt.savefig(fig_path + "1D.test.png")

    plt.show()

    plt.figure(2)
    plt.subplot(1, 3, 1)
    plt.plot(pre_x[:N_step, 0], pre_x[:N_step, 1],
             linewidth=1.0,  color="firebrick", dashes=[6, 2])
    plt.plot(pre_x[-test_step:, 0], pre_x[-test_step:, 1],
             linewidth=1.0,  color="peru", dashes=[6, 2])
    plt.plot(real_x[:, 0], real_x[:, 1],
             linewidth=1.0,  color="steelblue", alpha=alpha)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.subplot(1, 3, 2)
    plt.plot(pre_x[:N_step, 0], pre_x[:N_step, 2],
             linewidth=1.0,  color="firebrick", dashes=[6, 2])
    plt.plot(pre_x[-test_step:, 0], pre_x[-test_step:, 2],
             linewidth=1.0,  color="peru", dashes=[6, 2])
    plt.plot(real_x[:, 0], real_x[:, 2],
             linewidth=1.0,  color="steelblue", alpha=alpha)
    plt.xlabel("x")
    plt.ylabel("z")
    plt.subplot(1, 3, 3)
    plt.plot(pre_x[:N_step, 1], pre_x[:N_step, 2],
             linewidth=1.0,  color="firebrick", dashes=[6, 2])
    plt.plot(pre_x[-test_step:, 1], pre_x[-test_step:, 2],
             linewidth=1.0,  color="peru", dashes=[6, 2])
    plt.plot(real_x[:, 1], real_x[:, 2],
             linewidth=1.0,  color="steelblue", alpha=alpha)
    plt.xlabel("y")
    plt.ylabel("z")
    # plt.suptitle("Real data from LORENZE system - 2D")
    plt.tight_layout()
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    if fig_path:
        plt.savefig(fig_path+"2D.eps", format="eps")
        plt.savefig(fig_path + "2D.test.png")

    plt.show()

    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    # 绘制3D折线图

    ax.plot(pre_x[:N_step, 0], pre_x[:N_step, 1],
            pre_x[:N_step, 2], linewidth=0.6, alpha=alpha, color="firebrick", dashes=[6, 2])
    ax.plot(pre_x[-test_step:, 0], pre_x[-test_step:, 1],
            pre_x[-test_step:, 2], linewidth=0.6, alpha=alpha, color="peru", dashes=[6, 2])
    ax.plot(real_x[:, 0], real_x[:, 1], real_x[:, 2], linewidth=0.6,
            alpha=0.1, color="steelblue")
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    if fig_path:
        plt.savefig(fig_path+"3D.eps", format="eps")
        plt.savefig(fig_path + "3D.test.png")

    plt.show()


def draw3d(real_x, pre_x, show=True, save_path=None):

    real_x = real_x.to('cpu').detach().numpy()
    pre_x = pre_x.to('cpu').detach().numpy()
    alpha = 0.8
    fig, axs = plt.subplots(3, 1, figsize=(8, 6))
    for index in range(3):
        # First subplot
        axs[index].plot(real_x[:, index],
                        linewidth=2.2, color="steelblue")
        axs[index].plot(pre_x[:, index], linewidth=2.2,
                        color="firebrick", dashes=[4, 3])
        axs[index].set_ylabel("x(t)")
        axs[2].set_xlabel("t (sec)")
    # Adjust layout to prevent overlap
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path+"1D.eps", format="eps")
        plt.savefig(save_path+"1D.png")
    if show:
        plt.show()
    plt.close()

    plt.figure(2)
    plt.subplot(1, 3, 1)
    plt.plot(pre_x[:, 0], pre_x[:, 1], linewidth=2, label="pre", alpha=alpha)
    plt.plot(real_x[:, 0], real_x[:, 1],
             linewidth=2, label="real", alpha=alpha)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.subplot(1, 3, 2)
    plt.plot(pre_x[:, 0], pre_x[:, 2], linewidth=2, label="pre", alpha=alpha)
    plt.plot(real_x[:, 0], real_x[:, 2],
             linewidth=2, label="real", alpha=alpha)
    plt.xlabel("x")
    plt.ylabel("z")
    plt.subplot(1, 3, 3)
    plt.plot(pre_x[:, 1], pre_x[:, 2], linewidth=2, label="pre", alpha=alpha)
    plt.plot(real_x[:, 1], real_x[:, 2],
             linewidth=2, label="real", alpha=alpha)
    plt.xlabel("y")
    plt.ylabel("z")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path+"2D.eps", format="eps")
        plt.savefig(save_path+"2D.png")

    if show:
        plt.show()
    plt.close()
    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    # 绘制3D折线图

    ax.plot(pre_x[:, 0], pre_x[:, 1], pre_x[:, 2], label='3D pre', alpha=alpha)
    ax.plot(real_x[:, 0], real_x[:, 1], real_x[:, 2],
            label='3D real', alpha=alpha)
    ax.legend()
    if save_path:
        plt.savefig(save_path+"3D.eps", format="eps")
        plt.savefig(save_path+"3D.png")

    if show:
        plt.show()
    plt.close()


def draw2d(real_x, pre_x):

    real_x = real_x.to('cpu').detach().numpy()
    pre_x = pre_x.to('cpu').detach().numpy()
    alpha = 0.8
    plt.figure(1)
    plt.subplot(1, 1, 1)
    plt.plot(pre_x[:, 0], linewidth=2, label="pre_0", alpha=alpha)
    plt.plot(pre_x[:, 1], linewidth=2, label="pre_1", alpha=alpha)
    plt.plot(real_x[:, 0], linewidth=2, label="real_0", alpha=alpha)
    plt.plot(real_x[:, 1], linewidth=2, label="real_1", alpha=alpha)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


# def generate_real_x(x0, N_step, X_rightfunc):
#     real_x = torch.zeros(size=(N_step+1, x0.shape[-1]))
#     real_x[0] = x0.clone()
#     for index in range(1, 1+N_step):
#         real_x[index] = X_rightfunc(real_x[index-1].reshape(1, -1))
#     return real_x.detach()


def fols_Fun(q, x0, t0, h, N, Nvar, ifplot, Funright):
    M = np.zeros((1, Nvar))
    x = np.zeros((N + 1, Nvar))
    x1 = np.zeros((N + 1, Nvar))
    Fx0 = Funright(x0)
    x1[0, :] = x0 + np.power(h, q) * Fx0 / (gamma(q) * q)
    Fx1 = Funright(x1[0, :][np.newaxis, :])
    x[0, :] = x0 + np.power(h, q) * (Fx1 + q * Fx0) / gamma(q + 2)
    for n in tqdm(range(1, N)):
        # 当n == 0的时候
        # X1的部分
        M = (np.power(n, q + 1) - (n - q) * np.power(n + 1, q)) * Fx0
        # YP的部分
        NN = (np.power(n + 1, q) - np.power(n, q)) * Fx0
        for j in range(1, n + 1):
            Fx = Funright(x[j - 1, :][np.newaxis, :])
            M = M + (np.power(n - j + 2, q + 1) + np.power(n - j, q +
                     1) - np.dot(2, np.power(n - j + 1, q + 1))) * Fx
            NN = NN + (np.power(n - j + 1, q) - np.power(n - j, q)) * Fx
        x1[n, :] = x0 + np.power(h, q) * NN / (gamma(q) * q)
        Fx = Funright(x1[n, :][np.newaxis, :])
        x[n, :] = x0 + np.power(h, q) * (Fx + M) / gamma(q + 2)
    X = np.vstack((x0, x[0:N, :]))
    detect_exceptions(X)
    t = np.array(range(1, N + 1))
    t = t0 + np.dot(h, t)
    t = t.tolist()
    t.insert(0, t0)
    return X, t


# def fols_Fun_tensor(q, x0, t0, h, N, Nvar, ifplot, Funright):
#     device = x0.device
#     M = np.zeros((1, Nvar))
#     x = torch.zeros((N + 1, Nvar), device=device)
#     x1 = torch.zeros((N + 1, Nvar), device=device)
#     Fx0 = Funright(x0)
#     x1[0, :] = x0 + np.power(h, q) * Fx0 / (gamma(q) * q)
#     Fx1 = Funright(x1[0, :])
#     x[0, :] = x0 + np.power(h, q) * (Fx1 + q * Fx0) / gamma(q + 2)
#     for n in range(1, N):
#         # 当n == 0的时候
#         # X1的部分
#         M = (np.power(n, q + 1) - (n - q) * np.power(n + 1, q)) * Fx0
#         # YP的部分
#         NN = (np.power(n + 1, q) - np.power(n, q)) * Fx0
#         for j in range(1, n + 1):
#             Fx = Funright(x[j - 1, :])
#             M = M + (np.power(n - j + 2, q + 1) + np.power(n - j, q +
#                      1) - np.dot(2, np.power(n - j + 1, q + 1))) * Fx
#             NN = NN + (np.power(n - j + 1, q) - np.power(n - j, q)) * Fx
#         x1[n, :] = x0 + np.power(h, q) * NN / (gamma(q) * q)
#         Fx = Funright(x1[n, :])
#         x[n, :] = x0 + np.power(h, q) * (Fx + M) / gamma(q + 2)
#     X = torch.cat((x0.unsqueeze(dim=0), x[0:N, :]), dim=0)
#     detect_exceptions(X)
#     t = np.array(range(1, N + 1))
#     t = t0 + np.dot(h, t)
#     t = t.tolist()
#     t.insert(0, t0)
#     return X, t
def fols_Fun_tensor(q, x0, t0, h, N, Nvar, ifplot, Funright):
    device = x0.device
    M = np.zeros((1, Nvar))
    x = [None for _ in range(N+1)]
    x1 = [None for _ in range(N+1)]
    Fx0 = Funright(x0)
    x1[0] = x0 + np.power(h, q) * Fx0 / (gamma(q) * q)
    Fx1 = Funright(x1[0])
    x[0] = x0 + np.power(h, q) * (Fx1 + q * Fx0) / gamma(q + 2)
    for n in range(1, N):
        # 当n == 0的时候
        # X1的部分
        M = (np.power(n, q + 1) - (n - q) * np.power(n + 1, q)) * Fx0
        # YP的部分
        NN = (np.power(n + 1, q) - np.power(n, q)) * Fx0
        for j in range(1, n + 1):
            Fx = Funright(x[j - 1])
            M = M + (np.power(n - j + 2, q + 1) + np.power(n - j, q +
                     1) - np.dot(2, np.power(n - j + 1, q + 1))) * Fx
            NN = NN + (np.power(n - j + 1, q) - np.power(n - j, q)) * Fx
        x1[n] = x0 + np.power(h, q) * NN / (gamma(q) * q)
        Fx = Funright(x1[n])
        x[n] = x0 + np.power(h, q) * (Fx + M) / gamma(q + 2)
    X = torch.cat((x0.unsqueeze(dim=0), torch.stack(x[:N], dim=0)), dim=0)
    detect_exceptions(X)
    t = np.array(range(1, N + 1))
    t = t0 + np.dot(h, t)
    t = t.tolist()
    t.insert(0, t0)
    return X, t


class fols_Fun_tensor_with_inputs:
    def __init__(self, feature_size, N, device):
        self.backbone = torch.nn.Sequential(torch.nn.Linear(N, 50*N),
                                            torch.nn.LeakyReLU(),
                                            torch.nn.Linear(50*N, 10*N),
                                            torch.nn.LeakyReLU(),
                                            torch.nn.Linear(
                                            10*N, 1))
        self.backbone.to(device)

    def __call__(self, q, x0, t0, h, N, Nvar, ifplot, Funright, inputs):
        x = [None for _ in range(N)]
        x1 = [None for _ in range(N)]
        Fx0 = Funright(x0, inputs[0])
        x1[0] = x0 + np.power(h, q) * Fx0 / (gamma(q) * q)
        Fx1 = Funright(x1[0], inputs[0])
        x[0] = x0 + np.power(h, q) * (Fx1 + q * Fx0) / gamma(q + 2)
        # x[0] = torch.sigmoid(x[0])
        for n in range(1, N):
            # 当n == 0的时候
            # X1的部分
            M = (np.power(n, q + 1) - (n - q) * np.power(n + 1, q)) * Fx0
            # YP的部分
            NN = (np.power(n + 1, q) - np.power(n, q)) * Fx0
            x1[n] = x0 + np.power(h, q) * NN / (gamma(q) * q)
            # print('x1[n]', x1[n])
            Fx = Funright(x1[n], inputs[n])
            detect_exceptions(Fx)
            x[n] = x0 + np.power(h, q) * (Fx + M) / gamma(q + 2)
            # if n != N-1:
            #     x[n] = torch.sigmoid(x[n])
        # return 0.001*self.proj(x[N-1].squeeze())+self.backbone(inputs.squeeze())
        return torch.sigmoid(x[N-1].squeeze().mean()+self.backbone(inputs.squeeze()))

# def fols_Fun_tensor_with_inputs(q, x0, t0, h, N, Nvar, ifplot, Funright, inputs):
#     device = x0.device
#     M = np.zeros((1, Nvar))
#     x = [None for _ in range(N+1)]
#     x[0] = x0
#     Fx0 = Funright(x0, inputs[0])
#     x[1] = Fx0
#     for n in range(1, N):
#         Fx = Funright(x[n], inputs[n])
#         detect_exceptions(Fx)
#         x[n+1] = Fx
#     # print('In fols:', x[n], Fx)
#     return x[n+1].squeeze()

# article:Data-driven Neural Network Discovery of Caputo Fractional Order Systems
# def I(F_function,start,end,width_step):

#     assert end >= start
#     step=(end-start)//width_step

#     result=0

#     for index in range(step):
#         result += (
#             F_function[index    * width_step]+
#             F_function[(index+1)* width_step]
#             ) * width_step /2

#     return result
