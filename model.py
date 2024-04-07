import numpy as np
import torch
import torch.nn as nn


class G_rightfunc_class:
    def __init__(self, A, B, tau, N, n):
        assert A.requires_grad == True
        assert B.requires_grad == True
        assert tau.requires_grad == True

        pad = torch.nn.ZeroPad2d(padding=(n, 0, 0, 0))  # 左右上下
        up_matrix = pad(A)
        BA = torch.matmul(B, A)
        pad = torch.nn.ZeroPad2d(padding=(n, 0, 0, 0))
        down_matrix = pad(BA)

        assert up_matrix.shape[-1] == down_matrix.shape[-1] == N+n
        self.padding_matrix = torch.cat((up_matrix, down_matrix), 0)
        assert self.padding_matrix.requires_grad == True
        assert self.padding_matrix.shape == (
            N+n, N+n), self.padding_matrix.shape
        assert self.padding_matrix.device == A.device
        self.tau = tau  # must be deepcopy

    def __call__(self, x):

        if len(x.shape) == 1:
            stand_input = x.unsqueeze(0)
        elif len(x.shape) == 2:
            stand_input = x.clone()
        else:
            raise TypeError

        assert torch.all(self.tau != 0.)
        temp = -1/self.tau
        elementA = temp * torch.sigmoid(stand_input)
        assert self.padding_matrix.dtype == stand_input.dtype, (
            self.padding_matrix.dtype, stand_input.dtype)

        elementB = torch.matmul(torch.sigmoid(
            stand_input), self.padding_matrix.t())

        assert elementA.shape == elementB.shape
        result = elementA + elementB

        return result.reshape(x.shape)


class Mixed_loss:
    def __init__(self, A, B, Theta, beta):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.original_A = A.clone().detach()
        self.original_B = B.clone().detach()
        self.original_Theta = Theta.clone().detach()
        assert self.original_A is not A
        self.beta = beta

    def __call__(self, labels, inputs, A, B, Theta):
        loss_output = self.mse(labels, inputs)
        loss_param = self.beta * \
            (((self.original_A-A)**2).mean() +
             ((self.original_B - B)**2).mean() +
             ((self.original_Theta-Theta)**2).mean()) + 1
        return loss_output*loss_param, loss_output, loss_param


class additional_model(nn.Module):
    def __init__(self, N, n):
        super().__init__()
        # 定义三层全连接层
        self.linear1 = nn.Linear(N+n, 1024)
        self.linear2 = nn.Linear(1024, N+n)

#         self.linear3 = nn.Linear(layers[2], layers[3])
#         self.linear4 = nn.Linear(layers[3], layers[4])

    def forward(self, x):
        output = self.linear1(x)

#         x = self.linear2(x)
        output = torch.tanh(output)
        output = self.linear2(output)

#         x = self.linear3(x)
#         x = torch.tanh(x)
        return output


class Linear_model(nn.Module):
    def __init__(self, layers):
        super().__init__()
        # 定义三层全连接层
        self.linear1 = nn.Linear(layers[0], layers[1], dtype=torch.float32)
        self.linear2 = nn.Linear(layers[1], layers[2], dtype=torch.float32)
#         self.linear3 = nn.Linear(layers[2], layers[3])
#         self.linear4 = nn.Linear(layers[3], layers[4])

    def forward(self, x):
        output = self.linear1(x)

#         x = self.linear2(x)
        output = torch.tanh(output)
#         x = self.linear3(x)
#         x = torch.tanh(x)
        return self.linear2(output)


class Model_for_init_param:
    def __init__(self, A, B, Theta):
        self.A = A
        self.B = B
        self.Theta = Theta

    def __call__(self, input):
        output = torch.matmul(input, self.B.t())+self.Theta.t()
        output = torch.sigmoid(output)
        return torch.matmul(output, self.A.t())
