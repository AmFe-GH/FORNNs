import numpy as np
import torch
import torch.nn as nn


class G_rightfunc_class:
    def __init__(self, A, B, tau, N, n, proj_input=None):
        assert A.requires_grad == True
        assert B.requires_grad == True
        assert tau.requires_grad == True

        pad = torch.nn.ZeroPad2d(padding=(n, 0, 0, 0))  # 左右上下
        up_matrix = pad(A)
        BA = torch.matmul(B, A)
        pad = torch.nn.ZeroPad2d(padding=(n, 0, 0, 0))
        down_matrix = pad(BA)

        self.proj_input = proj_input
        if proj_input is not None:
            self.proj_input.to(A.device)

        assert up_matrix.shape[-1] == down_matrix.shape[-1] == N+n
        self.padding_matrix = torch.cat((up_matrix, down_matrix), 0)
        assert self.padding_matrix.requires_grad == True
        assert self.padding_matrix.shape == (
            N+n, N+n), self.padding_matrix.shape
        assert self.padding_matrix.device == A.device
        self.tau = tau  # must be deepcopy

    def __call__(self, x, inputs=None):

        if inputs is not None:
            assert self.proj_input is not None

        if len(x.shape) == 1:
            stand_x = x.unsqueeze(0)
        elif len(x.shape) == 2:
            stand_x = x
        else:
            raise TypeError

        if inputs is not None:
            if len(inputs.shape) == 1:
                stand_inputs = inputs.unsqueeze(0)
            elif len(inputs.shape) == 2:
                stand_inputs = inputs
            elif len(inputs.shape) == 0:
                stand_inputs = inputs.unsqueeze(0).unsqueeze(0)
                assert len(stand_inputs.shape) == 2
            else:
                raise TypeError("inputs.shape", inputs, inputs.shape)

        assert torch.all(self.tau != 0.)
        temp = -1/self.tau
        elementA = temp*stand_x
        # elementA = torch.zeros_like(stand_x)-1/self.tau

        assert self.padding_matrix.dtype == stand_x.dtype, (
            self.padding_matrix.dtype, stand_x.dtype)

        elementB = torch.matmul(torch.sigmoid(
            stand_x), self.padding_matrix.t())

        assert elementA.shape == elementB.shape
        result = elementA + elementB

        if inputs is not None:
            elementC = self.proj_input(
                torch.cat([stand_x, stand_inputs], dim=-1))
            # elementC = stand_inputs
            result = elementC + result
        # print('in G_F: ', elementA, elementB, elementC)
        # print('in G_F: ', result)
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
