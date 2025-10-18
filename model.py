import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from typing import Tuple, Optional


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

        assert up_matrix.shape[-1] == down_matrix.shape[-1] == N + n
        self.padding_matrix = torch.cat((up_matrix, down_matrix), 0)
        assert self.padding_matrix.requires_grad == True
        assert self.padding_matrix.shape == (N + n, N + n), self.padding_matrix.shape
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

        assert torch.all(self.tau != 0.0)
        temp = -1 / self.tau
        elementA = temp * stand_x
        # elementA = torch.zeros_like(stand_x)-1/self.tau

        assert self.padding_matrix.dtype == stand_x.dtype, (
            self.padding_matrix.dtype,
            stand_x.dtype,
        )

        elementB = torch.matmul(torch.sigmoid(stand_x), self.padding_matrix.t())

        assert elementA.shape == elementB.shape
        result = elementA + elementB

        if inputs is not None:
            elementC = self.proj_input(torch.cat([stand_x, stand_inputs], dim=-1))
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
        loss_param = (
            self.beta
            * (
                ((self.original_A - A) ** 2).mean()
                + ((self.original_B - B) ** 2).mean()
                + ((self.original_Theta - Theta) ** 2).mean()
            )
            + 1
        )
        return loss_output * loss_param, loss_output, loss_param


class additional_model(nn.Module):
    def __init__(self, N, n):
        super().__init__()
        # 定义三层全连接层
        self.linear1 = nn.Linear(N + n, 1024)
        self.linear2 = nn.Linear(1024, N + n)

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
        output = torch.matmul(input, self.B.t()) + self.Theta.t()
        output = torch.sigmoid(output)
        return torch.matmul(output, self.A.t())


class CTRNNCell(nn.Module):
    def __init__(
        self, units, method, num_unfolds=None, tau=1, input_dim=None, output_dim=3, device="cpu"
    ):
        super(CTRNNCell, self).__init__()

        self.device = torch.device(device)
        self.output_dim = output_dim
        self.fixed_step_methods = {
            "euler": self.euler,
            "heun": self.heun,
            "rk4": self.rk4,
        }
        allowed_methods = ["euler", "heun", "rk4", "dopri5"]

        if method not in allowed_methods:
            raise ValueError(f"Unknown ODE solver '{method}', expected one of '{allowed_methods}'")
        if method in self.fixed_step_methods.keys() and num_unfolds is None:
            raise ValueError(
                "Fixed-step ODE solver requires argument 'num_unfolds' to be specified!"
            )

        self.units = units
        self.num_unfolds = num_unfolds
        self.method = method
        self.tau = tau

        if input_dim is not None:
            self._build_weights(input_dim)
        else:
            self.kernel = None

        if self.method == "dopri5":
            try:
                import torchode as to

                # 配置ODE求解器
                self.solver_term = to.ODETerm(self._ode_func)
                self.solver = to.AutoDiffAdjoint(to.Dopri5(atol=1e-4, rtol=0.01), self.solver_term)
            except ImportError:
                raise ImportError("dopri5 method requires torchode library")

    def _build_weights(self, input_dim):
        self.kernel = nn.Linear(input_dim, self.units, bias=False).to(self.device)
        self.recurrent_kernel = nn.Linear(self.units, self.units, bias=False).to(self.device)
        self.bias = nn.Parameter(torch.zeros(self.units, device=self.device))
        self.scale = nn.Parameter(torch.ones(self.units, device=self.device))
        self.init_state_proj = nn.Linear(self.output_dim, self.units, bias=False).to(self.device)
        nn.init.xavier_uniform_(self.kernel.weight)
        nn.init.orthogonal_(self.recurrent_kernel.weight)

    def _ode_func(self, t, y, args):
        """ODE函数，用于torchode"""
        inputs = args["inputs"]
        return self.dfdt(inputs, y)

    def get_initial_state(self, batch_size, init_position):
        return self.init_state_proj(init_position)

    def forward(self, inputs, hidden_state, elapsed=1.0):
        if isinstance(inputs, (tuple, list)):
            if len(inputs) > 1:
                elapsed = inputs[1]
                inputs = inputs[0]
            else:
                inputs = inputs[0]

        if self.kernel is None:
            self._build_weights(inputs.shape[-1])

        inputs = inputs.to(self.device)
        hidden_state = hidden_state.to(self.device)

        if self.method == "dopri5":
            if not isinstance(elapsed, torch.Tensor):
                elapsed = torch.tensor(elapsed, device=self.device)

            if elapsed.dim() > 0 and len(elapsed) > 1:
                results = []
                for i in range(len(elapsed)):
                    t_span = torch.tensor([0.0, elapsed[i].item()], device=self.device)
                    sol = self.solver.solve(
                        self.solver_term,
                        t_span,
                        hidden_state[i : i + 1],
                        args={"inputs": inputs[i : i + 1]},
                    )
                    results.append(sol.ys[-1])
                hidden_state = torch.cat(results, dim=0)
            else:
                t_span = torch.tensor(
                    [0.0, elapsed.item() if isinstance(elapsed, torch.Tensor) else elapsed],
                    device=self.device,
                )
                sol = self.solver.solve(
                    self.solver_term, t_span, hidden_state, args={"inputs": inputs}
                )
                hidden_state = sol.ys[-1]
        else:
            delta_t = elapsed / self.num_unfolds
            method = self.fixed_step_methods[self.method]
            for _ in range(self.num_unfolds):
                hidden_state = method(inputs, hidden_state, delta_t)

        return hidden_state

    def dfdt(self, inputs, hidden_state):
        h_in = self.kernel(inputs)
        h_rec = self.recurrent_kernel(hidden_state)
        dh_in = self.scale * torch.tanh(h_in + h_rec + self.bias)

        if self.tau > 0:
            dh = dh_in - hidden_state * self.tau
        else:
            dh = dh_in
        return dh

    def euler(self, inputs, hidden_state, delta_t):
        dy = self.dfdt(inputs, hidden_state)
        return hidden_state + delta_t * dy

    def heun(self, inputs, hidden_state, delta_t):
        k1 = self.dfdt(inputs, hidden_state)
        k2 = self.dfdt(inputs, hidden_state + delta_t * k1)
        return hidden_state + delta_t * 0.5 * (k1 + k2)

    def rk4(self, inputs, hidden_state, delta_t):
        k1 = self.dfdt(inputs, hidden_state)
        k2 = self.dfdt(inputs, hidden_state + k1 * delta_t * 0.5)
        k3 = self.dfdt(inputs, hidden_state + k2 * delta_t * 0.5)
        k4 = self.dfdt(inputs, hidden_state + k3 * delta_t)

        return hidden_state + delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


CTRNN = CTRNNCell


class LSTMCell(nn.Module):
    def __init__(self, units, input_dim=None, output_dim=3, device="cpu"):
        super(LSTMCell, self).__init__()
        self.units = units
        self.device = torch.device(device)
        self.output_dim = output_dim
        if input_dim is not None:
            self._build_weights(input_dim)
        else:
            self.input_kernel = None

    def _build_weights(self, input_dim):
        self.input_kernel = nn.Linear(input_dim, 4 * self.units).to(self.device)
        self.recurrent_kernel = nn.Linear(self.units, 4 * self.units, bias=False).to(self.device)
        self.init_state_proj1 = nn.Linear(self.output_dim, self.units).to(self.device)
        self.init_state_proj2 = nn.Linear(self.output_dim, self.units).to(self.device)
        # 初始化
        nn.init.xavier_uniform_(self.input_kernel.weight)
        nn.init.orthogonal_(self.recurrent_kernel.weight)
        nn.init.zeros_(self.input_kernel.bias)

    def get_initial_state(self, batch_size, init_position):
        return (
            self.init_state_proj1(init_position),
            self.init_state_proj2(init_position),
        )
        # return (
        #     torch.zeros(batch_size, self.units, device=self.device),
        #     torch.zeros(batch_size, self.units, device=self.device),
        # )

    def forward(self, inputs, states, elapsed=1.0):
        cell_state, output_state = states

        if isinstance(inputs, (tuple, list)):
            if len(inputs) > 1:
                inputs = torch.cat([inputs[0], inputs[1]], dim=-1)
            else:
                inputs = inputs[0]

        inputs = inputs.to(self.device)
        cell_state = cell_state.to(self.device)
        output_state = output_state.to(self.device)

        if self.input_kernel is None:
            self._build_weights(inputs.shape[-1])

        z = self.input_kernel(inputs) + self.recurrent_kernel(output_state)
        i, ig, fg, og = torch.chunk(z, 4, dim=-1)

        input_activation = torch.tanh(i)
        input_gate = torch.sigmoid(ig)
        forget_gate = torch.sigmoid(fg + 1.0)
        output_gate = torch.sigmoid(og)

        new_cell = cell_state * forget_gate + input_activation * input_gate
        new_output = torch.tanh(new_cell) * output_gate

        return new_output, (new_cell, new_output)


AugmentedLSTM = LSTMCell


class ODELSTM(nn.Module):
    def __init__(self, units, input_dim=None, output_dim=3, device="cpu"):
        super(ODELSTM, self).__init__()
        self.units = units
        self.device = torch.device(device)
        self.ctrnn = CTRNNCell(
            self.units, num_unfolds=10, method="euler", input_dim=self.units, device=device
        )
        self.output_dim = output_dim

        if input_dim is not None:
            self._build_weights(input_dim)
        else:
            self.input_kernel = None

    def _build_weights(self, input_dim):
        self.input_kernel = nn.Linear(input_dim, 4 * self.units).to(self.device)
        self.recurrent_kernel = nn.Linear(self.units, 4 * self.units, bias=False).to(self.device)
        self.init_state_proj1 = nn.Linear(self.output_dim, self.units).to(self.device)
        self.init_state_proj2 = nn.Linear(self.output_dim, self.units).to(self.device)
        nn.init.xavier_uniform_(self.input_kernel.weight)
        nn.init.orthogonal_(self.recurrent_kernel.weight)
        nn.init.zeros_(self.input_kernel.bias)

    def get_initial_state(self, batch_size, init_position):
        return (
            self.init_state_proj1(init_position),
            self.init_state_proj2(init_position),
        )

    def forward(self, inputs, states, elapsed=1.0):
        cell_state, ode_state = states
        if isinstance(inputs, (tuple, list)):
            if len(inputs) > 1:
                elapsed = inputs[1]
                inputs = inputs[0]
            else:
                inputs = inputs[0]

        inputs = inputs.to(self.device)
        cell_state = cell_state.to(self.device)
        ode_state = ode_state.to(self.device)

        if self.input_kernel is None:
            self._build_weights(inputs.shape[-1])
        z = self.input_kernel(inputs) + self.recurrent_kernel(ode_state)
        i, ig, fg, og = torch.chunk(z, 4, dim=-1)

        input_activation = torch.tanh(i)
        input_gate = torch.sigmoid(ig)
        forget_gate = torch.sigmoid(fg + 3.0)
        output_gate = torch.sigmoid(og)

        new_cell = cell_state * forget_gate + input_activation * input_gate
        ode_input = torch.tanh(new_cell) * output_gate
        ode_output = self.ctrnn(ode_input, ode_state, elapsed)

        return ode_output, (new_cell, ode_output)


class CTGRU(nn.Module):
    def __init__(self, units, M=8, input_dim=None, output_dim=3, device="cpu"):
        super(CTGRU, self).__init__()
        self.units = units
        self.M = M
        self.device = torch.device(device)
        self.output_dim = output_dim
        self.ln_tau_table = np.empty(self.M, dtype=np.float32)
        self.tau_table = np.empty(self.M, dtype=np.float32)
        tau = 1.0
        for i in range(self.M):
            self.ln_tau_table[i] = np.log(tau)
            self.tau_table[i] = tau
            tau = tau * (10.0**0.5)

        self.register_buffer(
            "ln_tau_table_tensor", torch.tensor(self.ln_tau_table, device=self.device)
        )
        self.register_buffer("tau_table_tensor", torch.tensor(self.tau_table, device=self.device))

        if input_dim is not None:
            self._build_layers(input_dim)
        else:
            self.retrieval_layer = None

    def _build_layers(self, input_dim):
        fused_dim = input_dim + self.units
        self.retrieval_layer = nn.Linear(fused_dim, self.units * self.M).to(self.device)
        self.detect_layer = nn.Linear(fused_dim, self.units).to(self.device)
        self.update_layer = nn.Linear(fused_dim, self.units * self.M).to(self.device)
        self.init_state_proj = nn.Linear(self.output_dim, self.units * self.M).to(self.device)

    def get_initial_state(self, batch_size, init_position):
        return self.init_state_proj(init_position)
        # return torch.zeros(batch_size, self.units * self.M, device=self.device)

    def forward(self, inputs, states, elapsed=1.0):
        if isinstance(inputs, (tuple, list)):
            if len(inputs) > 1:
                elapsed = inputs[1]
                inputs = inputs[0]
            else:
                inputs = inputs[0]

        inputs = inputs.to(self.device)
        states = states.to(self.device)

        batch_dim = inputs.shape[0]

        if self.retrieval_layer is None:
            self._build_layers(inputs.shape[-1])

        h_hat = states.view(batch_dim, self.units, self.M)
        h = torch.sum(h_hat, dim=2)

        fused_input = torch.cat([inputs, h], dim=-1)
        ln_tau_r = self.retrieval_layer(fused_input)
        ln_tau_r = ln_tau_r.view(batch_dim, self.units, self.M)
        sf_input_r = -torch.square(ln_tau_r - self.ln_tau_table_tensor)
        rki = F.softmax(sf_input_r, dim=2)

        q_input = torch.sum(rki * h_hat, dim=2)
        reset_value = torch.cat([inputs, q_input], dim=1)
        qk = torch.tanh(self.detect_layer(reset_value))
        qk = qk.unsqueeze(2)  # 广播用

        ln_tau_s = self.update_layer(fused_input)
        ln_tau_s = ln_tau_s.view(batch_dim, self.units, self.M)
        sf_input_s = -torch.square(ln_tau_s - self.ln_tau_table_tensor)
        ski = F.softmax(sf_input_s, dim=2)

        base_term = (1 - ski) * h_hat + ski * qk
        exp_term = torch.exp(-elapsed / self.tau_table_tensor)
        exp_term = exp_term.view(1, 1, self.M)
        h_hat_next = base_term * exp_term

        h_next = torch.sum(h_hat_next, dim=2)
        h_hat_next_flat = h_hat_next.view(batch_dim, self.units * self.M)

        return h_next, h_hat_next_flat


class BidirectionalRNN(nn.Module):
    def __init__(self, units, input_dim=None, output_dim=3, device="cpu"):
        super(BidirectionalRNN, self).__init__()
        self.units = units
        self.device = torch.device(device)
        self.init_state_proj1 = nn.Linear(output_dim, units).to(self.device)
        self.init_state_proj2 = nn.Linear(output_dim, units).to(self.device)
        self.init_state_proj3 = nn.Linear(output_dim, units).to(self.device)

        self.ctrnn = CTRNNCell(
            self.units,
            num_unfolds=4,
            method="euler",
            input_dim=input_dim + units if input_dim else None,
            device=device,
        )
        self.lstm = LSTMCell(
            units=self.units, input_dim=input_dim + units if input_dim else None, device=device
        )

        if input_dim is not None:
            self.out_layer = nn.Linear(self.units, self.units).to(self.device)
        else:
            self.out_layer = None

    def get_initial_state(self, batch_size, init_position):
        return (
            self.init_state_proj1(init_position),  # LSTM cell state
            self.init_state_proj2(init_position),  # LSTM hidden state
            self.init_state_proj3(init_position),  # CTRNN state
        )

    def forward(self, inputs, states, elapsed=1.0):
        if isinstance(inputs, (tuple, list)):
            if len(inputs) > 1:
                elapsed = inputs[1]
                inputs = inputs[0]
            else:
                inputs = inputs[0]

        inputs = inputs.to(self.device)
        lstm_cell_state, lstm_hidden_state, ctrnn_state = [s.to(self.device) for s in states]

        if self.out_layer is None:
            self.out_layer = nn.Linear(self.units, self.units).to(self.device)

        lstm_input = torch.cat([inputs, ctrnn_state], dim=-1)
        ctrnn_input = torch.cat([inputs, lstm_hidden_state], dim=-1)

        lstm_out, (new_lstm_cell, new_lstm_hidden) = self.lstm(
            lstm_input, (lstm_cell_state, lstm_hidden_state), elapsed
        )
        ctrnn_out = self.ctrnn(ctrnn_input, ctrnn_state, elapsed)

        fused_output = lstm_out + ctrnn_out
        return fused_output, (new_lstm_cell, new_lstm_hidden, ctrnn_out)


class GRUD(nn.Module):
    def __init__(self, units, input_dim=None, output_dim=3, device="cpu"):
        super(GRUD, self).__init__()
        self.units = units
        self.device = torch.device(device)
        self.output_dim = output_dim
        if input_dim is not None:
            self._build_layers(input_dim)
        else:
            self.reset_gate = None

    def _build_layers(self, input_dim):
        fused_dim = input_dim + self.units
        self.reset_gate = nn.Linear(fused_dim, self.units).to(self.device)
        self.detect_signal = nn.Linear(fused_dim, self.units).to(self.device)
        self.update_gate = nn.Linear(fused_dim, self.units).to(self.device)
        self.d_gate = nn.Linear(1, self.units).to(self.device)  # elapsed time input
        self.init_state_proj = nn.Linear(self.output_dim, self.units).to(self.device)

    def get_initial_state(self, batch_size, init_position):
        return self.init_state_proj(init_position)

    def forward(self, inputs, states, elapsed=1.0):
        if isinstance(inputs, (tuple, list)):
            if len(inputs) > 1:
                elapsed = inputs[1]
                inputs = inputs[0]
            else:
                inputs = inputs[0]

        inputs = inputs.to(self.device)
        states = states.to(self.device)

        if not isinstance(elapsed, torch.Tensor):
            elapsed = torch.tensor(elapsed, device=self.device)
        if elapsed.dim() == 0:
            elapsed = elapsed.expand(inputs.shape[0], 1)
        elif elapsed.dim() == 1:
            elapsed = elapsed.unsqueeze(1)

        elapsed = elapsed.to(self.device)

        if self.reset_gate is None:
            self._build_layers(inputs.shape[-1])

        dt = F.relu(self.d_gate(elapsed))
        gamma = torch.exp(-dt)
        h_hat = states * gamma

        fused_input = torch.cat([inputs, h_hat], dim=-1)
        rt = torch.sigmoid(self.reset_gate(fused_input))
        zt = torch.sigmoid(self.update_gate(fused_input))

        reset_value = torch.cat([inputs, rt * h_hat], dim=-1)
        h_tilde = torch.tanh(self.detect_signal(reset_value))

        ht = zt * h_hat + (1.0 - zt) * h_tilde

        return ht, ht


class PhasedLSTM(nn.Module):
    def __init__(self, units, input_dim=None, output_dim=3, device="cpu"):
        super(PhasedLSTM, self).__init__()
        self.units = units
        self.device = torch.device(device)
        self.output_dim = output_dim
        if input_dim is not None:
            self._build_weights(input_dim)
        else:
            self.input_kernel = None

        self.tau = nn.Parameter(torch.zeros(1, device=self.device))
        self.ron = nn.Parameter(torch.zeros(1, device=self.device))
        self.s = nn.Parameter(torch.zeros(1, device=self.device))

    def _build_weights(self, input_dim):
        self.input_kernel = nn.Linear(input_dim, 4 * self.units).to(self.device)
        self.recurrent_kernel = nn.Linear(self.units, 4 * self.units, bias=False).to(self.device)
        self.init_state_proj1 = nn.Linear(self.output_dim, self.units).to(self.device)
        self.init_state_proj2 = nn.Linear(self.output_dim, self.units).to(self.device)
        nn.init.xavier_uniform_(self.input_kernel.weight)
        nn.init.orthogonal_(self.recurrent_kernel.weight)
        nn.init.zeros_(self.input_kernel.bias)

    def get_initial_state(self, batch_size, init_position):
        return (
            self.init_state_proj1(init_position),
            self.init_state_proj2(init_position),
        )

    def forward(self, inputs, states, elapsed=1.0):
        cell_state, hidden_state = states

        if isinstance(inputs, (tuple, list)):
            if len(inputs) > 1:
                elapsed = inputs[1]
                inputs = inputs[0]
            else:
                inputs = inputs[0]

        inputs = inputs.to(self.device)
        cell_state = cell_state.to(self.device)
        hidden_state = hidden_state.to(self.device)

        if self.input_kernel is None:
            self._build_weights(inputs.shape[-1])

        alpha = 0.001
        tau = F.softplus(self.tau)
        s = F.softplus(self.s)
        ron = F.softplus(self.ron)

        if not isinstance(elapsed, torch.Tensor):
            elapsed = torch.tensor(elapsed, device=self.device)

        phit = torch.fmod(elapsed - s, tau) / tau

        cond1 = phit < 0.5 * ron
        cond2 = phit < ron

        kt = torch.where(
            cond1, 2 * phit / ron, torch.where(cond2, 2.0 - 2 * phit / ron, alpha * phit)
        )
        z = self.input_kernel(inputs) + self.recurrent_kernel(hidden_state)
        i, ig, fg, og = torch.chunk(z, 4, dim=-1)

        input_activation = torch.tanh(i)
        input_gate = torch.sigmoid(ig)
        forget_gate = torch.sigmoid(fg + 1.0)
        output_gate = torch.sigmoid(og)

        c_tilde = cell_state * forget_gate + input_activation * input_gate
        c = kt * c_tilde + (1.0 - kt) * cell_state

        h_tilde = torch.tanh(c_tilde) * output_gate
        h = kt * h_tilde + (1.0 - kt) * hidden_state

        return h, (c, h)


class GRUODE(nn.Module):
    def __init__(self, units, num_unfolds=4, input_dim=None, output_dim=3, device="cpu"):
        super(GRUODE, self).__init__()
        self.units = units
        self.num_unfolds = num_unfolds
        self.device = torch.device(device)
        self.output_dim = output_dim
        if input_dim is not None:
            self._build_layers(input_dim)
        else:
            self.reset_gate = None

    def _build_layers(self, input_dim):
        fused_dim = input_dim + self.units
        self.reset_gate = nn.Linear(fused_dim, self.units).to(self.device)
        self.detect_signal = nn.Linear(fused_dim, self.units).to(self.device)
        self.update_gate = nn.Linear(fused_dim, self.units).to(self.device)
        self.init_state_proj = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.Sigmoid(),
            nn.Linear(self.output_dim, self.units),
        ).to(self.device)
        nn.init.constant_(self.reset_gate.bias, 1.0)

    def get_initial_state(self, batch_size, init_position):
        return self.init_state_proj(init_position)

    def _dh_dt(self, inputs, states):
        fused_input = torch.cat([inputs, states], dim=-1)
        rt = torch.sigmoid(self.reset_gate(fused_input))
        zt = torch.sigmoid(self.update_gate(fused_input))

        reset_value = torch.cat([inputs, rt * states], dim=-1)
        gt = torch.tanh(self.detect_signal(reset_value))

        dhdt = (1.0 - zt) * (gt - states)
        return dhdt

    def euler(self, inputs, hidden_state, delta_t):
        dy = self._dh_dt(inputs, hidden_state)
        return hidden_state + delta_t * dy

    def forward(self, inputs, states, elapsed=1.0):
        if isinstance(inputs, (tuple, list)):
            if len(inputs) > 1:
                elapsed = inputs[1]
                inputs = inputs[0]
            else:
                inputs = inputs[0]

        inputs = inputs.to(self.device)
        states = states.to(self.device)

        if self.reset_gate is None:
            self._build_layers(inputs.shape[-1])

        delta_t = elapsed / self.num_unfolds
        hidden_state = states
        for _ in range(self.num_unfolds):
            hidden_state = self.euler(inputs, hidden_state, delta_t)

        return hidden_state, hidden_state


class S4Layer(torch.nn.Module):
    """
    S4 Layer implementation based on Structured State Space Sequence model
    """

    def __init__(self, state_dim, expand_dim, dt=0.1, device="cpu"):
        super(S4Layer, self).__init__()
        self.state_dim = state_dim
        self.expand_dim = expand_dim
        self.dt = torch.tensor(dt, device=device)
        self.device = device

        self.Lambda_real = torch.nn.Parameter(torch.randn(self.state_dim, device=device))
        self.Lambda_imag = torch.nn.Parameter(torch.randn(self.state_dim, device=device))

        self.B_real = torch.nn.Parameter(
            torch.randn(self.expand_dim, self.state_dim, device=device)
        )
        self.B_imag = torch.nn.Parameter(
            torch.randn(self.expand_dim, self.state_dim, device=device)
        )
        self.C_real = torch.nn.Parameter(
            torch.randn(self.expand_dim, self.state_dim, device=device)
        )
        self.C_imag = torch.nn.Parameter(
            torch.randn(self.expand_dim, self.state_dim, device=device)
        )

        self.D = torch.nn.Parameter(torch.randn(self.expand_dim, device=device))
        self.log_scale = torch.nn.Parameter(torch.zeros(self.state_dim, device=device))

    def forward(self, u, x0=None):
        """
        Forward pass of S4 layer
        u: input sequence [batch_size, seq_len, features]
        x0: initial state [batch_size, state_dim]
        """
        batch_size, seq_len, _ = u.shape

        Lambda = -torch.exp(self.Lambda_real) + 1j * self.Lambda_imag

        B = self.B_real + 1j * self.B_imag
        C = self.C_real + 1j * self.C_imag

        B = B * torch.exp(self.log_scale)[None, :]
        C = C * torch.exp(-self.log_scale)[None, :]

        I = torch.ones(self.state_dim, device=self.device)
        denom = I - (self.dt / 2) * Lambda
        A = (I + (self.dt / 2) * Lambda) / denom
        B_disc = (self.dt * B) / denom

        if x0 is None:
            x = torch.zeros(batch_size, self.state_dim, dtype=torch.complex64, device=self.device)
        else:
            x = x0

        outputs = []
        hidden_states = []

        for i in range(seq_len):
            u_i = u[:, i, :]
            u_i = u_i.to(dtype=torch.complex64)
            x = A * x + torch.matmul(u_i, B_disc)

            y = torch.real(torch.matmul(x, C.t())) + torch.matmul(u_i.real, self.D)

            outputs.append(y)
            hidden_states.append(x)

        outputs = torch.stack(outputs, dim=1)
        final_state = x

        return outputs, final_state


class S4Model(torch.nn.Module):
    """
    Complete S4 model with embedding and projection layers
    """

    def __init__(self, input_dim, hidden_dim, output_dim=3, device="cpu"):
        super(S4Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.state_dim = hidden_dim
        self.device = device
        self.output_dim = output_dim
        self.embedding = torch.nn.Linear(input_dim, hidden_dim).to(device)

        self.s4_layer = S4Layer(state_dim=self.state_dim, expand_dim=hidden_dim, device=device)

        self.projection = torch.nn.Linear(hidden_dim, hidden_dim).to(device)
        self.activation = torch.nn.GELU()

        self.init_state_proj = torch.nn.Linear(self.output_dim, self.state_dim).to(device)

    def get_initial_state(self, batch_size, init_position):
        real_part = self.init_state_proj(init_position)
        imag_part = torch.zeros_like(real_part)
        return torch.complex(real_part, imag_part)

    def forward(self, inputs, states):
        batch_size = inputs.shape[0]

        inputs_expanded = inputs.unsqueeze(1)

        embedded = self.embedding(inputs_expanded)

        outputs, new_state = self.s4_layer(embedded, states)

        output = self.projection(outputs[:, -1, :])  # Take last timestep
        output = self.activation(output)

        return output, new_state


S4 = S4Model


class KKTFRNNs(nn.Module):
    """
    KKT Fractional-Order RNN Model based on the paper:
    "Multi-UUV Maneuvering Counter-Game for Dynamic Target Scenario Based on Fractional-Order Recurrent Neural Network"

    Implementation of the fractional-order RNN model described in Section V of the paper.
    The model is constructed based on KKT optimality conditions for strategy optimization.
    """

    def __init__(
        self,
        units: int,
        gamma: float = 0.99,
        theta: float = 1.0,
        input_dim: Optional[int] = None,
        output_dim: int = 3,
        device: str = "cpu",
        dt: float = 0.01,
    ):
        super(KKTFRNNs, self).__init__()
        self.units = units
        self.gamma = gamma
        self.theta = theta
        self.device = torch.device(device)
        self.output_dim = output_dim
        self.dt = dt

        if not (0 < gamma <= 2):
            raise ValueError("Fractional order gamma must be in (0, 2]")
        if gamma > 1:
            print(f"Warning: gamma={gamma} > 1 may cause instability. Recommend 0 < gamma <= 1")

        if input_dim is not None:
            self._build_layers(input_dim)
        else:
            self.weight_matrix = None
            self.bias_vector = None

    def _build_layers(self, input_dim: int):
        total_dim = self.units * 2
        self.weight_matrix = nn.Parameter(
            torch.randn(total_dim, total_dim, device=self.device) * 0.1
        )

        self.bias_vector = nn.Parameter(torch.randn(total_dim, device=self.device) * 0.1)

        self.cost_vector = nn.Parameter(torch.randn(total_dim, device=self.device) * 0.1)

        self.init_state_proj = nn.Linear(self.output_dim, total_dim).to(self.device)
        self.memory_states = None
        self.register_buffer("memory_index", torch.tensor(0, device=self.device))
        self.memory_size = 100

    def get_initial_state(self, batch_size: int, init_position: torch.Tensor) -> torch.Tensor:
        """Get initial state for the RNN."""
        if self.weight_matrix is None:
            self._build_layers(init_position.shape[-1])

        return self.init_state_proj(init_position)

    def _compute_rho(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        mid_dim = z.shape[1] // 2

        beta = z[:, :mid_dim]  # First half
        omega = z[:, mid_dim:]  # Second half

        G_beta = torch.matmul(beta, self.weight_matrix[:mid_dim, :mid_dim])
        constraint_term = omega + G_beta - self.bias_vector[:mid_dim]
        constraint_pos = F.relu(constraint_term)

        GT_constraint_pos = torch.matmul(constraint_pos, self.weight_matrix[:mid_dim, :mid_dim].T)
        rho_beta = -self.cost_vector[:mid_dim] - GT_constraint_pos
        rho_omega = constraint_pos - omega

        rho = torch.cat([rho_beta, rho_omega], dim=1)

        return rho

    def forward(
        self, inputs: torch.Tensor, states: torch.Tensor, elapsed: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(inputs, (tuple, list)):
            if len(inputs) > 1:
                elapsed = inputs[1]
                inputs = inputs[0]
            else:
                inputs = inputs[0]

        inputs = inputs.to(self.device)
        states = states.to(self.device)

        if self.weight_matrix is None:
            self._build_layers(inputs.shape[-1])

        batch_size = states.shape[0]

        if self.memory_states is None:
            self.memory_states = torch.zeros(self.memory_size, *states.shape, device=self.device)

        current_idx = self.memory_index.item() % self.memory_states.shape[0]
        self.memory_states[current_idx] = states.detach()
        self.memory_index += 1

        rho_z = self._compute_rho(states)
        fractional_deriv = self.theta * rho_z

        gamma_factor = torch.exp(torch.lgamma(torch.tensor(self.gamma + 1.0)))
        dt_gamma = self.dt**self.gamma
        new_states = states + (dt_gamma / gamma_factor) * fractional_deriv
        mid_dim = new_states.shape[1] // 2
        output = new_states[:, :mid_dim]

        return output, new_states

    def solve_optimization(
        self, max_iterations: int = 1000, tolerance: float = 1e-5
    ) -> torch.Tensor:
        batch_size = 1
        total_dim = self.units * 2
        z = torch.randn(batch_size, total_dim, device=self.device) * 0.1

        for iteration in range(max_iterations):
            rho_z = self._compute_rho(z)

            if torch.norm(rho_z) < tolerance:
                print(f"Converged after {iteration} iterations")
                break

            z = z - 0.01 * rho_z

        return z

    def get_mixed_strategies(
        self, equilibrium_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mid_dim = equilibrium_state.shape[1] // 2
        beta = equilibrium_state[:, :mid_dim]

        # Apply softmax to ensure probability constraints
        strategy_w = F.softmax(beta[:, : mid_dim // 2], dim=1)
        strategy_e = F.softmax(beta[:, mid_dim // 2 :], dim=1)

        return strategy_w, strategy_e
