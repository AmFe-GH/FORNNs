import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        output = torch.matmul(input, self.B.t()) + self.Theta.t()
        output = torch.sigmoid(output)
        return torch.matmul(output, self.A.t())







class CTRNNCell(nn.Module):
    def __init__(self, units, method, num_unfolds=None, tau=1, input_dim=None, device="cpu"):
        super(CTRNNCell, self).__init__()

        self.device = torch.device(device)

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

        # 如果提供了input_dim，直接初始化权重
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

        # 初始化权重
        nn.init.xavier_uniform_(self.kernel.weight)
        nn.init.orthogonal_(self.recurrent_kernel.weight)

    def _ode_func(self, t, y, args):
        """ODE函数，用于torchode"""
        inputs = args["inputs"]
        return self.dfdt(inputs, y)

    def forward(self, inputs, hidden_state, elapsed=1.0):
        # 处理输入
        if isinstance(inputs, (tuple, list)):
            if len(inputs) > 1:
                elapsed = inputs[1]
                inputs = inputs[0]
            else:
                inputs = inputs[0]

        # 动态构建权重
        if self.kernel is None:
            self._build_weights(inputs.shape[-1])

        inputs = inputs.to(self.device)
        hidden_state = hidden_state.to(self.device)

        if self.method == "dopri5":
            # 使用torchode求解
            if not isinstance(elapsed, torch.Tensor):
                elapsed = torch.tensor(elapsed, device=self.device)

            # 处理批次维度
            if elapsed.dim() > 0 and len(elapsed) > 1:
                # 批次处理
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


class LSTMCell(nn.Module):
    def __init__(self, units, input_dim=None, device="cpu"):
        super(LSTMCell, self).__init__()
        self.units = units
        self.device = torch.device(device)

        if input_dim is not None:
            self._build_weights(input_dim)
        else:
            self.input_kernel = None

    def _build_weights(self, input_dim):
        self.input_kernel = nn.Linear(input_dim, 4 * self.units).to(self.device)
        self.recurrent_kernel = nn.Linear(self.units, 4 * self.units, bias=False).to(self.device)

        # 初始化
        nn.init.xavier_uniform_(self.input_kernel.weight)
        nn.init.orthogonal_(self.recurrent_kernel.weight)
        nn.init.zeros_(self.input_kernel.bias)

    def get_initial_state(self, batch_size):
        return (
            torch.zeros(batch_size, self.units, device=self.device),
            torch.zeros(batch_size, self.units, device=self.device),
        )

    def forward(self, inputs, states, elapsed=1.0):
        cell_state, output_state = states

        # 处理输入连接
        if isinstance(inputs, (tuple, list)):
            if len(inputs) > 1:
                inputs = torch.cat([inputs[0], inputs[1]], dim=-1)
            else:
                inputs = inputs[0]

        inputs = inputs.to(self.device)
        cell_state = cell_state.to(self.device)
        output_state = output_state.to(self.device)

        # 动态构建
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


class ODELSTM(nn.Module):
    def __init__(self, units, input_dim=None, device="cpu"):
        super(ODELSTM, self).__init__()
        self.units = units
        self.device = torch.device(device)
        self.ctrnn = CTRNNCell(
            self.units, num_unfolds=4, method="euler", input_dim=self.units, device=device
        )

        if input_dim is not None:
            self._build_weights(input_dim)
        else:
            self.input_kernel = None

    def _build_weights(self, input_dim):
        self.input_kernel = nn.Linear(input_dim, 4 * self.units).to(self.device)
        self.recurrent_kernel = nn.Linear(self.units, 4 * self.units, bias=False).to(self.device)

        nn.init.xavier_uniform_(self.input_kernel.weight)
        nn.init.orthogonal_(self.recurrent_kernel.weight)
        nn.init.zeros_(self.input_kernel.bias)

    def get_initial_state(self, batch_size):
        return (
            torch.zeros(batch_size, self.units, device=self.device),
            torch.zeros(batch_size, self.units, device=self.device),
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
        forget_gate = torch.sigmoid(fg + 3.0)  # 注意这里是+3.0
        output_gate = torch.sigmoid(og)

        new_cell = cell_state * forget_gate + input_activation * input_gate
        ode_input = torch.tanh(new_cell) * output_gate

        # ODE组件
        ode_output = self.ctrnn(ode_input, ode_state, elapsed)

        return ode_output, (new_cell, ode_output)


class CTGRU(nn.Module):
    def __init__(self, units, M=8, input_dim=None, device="cpu"):
        super(CTGRU, self).__init__()
        self.units = units
        self.M = M
        self.device = torch.device(device)

        # 预计算tau表
        self.ln_tau_table = np.empty(self.M)
        self.tau_table = np.empty(self.M)
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

    def get_initial_state(self, batch_size):
        return torch.zeros(batch_size, self.units * self.M, device=self.device)

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

        # 重塑状态
        h_hat = states.view(batch_dim, self.units, self.M)
        h = torch.sum(h_hat, dim=2)

        # 检索
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

        # 时间更新
        base_term = (1 - ski) * h_hat + ski * qk
        exp_term = torch.exp(-elapsed / self.tau_table_tensor)
        exp_term = exp_term.view(1, 1, self.M)
        h_hat_next = base_term * exp_term

        # 计算新状态
        h_next = torch.sum(h_hat_next, dim=2)
        h_hat_next_flat = h_hat_next.view(batch_dim, self.units * self.M)

        return h_next, h_hat_next_flat


class VanillaRNN(nn.Module):
    def __init__(self, units, input_dim=None, device="cpu"):
        super(VanillaRNN, self).__init__()
        self.units = units
        self.device = torch.device(device)

        if input_dim is not None:
            self._build_layers(input_dim)
        else:
            self.layer = None

        self.tau = nn.Parameter(torch.full((self.units,), 0.1, device=self.device))

    def _build_layers(self, input_dim):
        fused_dim = input_dim + self.units
        self.layer = nn.Linear(fused_dim, self.units).to(self.device)
        self.out_layer = nn.Linear(self.units, self.units).to(self.device)

    def get_initial_state(self, batch_size):
        return torch.zeros(batch_size, self.units, device=self.device)

    def forward(self, inputs, states, elapsed=1.0):
        if isinstance(inputs, (tuple, list)):
            if len(inputs) > 1:
                elapsed = inputs[1]
                inputs = inputs[0]
            else:
                inputs = inputs[0]

        inputs = inputs.to(self.device)
        states = states.to(self.device)

        if self.layer is None:
            self._build_layers(inputs.shape[-1])

        fused_input = torch.cat([inputs, states], dim=-1)
        new_states = self.out_layer(torch.tanh(self.layer(fused_input))) - elapsed * self.tau

        return new_states, new_states


class BidirectionalRNN(nn.Module):
    def __init__(self, units, input_dim=None, device="cpu"):
        super(BidirectionalRNN, self).__init__()
        self.units = units
        self.device = torch.device(device)

        # 传递device参数给子模块
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

    def get_initial_state(self, batch_size):
        return (
            torch.zeros(batch_size, self.units, device=self.device),  # LSTM cell state
            torch.zeros(batch_size, self.units, device=self.device),  # LSTM hidden state
            torch.zeros(batch_size, self.units, device=self.device),  # CTRNN state
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
    def __init__(self, units, input_dim=None, device="cpu"):
        super(GRUD, self).__init__()
        self.units = units
        self.device = torch.device(device)

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

    def get_initial_state(self, batch_size):
        return torch.zeros(batch_size, self.units, device=self.device)

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

        # 计算新状态
        ht = zt * h_hat + (1.0 - zt) * h_tilde

        return ht, ht


class PhasedLSTM(nn.Module):
    def __init__(self, units, input_dim=None, device="cpu"):
        super(PhasedLSTM, self).__init__()
        self.units = units
        self.device = torch.device(device)

        if input_dim is not None:
            self._build_weights(input_dim)
        else:
            self.input_kernel = None

        # 时间相关参数
        self.tau = nn.Parameter(torch.zeros(1, device=self.device))
        self.ron = nn.Parameter(torch.zeros(1, device=self.device))
        self.s = nn.Parameter(torch.zeros(1, device=self.device))

    def _build_weights(self, input_dim):
        self.input_kernel = nn.Linear(input_dim, 4 * self.units).to(self.device)
        self.recurrent_kernel = nn.Linear(self.units, 4 * self.units, bias=False).to(self.device)

        nn.init.xavier_uniform_(self.input_kernel.weight)
        nn.init.orthogonal_(self.recurrent_kernel.weight)
        nn.init.zeros_(self.input_kernel.bias)

    def get_initial_state(self, batch_size):
        return (
            torch.zeros(batch_size, self.units, device=self.device),
            torch.zeros(batch_size, self.units, device=self.device),
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

        # 泄漏常数
        alpha = 0.001
        # 确保这些值为正
        tau = F.softplus(self.tau)
        s = F.softplus(self.s)
        ron = F.softplus(self.ron)

        if not isinstance(elapsed, torch.Tensor):
            elapsed = torch.tensor(elapsed, device=self.device)

        phit = torch.fmod(elapsed - s, tau) / tau

        # 计算kt
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
    def __init__(self, units, num_unfolds=4, input_dim=None, device="cpu"):
        super(GRUODE, self).__init__()
        self.units = units
        self.num_unfolds = num_unfolds
        self.device = torch.device(device)

        if input_dim is not None:
            self._build_layers(input_dim)
        else:
            self.reset_gate = None

    def _build_layers(self, input_dim):
        fused_dim = input_dim + self.units
        self.reset_gate = nn.Linear(fused_dim, self.units).to(self.device)
        self.detect_signal = nn.Linear(fused_dim, self.units).to(self.device)
        self.update_gate = nn.Linear(fused_dim, self.units).to(self.device)

        # 重置门偏置初始化为1
        nn.init.constant_(self.reset_gate.bias, 1.0)

    def get_initial_state(self, batch_size):
        return torch.zeros(batch_size, self.units, device=self.device)

    def _dh_dt(self, inputs, states):
        fused_input = torch.cat([inputs, states], dim=-1)
        rt = torch.sigmoid(self.reset_gate(fused_input))
        zt = torch.sigmoid(self.update_gate(fused_input))

        reset_value = torch.cat([inputs, rt * states], dim=-1)
        gt = torch.tanh(self.detect_signal(reset_value))

        # 计算导数
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


class HawkLSTMCell(nn.Module):
    def __init__(self, units, input_dim=None, device="cpu"):
        super(HawkLSTMCell, self).__init__()
        self.units = units
        self.device = torch.device(device)

        if input_dim is not None:
            self._build_weights(input_dim)
        else:
            self.input_kernel = None

    def _build_weights(self, input_dim):
        self.input_kernel = nn.Linear(input_dim, 7 * self.units).to(self.device)
        self.recurrent_kernel = nn.Linear(self.units, 7 * self.units, bias=False).to(self.device)

        nn.init.xavier_uniform_(self.input_kernel.weight)
        nn.init.orthogonal_(self.recurrent_kernel.weight)
        nn.init.zeros_(self.input_kernel.bias)

    def get_initial_state(self, batch_size):
        return (
            torch.zeros(batch_size, self.units, device=self.device),  # c
            torch.zeros(batch_size, self.units, device=self.device),  # c_bar
            torch.zeros(batch_size, self.units, device=self.device),  # h
        )

    def forward(self, inputs, states):
        # inputs应该是(k, delta_t)的元组
        k, delta_t = inputs
        c, c_bar, h = states

        k = k.to(self.device)
        if not isinstance(delta_t, torch.Tensor):
            delta_t = torch.tensor(delta_t, device=self.device)
        delta_t = delta_t.to(self.device)

        c = c.to(self.device)
        c_bar = c_bar.to(self.device)
        h = h.to(self.device)

        if self.input_kernel is None:
            self._build_weights(k.shape[-1])

        z = self.input_kernel(k) + self.recurrent_kernel(h)
        i, ig, fg, og, ig_bar, fg_bar, d = torch.chunk(z, 7, dim=-1)

        input_activation = torch.tanh(i)
        input_gate = torch.sigmoid(ig)
        input_gate_bar = torch.sigmoid(ig_bar)
        forget_gate = torch.sigmoid(fg)
        forget_gate_bar = torch.sigmoid(fg_bar)
        output_gate = torch.sigmoid(og)
        delta_gate = F.softplus(d)

        new_c = c * forget_gate + input_activation * input_gate
        new_c_bar = c_bar * forget_gate_bar + input_activation * input_gate_bar

        c_t = new_c_bar + (new_c - new_c_bar) * torch.exp(-delta_gate * delta_t)
        output_state = torch.tanh(c_t) * output_gate

        return output_state, (new_c, new_c_bar, output_state)


# 使用示例
if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)

    # 可以灵活指定设备
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 示例：使用CTRNN
    batch_size = 32
    seq_len = 10
    input_dim = 64
    hidden_dim = 128

    # 创建模型，指定设备
    ctrnn = CTRNNCell(
        units=hidden_dim, method="euler", num_unfolds=4, input_dim=input_dim, device=device
    )

    # 创建输入数据
    inputs = torch.randn(batch_size, input_dim, device=device)
    hidden_state = torch.zeros(batch_size, hidden_dim, device=device)
    elapsed = 1.0

    # 前向传播
    output = ctrnn(inputs, hidden_state, elapsed)
    print(f"CTRNN output shape: {output.shape}")
    print(f"CTRNN output device: {output.device}")

    # 示例：使用LSTM
    lstm = LSTMCell(units=hidden_dim, input_dim=input_dim, device=device)
    cell_state, hidden_state = lstm.get_initial_state(batch_size)

    output, (new_cell, new_hidden) = lstm(inputs, (cell_state, hidden_state))
    print(f"LSTM output shape: {output.shape}")
    print(f"LSTM output device: {output.device}")

    # 示例：在不同设备上创建模型
    cpu_model = ODELSTM(units=64, input_dim=32, device="cpu")
    if torch.cuda.is_available():
        gpu_model = ODELSTM(units=64, input_dim=32, device="cuda:0")
        print("GPU model created successfully")
