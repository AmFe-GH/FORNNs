from model import GRUODE, PhasedLSTM, GRUD, BidirectionalRNN, CTGRU, ODELSTM, AugmentedLSTM, CTRNN, S4, KKTFRNNs
import numpy as np
import utils
import torch
import os
import pdb
import lib
from pathlib import Path
from lib.ode_rnn import ODE_RNN as ODERNN
from lib.ode_func import ODEFunc
from lib.diffeq_solver import DiffeqSolver
from lib.encoder_decoder import Encoder_z0_ODE_RNN
from lib.rnn_baselines import Classic_RNN

# Simulation_system = str(input("Simulation system:  Lorenz or Chua \n"))
Simulation_system = "Chua"
assert Simulation_system in ["Lorenz", "Chua"]  # Restrict to supported dynamical systems

model_list = {
    #Available RNN-based models for dynamical system simulation
    "GRUODE": GRUODE,
    "PhasedLSTM": PhasedLSTM,
    "GRUD": GRUD,
    "BidirectionalRNN": BidirectionalRNN,
    "CTGRU": CTGRU,
    "ODELSTM": ODELSTM,
    "AugmentedLSTM": AugmentedLSTM,
    "CTRNN": CTRNN,
    "ODERNN": ODERNN,
    "RNNDecay": Classic_RNN,
    "S4": S4,  
    "KKTFRNNs": KKTFRNNs,
}

target_system_config = utils.load_config(config_file="base_model_hyperparameters.yaml")[
    Simulation_system
]

dtype = torch.float32
device = target_system_config["device"]
n_of_func_right = target_system_config["n_of_func_right"]
N_step = target_system_config["N_step"]
test_step = target_system_config["test_step"]

real_x, _ = utils.fols_Fun(
    target_system_config["alpha"],
    np.array(target_system_config["x0"])[np.newaxis, :],
    0,
    target_system_config["width_step"],
    N_step + test_step,
    n_of_func_right,
    False,
    getattr(utils, f"X_rightfunc_{Simulation_system}_numpy"),
)
real_x = torch.tensor(real_x, dtype=dtype, device=device)
assert len(real_x) == N_step + test_step + 1  # +1 accounts for initial state at t=0

for model_name, cell_class in model_list.items():
    model_config = target_system_config[model_name]
    save_fig_path = f"./Figure/{Simulation_system}/{model_name}/"
    os.makedirs(save_fig_path, exist_ok=True)
    save_model_path = Path("./trained_model/") / f"{model_name}_{Simulation_system}.pth"
    save_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(model_config["random_seed"])
    np.random.seed(model_config["random_seed"])
    used_N_step = target_system_config["N_step"]
    used_test_step = target_system_config["test_step"]

    if model_name == "CTRNN":
        model_cell = cell_class(
            units=model_config["hidden_dim"],
            method="euler",
            num_unfolds=10,
            tau=1,
            input_dim=model_config["hidden_dim"],
            device=device,
        )
    elif model_name == "S4":
        model_cell = S4(
            input_dim=model_config["hidden_dim"],
            hidden_dim=model_config["hidden_dim"],
            device=device,
        ).to(device)
    elif model_name == "ODERNN":
        ode_func_net = torch.nn.Sequential(
            torch.nn.Linear(model_config["hidden_dim"], 50),
            torch.nn.Tanh(),
            torch.nn.Linear(50, model_config["hidden_dim"]),
        ).to(device)

        ode_func = ODEFunc(
            input_dim=model_config["hidden_dim"],
            latent_dim=model_config["hidden_dim"],
            ode_func_net=ode_func_net,
            device=device,
        ).to(device)

        diffeq_solver = DiffeqSolver(
            input_dim=model_config["hidden_dim"],
            ode_func=ode_func,
            method="rk4",
            latents=model_config["hidden_dim"],
            device=device,
        ).to(device)

        model_cell = ODERNN(
            input_dim=n_of_func_right,
            latent_dim=model_config["hidden_dim"],
            device=device,
            z0_diffeq_solver=diffeq_solver,
            n_gru_units=model_config["hidden_dim"],
            n_units=model_config["hidden_dim"],
        ).to(device) 
    elif model_name == "RNNDecay":
        model_cell = Classic_RNN(
            input_dim=n_of_func_right,
            latent_dim=model_config["hidden_dim"],
            device=device,
            cell="expdecay",
            n_units=model_config["hidden_dim"],
            concat_mask=True,
        ).to(device)
    elif model_name == "KKTFRNNs":
        model_cell = KKTFRNNs(
            units=model_config["hidden_dim"],
            gamma=model_config.get("gamma", 0.99), 
            theta=model_config.get("theta", 1.0),
            input_dim=model_config["hidden_dim"],
            output_dim=n_of_func_right,
            device=device,
            dt=model_config.get("dt", 0.01)         # Time step for numerical integration
        ).to(device)
    else:
        model_cell = cell_class(
            units=model_config["hidden_dim"], input_dim=model_config["hidden_dim"], device=device
        )
    model_output_proj = torch.nn.Sequential(
        torch.nn.Linear(model_config["hidden_dim"], model_config["hidden_dim"]),
        torch.nn.Tanh(),
        torch.nn.Linear(model_config["hidden_dim"], n_of_func_right),
    ).to(device)
    optimizer = torch.optim.Adam(
            list(model_cell.parameters()) + list(model_output_proj.parameters()),
            lr=model_config["lr"],
        )
    zero_input = torch.zeros(
        model_config["hidden_dim"],
        dtype=dtype,
        device=device,
        requires_grad=False,
    ).reshape(1, -1)
    init_position = torch.tensor(
        target_system_config["x0"], dtype=dtype, device=device, requires_grad=False
    ).reshape(1, -1)

    if save_model_path.exists():
        checkpoint = torch.load(save_model_path)
        model_cell.load_state_dict(checkpoint["model_cell_weights"])
        model_output_proj.load_state_dict(checkpoint["model_output_proj_weights"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        base_epoch = checkpoint["epoch"] + 1
        print(f"Loading checkpoint: {save_model_path}")
    else:
        base_epoch = 0

    if model_name == "ODERNN":
        min_loss = 1e10
        for epoch_index in range(base_epoch, base_epoch + model_config["epoch_num"]):
            # Generate time steps for the entire simulation period
            time_steps = (
                torch.arange(used_N_step + used_test_step + 1, dtype=dtype, device=device)
                * target_system_config["width_step"]
            )
            data = real_x.unsqueeze(0).to(device)  # Add batch dimension and ensure correct device
            mask = torch.ones_like(
                data, dtype=dtype, device=device
            )


            outputs, _ = model_cell.get_reconstruction(time_steps, data, time_steps, mask=mask)
            output_sequence = outputs.squeeze(0).squeeze(0)

            loss = torch.mean(
                (output_sequence[: used_N_step + 1, :] - real_x[: used_N_step + 1, :]) ** 2
            )
            epsilon_train = torch.sqrt(
                torch.sum(
                    (output_sequence[: used_N_step + 1, :] - real_x[: used_N_step + 1, :]) ** 2,
                    dim=-1,
                )
            ).max()
            epsilon_test = torch.sqrt(
                torch.sum(
                    (output_sequence[-used_test_step:, :] - real_x[-used_test_step:, :]) ** 2,
                    dim=-1,
                )
            ).max()

            if epoch_index % target_system_config["terminal_log_interval"] == 0:
                print(
                    f"{model_name} epoch: {epoch_index} loss: {loss.item()}, epsilon_train: {epsilon_train.item()} ({used_N_step}), epsilon_test: {epsilon_test.item()} ({used_test_step})"
                )

            optimizer.zero_grad()
            loss.backward()

            if any(
                torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                for param in model_cell.parameters()
                if param.grad is not None
            ):
                torch.nn.utils.clip_grad_norm_(model_cell.parameters(), max_norm=10.0)
                print("NaN or INF detected in gradients.")
            else:
                optimizer.step()
                if epoch_index % target_system_config["fig_log_interval"] == 0:
                    utils.draw3d(
                        real_x,
                        output_sequence,
                        show=False,
                        save_path=save_fig_path
                        + str(epoch_index)
                        + "_"
                        + str(int(loss))
                        + f"_{epsilon_train:.1f}"
                        + f"_{epsilon_test:.1f}"
                        + "_",
                    )
                    if min_loss > loss.item():
                        min_loss = loss.item()
                        torch.save(
                            {
                                "model_cell_weights": model_cell.state_dict(),
                                "model_output_proj_weights": model_output_proj.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "epoch": epoch_index,
                            },
                            save_model_path,
                        )
    elif model_name == "RNNDecay":
        min_loss = 1e10
        for epoch_index in range(base_epoch, base_epoch + model_config["epoch_num"]):
            time_steps = (
                torch.arange(used_N_step + used_test_step + 1, dtype=dtype, device=device)
                * target_system_config["width_step"]
            )

            data = real_x.unsqueeze(0).to(device)
            mask = torch.ones_like(data, dtype=dtype, device=device)
            outputs, _ = model_cell.get_reconstruction(time_steps, data, time_steps, mask=mask)
            output_sequence = outputs.squeeze(0).squeeze(0)

            loss = torch.mean(
                (output_sequence[: used_N_step + 1, :] - real_x[: used_N_step + 1, :]) ** 2
            )
            epsilon_train = torch.sqrt(
                torch.sum(
                    (output_sequence[: used_N_step + 1, :] - real_x[: used_N_step + 1, :]) ** 2,
                    dim=-1,
                )
            ).max()
            epsilon_test = torch.sqrt(
                torch.sum(
                    (output_sequence[-used_test_step:, :] - real_x[-used_test_step:, :]) ** 2,
                    dim=-1,
                )
            ).max()

            if epoch_index % target_system_config["terminal_log_interval"] == 0:
                print(
                    f"{model_name} epoch: {epoch_index} loss: {loss.item()}, epsilon_train: {epsilon_train.item()} ({used_N_step}), epsilon_test: {epsilon_test.item()} ({used_test_step})"
                )

            optimizer.zero_grad()
            loss.backward()

            if any(
                torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                for param in model_cell.parameters()
                if param.grad is not None
            ):
                torch.nn.utils.clip_grad_norm_(model_cell.parameters(), max_norm=10.0)
                print("NaN or INF detected in gradients.")
            else:
                optimizer.step()
                if epoch_index % target_system_config["fig_log_interval"] == 0:
                    utils.draw3d(
                        real_x,
                        output_sequence,
                        show=False,
                        save_path=save_fig_path
                        + str(epoch_index)
                        + "_"
                        + str(int(loss))
                        + f"_{epsilon_train:.1f}"
                        + f"_{epsilon_test:.1f}"
                        + "_",
                    )
                    if min_loss > loss.item():
                        min_loss = loss.item()
                        torch.save(
                            {
                                "model_cell_weights": model_cell.state_dict(),
                                "model_output_proj_weights": model_output_proj.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "epoch": epoch_index,
                            },
                            save_model_path,
                        )
    else:
        min_loss = 1e10
        for epoch_index in range(base_epoch, base_epoch + model_config["epoch_num"]):
            init_state = model_cell.get_initial_state(1, init_position)
            output_list = [init_position[:]]
            current_state = init_state

            for step_index in range(used_N_step + used_test_step):
                output_of_step = model_cell(zero_input, current_state)
                if isinstance(output_of_step, tuple):
                    output, current_state = output_of_step
                elif isinstance(output_of_step, torch.Tensor):
                    output = output_of_step
                    current_state = output_of_step
                output = model_output_proj(output)
                output_list.append(output + output_list[-1])
            output_sequence = torch.stack(output_list, dim=1).squeeze(0)
            loss = torch.mean(
                (output_sequence[: used_N_step + 1, :] - real_x[: used_N_step + 1, :]) ** 2
            )
            epsilon_train = torch.sqrt(
                torch.sum(
                    (output_sequence[: used_N_step + 1, :] - real_x[: used_N_step + 1, :]) ** 2,
                    dim=-1,
                )
            ).max()
            epsilon_test = torch.sqrt(
                torch.sum(
                    (output_sequence[-used_test_step:, :] - real_x[-used_test_step:, :]) ** 2,
                    dim=-1,
                )
            ).max()
            if epoch_index % target_system_config["terminal_log_interval"] == 0:
                print(
                    f"{model_name} epoch: {epoch_index} loss: {loss.item()}, epsilon_train: {epsilon_train.item()} ({used_N_step}), epsilon_test: {epsilon_test.item()} ({used_test_step})"
                )

            optimizer.zero_grad()
            loss.backward()

            if any(
                torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                for param in model_cell.parameters()
                if param.grad is not None
            ):
                torch.nn.utils.clip_grad_norm_(model_cell.parameters(), max_norm=10.0)

                print("NaN or INF detected in gradients.")

            else:
                optimizer.step()
                if epoch_index % target_system_config["fig_log_interval"] == 0:
                    utils.draw3d(
                        real_x,
                        output_sequence,
                        show=False,
                        save_path=save_fig_path
                        + str(epoch_index)
                        + "_"
                        + str(int(loss))
                        + f"_{epsilon_train:.1f}"
                        + f"_{epsilon_test:.1f}"
                        + "_",
                    )
                    if min_loss > loss.item():
                        min_loss = loss.item()
                        torch.save(
                            {
                                "model_cell_weights": model_cell.state_dict(),
                                "model_output_proj_weights": model_output_proj.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "epoch": epoch_index,
                            },
                            save_model_path,
                        )
