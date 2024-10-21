import torch


def set_Hyperparameters(Simulation_system):

    if Simulation_system == 'Lorenz':
        N_of_func_right = 1000
        n_of_func_right = 3
        dtype = torch.float32
        device = torch.device('cuda:0')
        N_step = 500
        test_step = 100
        alpha = 0.996
        beta = 0.9
        epoch_number = 20
        width_step = 0.01
        x0 = torch.tensor([-8., 7, 27], dtype=dtype, device=device)
        random_seed = 490
        lr = 1e-4
        LDN_lr = 4e-3
        epochs_LND = 100000
    elif Simulation_system == 'Chua':
        N_of_func_right = 5
        n_of_func_right = 3
        dtype = torch.float32
        device = torch.device('cuda:0')
        N_step = 300
        test_step = 60
        alpha = 0.99
        beta = 0.9
        epoch_number = 30
        width_step = 0.01
        x0 = torch.tensor([1.45305, -4.36956, 0.15034],
                          dtype=dtype, device=device)
        random_seed = 42
        lr = 4e-4
        LDN_lr = 1e-3
        epochs_LND = 400000
    elif Simulation_system == 'XOR':
        N_of_func_right = 2
        n_of_func_right = 1
        dtype = torch.float32
        device = torch.device('cuda:0')
        N_step = 10
        test_step = None
        alpha = 0.94
        beta = None
        epoch_number = 20000
        width_step = None
        x0 = torch.tensor([0.],
                          dtype=dtype, device=device)
        random_seed = 45
        lr = 1e-3
        LDN_lr = None
        epochs_LND = None
    else:
        raise ValueError(
            'Simulation_system must be "Lorenz" , "XOR" or "Chua"')
    return N_of_func_right, n_of_func_right, dtype, device, N_step, test_step, alpha, beta, epoch_number, width_step, x0, random_seed, lr, LDN_lr, epochs_LND
