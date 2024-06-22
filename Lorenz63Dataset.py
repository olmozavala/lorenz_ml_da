
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# %% Lorenz 63 system manual?
def F(x, sigma=10, beta=8/3, rho=28):
    dx = sigma * (x[1] - x[0])
    dy = x[0] * (rho - x[2]) - x[1]
    dz = x[0] * x[1] - beta * x[2]
    return np.array([dx, dy, dz])


class Lorenz63Dataset(Dataset):

    def __init__(self, x0=4.0, y0=10.0, z0=1.0, save_Dt=10, dt=0.01, 
                 Ns=5000, std=.001, Nx=3, prev_time_steps=1):

        initial_conditions = np.array([x0, y0, z0])
        # Spinup run length and data generating run length
        Nt_truth = Ns*save_Dt+save_Dt
        x_truth = np.empty((Nt_truth,Nx))
        x_truth[0] = initial_conditions
        x = initial_conditions.copy()  # Initialize x with initial conditions
        # %%
        with tqdm(total=Nt_truth - 1, desc='L63 trajectory') as progress:
            for i in range(1,Nt_truth):  # Note: -1 because the initial condition is already added
                x += dt * F(x)
                x_truth[i] =x   # Append the new state to the trajectory
                progress.set_postfix_str(x, refresh=False)
                progress.update()

        x_perturbed_all_t = x_truth + np.random.normal(loc=0, scale=std, size=(Nt_truth, Nx))#Used to introduce noise to the trajectory of the L63 system. Does help avoid overfitting.
        self.data = x_perturbed_all_t[:Nt_truth-save_Dt:save_Dt]#Spaced by save_Dt. x_perturbed has perturbed trajectory of the L63. Sampled starting from beginning and each sample is separated by a lead time of save_Dt 
        self.target = x_perturbed_all_t[save_Dt:Nt_truth:save_Dt]#Represents state of system at future points, shifted by save_Dt.
        self.scaler = StandardScaler()
        self.prev_time_steps = prev_time_steps
        self.scaler.fit(self.data)
        self.data = self.transform(self.data)
        self.target = self.transform(self.target)

    def __len__(self):
        return len(self.data)

    def get_all_data(self):
        return self.data, self.target

    def __getitem__(self, idx):
        idx = min(idx, len(self.data)-self.prev_time_steps-1) # To avoid negative index
        return torch.tensor(self.data[idx:idx+self.prev_time_steps].flatten(), dtype=torch.float32), torch.tensor(self.target[idx], dtype=torch.float32)

    def inverse_transform(self, x):
        return self.scaler.inverse_transform(x)

    def transform(self, x):
        return self.scaler.transform(x)