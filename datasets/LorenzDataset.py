import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from lorenz.lorenz_systems import LorenzSystems

class LorenzDataset(Dataset):
    def __init__(self, system_type='63', x0=None, dt=0.01, Ns=5000, save_Dt=10, 
                 std=0.000, prev_time_steps=1, num_start_locations=1, **system_params):
        
        self.data_list = []
        self.target_list = []
        
        nx = 3 if system_type == '63' else system_params.get('N', 40)
        
        for _ in range(num_start_locations):
            if x0 is None or num_start_locations > 1:
                if system_type == '63':
                    current_x0 = np.random.normal(0, 10, 3)
                else:
                    current_x0 = np.random.rand(nx) * 0.1 + 8.0
            else:
                current_x0 = x0

            nt_truth = Ns * save_Dt + save_Dt
            
            print(f"Generating {system_type} trajectory...")
            x_truth = LorenzSystems.generate_trajectory(system_type, current_x0, dt, nt_truth, **system_params)
            
            # Add noise
            x_perturbed = x_truth + np.random.normal(loc=0, scale=std, size=x_truth.shape)
            
            # Sample data
            d = x_perturbed[:nt_truth-save_Dt:save_Dt]
            t = x_perturbed[save_Dt:nt_truth:save_Dt]
            
            self.data_list.append(d)
            self.target_list.append(t)

        self.data = np.vstack(self.data_list)
        self.target = np.vstack(self.target_list)
        
        self.scaler = StandardScaler()
        self.prev_time_steps = prev_time_steps
        self.scaler.fit(self.data)
        
        self.data = self.transform(self.data)
        self.target = self.transform(self.target)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        max_future = 10
        idx = max(self.prev_time_steps, idx)
        idx = min(idx, len(self.data) - max_future)
        
        input_seq = torch.tensor(self.data[idx-self.prev_time_steps:idx].flatten(), dtype=torch.float32)
        targets = torch.tensor(self.target[idx-1 : idx-1 + max_future], dtype=torch.float32)
        
        return input_seq, targets

    def inverse_transform(self, x):
        return self.scaler.inverse_transform(x)

    def transform(self, x):
        return self.scaler.transform(x)
