# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from os.path import join
import tqdm
from SurrogateModel import SurrogateModel
from lorenz.lorenz_systems import LorenzSystems

torch.set_default_dtype(torch.float64)

# ============================================================
# Model initialization
# ============================================================
model_dir = "models"

def init_models(n_steps):
    if n_steps == 1:
        print("Initializing single time step models")
        model_paths = {
            'DenseNN': join(model_dir, 'DenseNN_L63_trial1_1775267887_best_model.pth'),
            'ResDenseNN': join(model_dir, 'ResDenseNN_L63_trial1_1775267929_best_model.pth'),
            'LSTMNN': join(model_dir, 'LSTMNN_L63_trial1_1775267969_best_model.pth'),
            'RNN_tanh': join(model_dir, 'RNN_L63_trial1_1775269773_best_model.pth'),
            'RNN_relu': join(model_dir, 'RNN_L63_trial1_1775269849_best_model.pth'),
        }
    elif n_steps == 4:
        print("Initializing 4 time step models")
        model_paths = {
            'DenseNN': join(model_dir, 'DenseNN_L63_trial1_1774989212_best_model.pth'),
            'ResDenseNN': join(model_dir, 'ResDenseNN_L63_trial1_1774989274_best_model.pth'),
            'LSTMNN': join(model_dir, 'LSTMNN_L63_trial1_1774989300_best_model.pth'),
            'RNN_tanh': join(model_dir, 'RNN_L63_trial1_1774989331_best_model.pth'),
            'RNN_relu': join(model_dir, 'RNN_L63_trial1_1775047936_best_model.pth'),
        }
    else:
        raise ValueError(f"Invalid number of steps: {n_steps}")

    surrogates_palette = {
        'Lorenz63': '#000000',
        'Truth': '#000000',
        'DenseNN': '#D85A30',
        'ResDenseNN': '#7F770A',
        'LSTMNN': '#7F77DD',
        'RNN_relu': '#1D9E75',
        'RNN_tanh': '#1DDE75',
    }

    surrogates = {
        'DenseNN': SurrogateModel(model_paths['DenseNN']),
        'ResDenseNN': SurrogateModel(model_paths['ResDenseNN']),
        'LSTMNN': SurrogateModel(model_paths['LSTMNN']),
        'RNN_relu': SurrogateModel(model_paths['RNN_relu']),
        'RNN_tanh': SurrogateModel(model_paths['RNN_tanh']),
    }

    return surrogates, surrogates_palette

surrogates, surrogates_palette = init_models(1)

VAR_NAMES = ['x', 'y', 'z']
_SEED = 132
np.random.seed(_SEED)

Ne = 100
Nx = 3
d0 = 0.01
n_steps = 200
dt = 0.01
time = np.arange(0, n_steps * dt + dt, dt)

x0 = np.ones(3)
Xf_pool = np.outer(np.ones(Ne), x0) + d0 * np.random.randn(Ne, Nx)

surrogate_pools = {name: [] for name in surrogates.keys()}
surrogate_series = {name: [] for name in surrogates.keys()}

for name, model in surrogates.items():
    pool = Xf_pool.copy()
    pbar = tqdm.tqdm(range(Ne), desc=f"Running {name}", unit="member", leave=False)
    for e in pbar:
        pbar.set_postfix(member=e, refresh=False)
        sol = model.rollout(pool[e, :], num_steps=n_steps)
        surrogate_pools[name].append(sol)
    surrogate_series[name] = np.stack(surrogate_pools[name])
    tqdm.tqdm.write(f"Ensemble pool ready for {name}")

# Lorenz baseline pool (perfect model)
pool_lorenz = Xf_pool.copy()
lorenz_pools = []
for e in tqdm.tqdm(range(Ne), desc="Running Lorenz63", unit="member", leave=False):
    pbar.set_postfix(member=e, refresh=False)
    sol = LorenzSystems.generate_trajectory_fast('63', pool_lorenz[e, :], dt, n_steps + 1)
    lorenz_pools.append(sol)
lorenz_series = np.stack(lorenz_pools)
print(f"Ensemble pool ready for Lorenz63 (baseline)")
    
# %%
fig, ax = plt.subplots(3, 1, figsize=(20, 10))
model_type = 'ResDenseNN'
for ii, var_name in enumerate(VAR_NAMES):
    ax[ii].plot(surrogate_series[model_type][..., ii].T, color=surrogates_palette[model_type])
    ax[ii].plot(lorenz_series[..., 1:, ii].T, color=surrogates_palette['Lorenz63'])


    #ax[ii].legend()
    #ax[ii].set_title(f'{var_name}')
plt.tight_layout()
plt.show()
# %%
