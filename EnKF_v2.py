# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from os.path import join
from SurrogateModel import SurrogateModel
from lorenz.lorenz_systems import LorenzSystems

torch.set_default_dtype(torch.float64)

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
model_name = 'DenseNN'
np.random.seed(10)
spinup_steps = 2000
x0 = np.random.randn(3).astype(np.float64)
dt = 0.01
n_steps = len(np.arange(0, 30, dt))

x = LorenzSystems.generate_trajectory_fast('63', x0, dt, spinup_steps+1)
x = x[1:,:]
xt = x[-1,:]

# True and perturbed trajectories
d0 = 0.05 # perturbation amplitude
xp = xt + d0 * np.random.randn(3)

# Propagate perturbed trajectory
x = LorenzSystems.generate_trajectory_fast('63', xp, dt, n_steps+1)
x = x[1:,:] # Remove initial condition
# Propagate perturbed trajectory with surrogate model
xss = surrogates[model_name].rollout(torch.tensor(xp), num_steps=n_steps)
# Last state of perturbed trajectory with surrogate model
xps = xss[-1,:] # Last state Surrogate perturbed
xp = x[-1,:] # Last state Lorenz perturbed

# Propagate true trajectory with both Lorenz and surrogate models
xs = LorenzSystems.generate_trajectory_fast('63', xt, dt, n_steps+1)
xs = xs[1:,:]
xts = surrogates[model_name].rollout(torch.tensor(xt), num_steps=n_steps)

# Last state of perturbed and true trajectories
xst = xts[-1,:] # Last state Surrogate unperturbed
xt = xs[-1,:] # Last state Lorenz unperturbed

# Plot true and perturbed trajectories
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for i, var_name in enumerate(VAR_NAMES):
    ax[i].plot(xs[:,i], label='True Lorenz')
    ax[i].plot(xts[:,i], label='True Surrogate')
    ax[i].plot(x[:,i], label='Perturbed Lorenz', linestyle='--', alpha=0.5)
    ax[i].plot(xss[:,i], label='Perturbed Surrogate', linestyle='--', alpha=0.5)
    ax[i].legend()
    ax[i].set_title(f'{var_name}')

plt.tight_layout()
plt.show()

# %% Initialize ensemble
Ne_total = 100 # ensemble size
d0 = 0.05 # perturbation amplitude
Nx = 3 # number of state variables

### Initial ensemble
ones = np.ones(Ne_total)
Xf = np.outer(ones,xp) + d0 * np.random.randn(Ne_total,Nx) # defined from perturbed run
# np.outer(ones,xb) repeats xp on each row
print('Xf shape', Xf.shape) # [Ne, Nx]

### Propagate ensemble
time = np.arange(0,10,0.01); # shorter than before
M = len(time)
snapshots = np.zeros((Ne_total,M,Nx)) # [Ne, Nt, Nx]
for e in range(0,Ne_total): # loop over ensemble and propagate each member
    sol = surrogates[model_name].rollout(Xf[e,:], num_steps=M) # [Nt, Nx]
    print(sol.shape)
    snapshots[e,:,:] = sol
    Xf[e,:] = sol[-1,:]; # sol[-1,:] = last time only, all state vars

### Propagate true state
xSt = LorenzSystems.generate_trajectory_fast('63', xt, dt, M+1)
xSt = xSt[1:,:]
xt = xSt[-1,:] # last time only, all state vars

# %% Correlation matrix and localization
### Define the Gaussian correlation matrix
def create_localization_matrix(r,n):
    L = np.zeros((n,n)); # n is the number of grid points (state dim)
    for i in range(0,n):
        for j in range(i,n):
            dij = np.min([np.abs(i-j),np.abs((n-1)-j+i)]); # accounts for periodicity
                                                           # draw example on board
            L[i,j] = (dij**2)/(2*r**2)
            L[j,i] = L[i,j]; # to ensure symmetry
    L = np.exp(-L)
    return L

### Visualize it for a certain cut-off radius
r = 14
L = create_localization_matrix(r,Nx)
plt.matshow(L)
plt.colorbar()

# %% Localization

### Select a subset of the entire forecast ensemble
Ne = 20
ind = np.random.permutation(np.arange(0,Ne_total))[:Ne]
Xf_sub = Xf[ind,:]; # notice how the indices were selected randomly

### Plot the raw forecast covariance matrix
Pf = np.cov(Xf_sub.T); # equivalent to Xf * Xf^T, Xf is [Ne, Nx]
plt.figure(figsize=(10,8))
plt.matshow(Pf)
plt.title('$P^f$')
plt.colorbar()

### Plot the localized forecast covariance matrix
for r in (2,5,10):
    L = create_localization_matrix(r,Nx)
    PfG_r = L*Pf; # Shur product
    plt.figure(figsize=(10,8))
    plt.matshow(PfG_r)
    plt.title(f'$C \odot P^f$ with cut-off radius r = {r}')
    plt.colorbar()

# %% Run a single EnKF assimilation cycle

np.random.seed(seed=10) # to ensure results are reproducible
time_step = 0.01 # dt for numerical integration
p = 1 # fraction of the directly observed state variables
T = 50 # number of time steps to simulate (not the absolute time)
errorf_k = np.zeros(T) # array to store the forecast errors
errora_k = np.zeros(T) # array to store the analysis errors
I = np.eye(Nx,Nx) # identity matrix
Ne = 20 # the ensemble size we will use in this experiment
ones = np.ones(Ne) # a vector of 1s, to be used later in creating
                    # the full ensemble matrices
loc = 3 # localization radius to be used in this experiment
infl = 1.02 # inflation factor to be used in this experiment
M = 50 # number of time steps run before assimilation window

### Initial true state and ensemble
xt_k = xt.copy() # this is the last true state from the
                  # previous time integration
ind = np.random.permutation(np.arange(0,Ne_total))[:Ne]
Xf_k = Xf[ind,:].copy().T

### Observation-related variables
Ny = int(round(p*Nx)) # number of observations
sig_obs = 1.1 # ob error std
R_k = (sig_obs**2)*np.eye(Ny,Ny) # ob error covariance matric

### Loop over time and assimilate observations (when present)
for k in range(0,T):

    print(f'Assimilation cycle: {k}')
    
    # Forecast mean
    xf_k = np.mean(Xf_k,1)
    
    # (Localized) forecast covariance
    L = create_localization_matrix(loc,Nx)
    Pf_k = L*np.cov(Xf_k) # localize ensemble via Schur product

    # RMSE (L2 or Euclidean norm) of forecast mean
    errorf_k[k] = np.linalg.norm(xf_k-xt_k)

    # Observation at time level k
    obs_comp = np.random.permutation(Nx)
    obs_comp = obs_comp[0:Ny]; # randomly select Ny state components
                               # (different each cycle)
    H_k = I[obs_comp,:] # ob operator
    err_obs_k = sig_obs*np.random.randn(Ny)
    y_k = H_k@xt_k + err_obs_k; # true observation value

    # Perturbed observations - one for each forecast member 
    Eobs_k = sig_obs*np.random.randn(Ny,Ne)
    Yobs_k = np.outer(y_k,np.ones(Ne)) + Eobs_k; # [Ny,Ne]
    
    # Scaled innovation matrix
    D_k = Yobs_k-H_k@Xf_k # [Ny,Ne]
    IN_k = R_k + H_k@Pf_k@H_k.T # [Ny,Ny]
    Z_k = np.linalg.solve(IN_k,D_k) # this gives (IN)^-1 * D =: Z [Ny,Ne]

    # EnKF update
    Xa_k = Xf_k + Pf_k@H_k.T@Z_k # [Nx,Ne]

    # Posterior multiplicative inflation
    xa_k = np.mean(Xa_k,1)
    DXa_k = Xa_k-np.outer(xa_k,ones) # deviations from anl mean
    Xa_k = np.outer(xa_k,ones)+infl*DXa_k

    # RMSE of analysis mean
    errora_k[k] = np.linalg.norm(xa_k-xt_k)

    # Forecast to next time (both ensemble and nature run)
    for e in range(0,Ne):
        Xa_e_S = surrogates[model_name].rollout(Xa_k[:,e], num_steps=M)
        Xf_k[:,e] = Xa_e_S[-1,:]
    xt_S = LorenzSystems.generate_trajectory_fast('63', xt_k, dt, M+1)
    xt_k = xt_S[-1,:]

# Plot forecast and analysis errors at each cycle
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(errorf_k, label='Forecast RMSE')
ax.plot(errora_k, label='Analysis RMSE')
ax.legend()
ax.set_title(f'Forecast and analysis RMSE at each cycle using {model_name} surrogate model')
ax.set_xlabel('Cycle')
ax.set_ylabel('RMSE')
plt.tight_layout()
plt.show()

# %%
