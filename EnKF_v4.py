# %%
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from os.path import join
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

# ============================================================
# Utility functions
# ============================================================
VAR_NAMES = ['x', 'y', 'z']

def create_localization_matrix(r, n):
    """Gaussian (Gaspari-Cohn-like) localization matrix with periodic distance."""
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            dij = min(abs(i - j), abs((n - 1) - j + i))
            L[i, j] = (dij ** 2) / (2 * r ** 2)
            L[j, i] = L[i, j]
    return np.exp(-L)

# ============================================================
# Spin-up: generate true initial condition on the attractor
# ============================================================
np.random.seed(10)
spinup_steps = 2000
x0 = np.random.randn(3).astype(np.float64)
dt = 0.01
d0 = 1.0 # perturbation amplitude
n_steps = len(np.arange(0, 30, dt))

traj = LorenzSystems.generate_trajectory_fast('63', x0, dt, spinup_steps + 1)
xt_init = traj[-1, :]  # on the attractor after spin-up

# Create a perturbed initial condition and propagate both
xp_init = xt_init + d0 * np.random.randn(3)

# Propagate both to build an initial ensemble pool
traj_true = LorenzSystems.generate_trajectory_fast('63', xt_init, dt, n_steps + 1)[1:, :]
traj_pert = LorenzSystems.generate_trajectory_fast('63', xp_init, dt, n_steps + 1)[1:, :]

xt = traj_true[-1, :]   # true state after propagation
xp = traj_pert[-1, :]   # perturbed state after propagation

# %% Build initial ensemble pool (propagated with each surrogate once)
Ne_total = 100
Nx = 3

# Ensemble centered on perturbed state
Xf_pool = np.outer(np.ones(Ne_total), xp) + d0 * np.random.randn(Ne_total, Nx)

# Propagate ensemble pool forward to decorrelate members
M_spinup = len(np.arange(0, 10, dt))

# We store per-surrogate propagated pools so each architecture
# starts from its own spun-up ensemble
Xf_pools = {}
for name, model in surrogates.items():
    pool = Xf_pool.copy()
    for e in range(Ne_total):
        sol = model.rollout(pool[e, :], num_steps=M_spinup)
        pool[e, :] = sol[-1, :]
    Xf_pools[name] = pool
    print(f"  Ensemble pool ready for {name}")

# Lorenz baseline pool (perfect model)
pool_lorenz = Xf_pool.copy()
for e in range(Ne_total):
    sol = LorenzSystems.generate_trajectory_fast('63', pool_lorenz[e, :], dt, M_spinup + 1)
    pool_lorenz[e, :] = sol[-1, :]
Xf_pools['Lorenz63'] = pool_lorenz
print(f"  Ensemble pool ready for Lorenz63 (baseline)")

# Propagate truth forward the same duration
traj_truth_spinup = LorenzSystems.generate_trajectory_fast('63', xt, dt, M_spinup + 1)
xt = traj_truth_spinup[-1, :]

# ============================================================
# EnKF configuration dataclass
# ============================================================
class EnKFConfig:
    """All tunable parameters for a single EnKF experiment."""
    def __init__(
        self,
        T=30,               # number of DA cycles
        M=10,               # forecast steps between assimilation cycles
        Ne=20,              # ensemble size (drawn from pool)
        dt=0.01,            # time step
        p=1.0,              # fraction of observed state variables
        sig_obs=1.1,        # observation error std
        use_localization=False,
        loc_radius=3.0,     # localization cut-off radius
        use_inflation=False,
        infl_factor=1.02,   # multiplicative inflation factor
        seed=10,            # random seed for reproducibility
    ):
        self.T = T
        self.M = M
        self.Ne = Ne
        self.dt = dt
        self.p = p
        self.sig_obs = sig_obs
        self.use_localization = use_localization
        self.loc_radius = loc_radius
        self.use_inflation = use_inflation
        self.infl_factor = infl_factor
        self.seed = seed

# ============================================================
# Core EnKF cycle
# ============================================================
def make_surrogate_forecaster(surrogate, M):
    """Wrap a SurrogateModel into a forecast callable: state -> state after M steps."""
    def forecaster(state):
        sol = surrogate.rollout(state, num_steps=M)
        return np.asarray(sol[-1, :])
    return forecaster

def make_lorenz_forecaster(dt, M):
    """Wrap the Lorenz solver into a forecast callable: state -> state after M steps."""
    def forecaster(state):
        traj = LorenzSystems.generate_trajectory_fast('63', np.asarray(state), dt, M + 1)
        return traj[-1, :]
    return forecaster

def run_enkf(forecast_fn, xt_0, Xf_pool, config: EnKFConfig):
    """
    Run a full EnKF assimilation experiment.

    Parameters
    ----------
    forecast_fn : callable
        Maps a state vector (Nx,) to the forecast state after M steps.
    xt_0 : np.ndarray, shape (Nx,)
        Initial true state (propagated with the real Lorenz solver).
    Xf_pool : np.ndarray, shape (Ne_total, Nx)
        Pre-propagated ensemble pool from which Ne members are drawn.
    config : EnKFConfig
        Experiment configuration.

    Returns
    -------
    dict with keys:
        'errorf'  : np.ndarray (T,)  - forecast RMSE per cycle
        'errora'  : np.ndarray (T,)  - analysis RMSE per cycle
        'spread'  : np.ndarray (T,)  - ensemble spread per cycle
        'xt_traj' : np.ndarray (T, Nx) - true state at each cycle
        'xf_traj' : np.ndarray (T, Nx) - forecast mean at each cycle
        'xa_traj' : np.ndarray (T, Nx) - analysis mean at each cycle
    """
    cfg = config
    np.random.seed(cfg.seed)

    Nx = xt_0.shape[0]
    I = np.eye(Nx)
    ones = np.ones(cfg.Ne)

    # Draw ensemble subset from pool
    ind = np.random.permutation(Xf_pool.shape[0])[:cfg.Ne]
    Xf_k = Xf_pool[ind, :].copy().T  # [Nx, Ne]
    xt_k = xt_0.copy()

    # Observation setup
    Ny = int(round(cfg.p * Nx))
    R_k = (cfg.sig_obs ** 2) * np.eye(Ny)

    # Pre-compute localization matrix (identity if disabled)
    if cfg.use_localization:
        L = create_localization_matrix(cfg.loc_radius, Nx)
    else:
        L = np.ones((Nx, Nx))

    # Storage
    errorf = np.zeros(cfg.T)
    errora = np.zeros(cfg.T)
    spread = np.zeros(cfg.T)
    xt_traj = np.zeros((cfg.T, Nx))
    xf_traj = np.zeros((cfg.T, Nx))
    xa_traj = np.zeros((cfg.T, Nx))

    for k in range(cfg.T):
        # --- Forecast statistics ---
        xf_k = np.mean(Xf_k, axis=1)
        Pf_k = L * np.cov(Xf_k)  # Schur product (localization or identity)

        # Ensemble spread: mean std across state variables
        spread[k] = np.mean(np.std(Xf_k, axis=1))

        # Forecast RMSE
        errorf[k] = np.linalg.norm(xf_k - xt_k)

        # Store trajectories
        xt_traj[k] = xt_k
        xf_traj[k] = xf_k

        # --- Observations ---
        obs_comp = np.random.permutation(Nx)[:Ny]
        H_k = I[obs_comp, :]
        y_k = H_k @ xt_k + cfg.sig_obs * np.random.randn(Ny)

        # Perturbed observations
        Yobs_k = np.outer(y_k, np.ones(cfg.Ne)) + cfg.sig_obs * np.random.randn(Ny, cfg.Ne)

        # --- EnKF update ---
        D_k = Yobs_k - H_k @ Xf_k
        IN_k = R_k + H_k @ Pf_k @ H_k.T
        Z_k = np.linalg.solve(IN_k, D_k)
        Xa_k = Xf_k + Pf_k @ H_k.T @ Z_k

        # --- Inflation ---
        xa_k = np.mean(Xa_k, axis=1)
        if cfg.use_inflation:
            DXa_k = Xa_k - np.outer(xa_k, ones)
            Xa_k = np.outer(xa_k, ones) + cfg.infl_factor * DXa_k

        # Analysis RMSE
        errora[k] = np.linalg.norm(xa_k - xt_k)
        xa_traj[k] = xa_k

        # --- Forecast to next cycle ---
        for e in range(cfg.Ne):
            Xf_k[:, e] = forecast_fn(Xa_k[:, e])

        # Truth forward (always the real Lorenz solver)
        xt_next = LorenzSystems.generate_trajectory_fast('63', xt_k, cfg.dt, cfg.M + 1)
        xt_k = xt_next[-1, :]

    return {
        'errorf': errorf,
        'errora': errora,
        'spread': spread,
        'xt_traj': xt_traj,
        'xf_traj': xf_traj,
        'xa_traj': xa_traj,
    }
# %%
# ============================================================
# Run EnKF for all surrogate architectures
# ============================================================

# --- Experiment configuration ---
cfg = EnKFConfig(
    T=50,
    M=100,
    Ne=50,
    dt=0.01,
    p=0.4,
    sig_obs=1.1,
    use_localization=False,   # toggle localization
    loc_radius=3.0,           # adjust radius
    use_inflation=False,      # toggle inflation
    infl_factor=1.02,         # adjust factor
    seed=10,
)

results = {}

# --- Lorenz63 baseline (perfect model) ---
print(f"\n{'='*50}")
print(f"Running EnKF with baseline: Lorenz63 (perfect model)")
print(f"{'='*50}")
lorenz_fn = make_lorenz_forecaster(cfg.dt, cfg.M)
res = run_enkf(
    forecast_fn=lorenz_fn,
    xt_0=xt.copy(),
    Xf_pool=Xf_pools['Lorenz63'],
    config=cfg,
)
results['Lorenz63'] = res
print(f"  Final forecast RMSE: {res['errorf'][-1]:.4f}")
print(f"  Final analysis RMSE: {res['errora'][-1]:.4f}")
print(f"  Mean ensemble spread: {np.mean(res['spread']):.4f}")

# --- Surrogate models ---
for name, model in surrogates.items():
    print(f"\n{'='*50}")
    print(f"Running EnKF with surrogate: {name}")
    print(f"  Localization: {'ON (r={})'.format(cfg.loc_radius) if cfg.use_localization else 'OFF'}")
    print(f"  Inflation:    {'ON (λ={})'.format(cfg.infl_factor) if cfg.use_inflation else 'OFF'}")
    print(f"{'='*50}")

    forecast_fn = make_surrogate_forecaster(model, cfg.M)
    res = run_enkf(
        forecast_fn=forecast_fn,
        xt_0=xt.copy(),
        Xf_pool=Xf_pools[name],
        config=cfg,
    )
    results[name] = res
    print(f"  Final forecast RMSE: {res['errorf'][-1]:.4f}")
    print(f"  Final analysis RMSE: {res['errora'][-1]:.4f}")
    print(f"  Mean ensemble spread: {np.mean(res['spread']):.4f}")

# ============================================================
# Plotting: compare all architectures
# ============================================================

# 1. Forecast & Analysis RMSE comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
cycles = np.arange(cfg.T)

for name, res in results.items():
    color = surrogates_palette[name]
    ls = '--' if name == 'Lorenz63' else '-'
    lw = 2.0 if name == 'Lorenz63' else 1.5
    axes[0].plot(cycles, res['errorf'], color=color, label=name, linewidth=lw, linestyle=ls)
    axes[1].plot(cycles, res['errora'], color=color, label=name, linewidth=lw, linestyle=ls)

axes[0].set_title('Forecast RMSE per cycle')
axes[0].set_xlabel('DA Cycle')
axes[0].set_ylabel('RMSE')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title('Analysis RMSE per cycle')
axes[1].set_xlabel('DA Cycle')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

loc_str = f"Loc r={cfg.loc_radius}" if cfg.use_localization else "No Loc"
infl_str = f"Infl λ={cfg.infl_factor}" if cfg.use_inflation else "No Infl"
obs_str = f"Obs p={cfg.p}" if cfg.p != 1.0 else "Obs All"
fig.suptitle(f'EnKF Architecture Comparison  |  {loc_str}  |  {infl_str}  |  Ne={cfg.Ne} | {obs_str}, Obs_err={cfg.sig_obs}, Steps={cfg.M}', fontsize=13)
plt.tight_layout()
plt.savefig(f'outputs/EnKF_Architecture_Comparison_p{cfg.p}_sig_obs{cfg.sig_obs}_M{cfg.M}_Ne{cfg.Ne}.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Ensemble spread comparison
fig, ax = plt.subplots(figsize=(10, 5))
for name, res in results.items():
    color = surrogates_palette[name]
    ls = '--' if name == 'Lorenz63' else '-'
    lw = 2.0 if name == 'Lorenz63' else 1.5
    ax.plot(cycles, res['spread'], color=color, label=name, linewidth=lw, linestyle=ls)

ax.set_title('Ensemble Spread per Cycle')
ax.set_xlabel('DA Cycle')
ax.set_ylabel('Mean Std across State Variables')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'outputs/EnKF_Ensemble_Spread_p{cfg.p}_sig_obs{cfg.sig_obs}_M{cfg.M}_Ne{cfg.Ne}.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Summary bar chart: mean RMSE over all cycles
model_names = list(results.keys())
mean_f = [np.mean(results[n]['errorf']) for n in model_names]
mean_a = [np.mean(results[n]['errora']) for n in model_names]
colors = [surrogates_palette[n] for n in model_names]

x_pos = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
bars_f = ax.bar(x_pos - width/2, mean_f, width, label='Forecast', color=colors, alpha=0.7, edgecolor='black')
bars_a = ax.bar(x_pos + width/2, mean_a, width, label='Analysis', color=colors, alpha=1.0, edgecolor='black')

ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, rotation=15)
ax.set_ylabel('Mean RMSE')
ax.set_title('Mean RMSE by Architecture')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'outputs/EnKF_Mean_RMSE_p{cfg.p}_sig_obs{cfg.sig_obs}_M{cfg.M}_Ne{cfg.Ne}.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
