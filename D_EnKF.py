# %%
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no display needed

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from scipy.linalg import sqrtm
import os

from os.path import join
from SurrogateModel import SurrogateModel
from lorenz.lorenz_systems import LorenzSystems

torch.set_default_dtype(torch.float64)

os.makedirs("figures", exist_ok=True)

# Global figure counter for unique filenames
_fig_count = [0]

def _savefig(label="fig"):
    """Save current figure to figures/ directory with a sequential counter prefix."""
    fname = f"figures/{_fig_count[0]:03d}_{label}.png"
    plt.savefig(fname, bbox_inches='tight', dpi=150)
    plt.close()
    _fig_count[0] += 1
    print(f"  [saved: {fname}]")

def _print_benchmark_table(results, title=""):
    """Print a summary DataFrame for a set of results."""
    rows = []
    for name, res in results.items():
        n_valid = int(np.sum(~np.isnan(res['errorf'])))
        mf = np.nanmean(res['errorf'])
        ma = np.nanmean(res['errora'])
        ms = np.nanmean(res['spread'])
        ratio = ms / mf if mf > 0 else np.nan
        rows.append({
            'Model': name,
            'fRMSE': round(mf, 4),
            'aRMSE': round(ma, 4),
            'Spread': round(ms, 4),
            'Spread/fRMSE': round(ratio, 4),
            'Valid_cycles': n_valid,
            'Diverged': res['diverged'],
        })
    df = pd.DataFrame(rows)
    if title:
        print(f"\n{title}")
    print(df.to_string(index=False))
    return df

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
n_steps = len(np.arange(0, 10, dt))

traj = LorenzSystems.generate_trajectory_fast('63', x0, dt, spinup_steps + 1)
xt_init = traj[-1, :]  # on the attractor after spin-up

# Create a perturbed initial condition and propagate both
d0 = 1.0
xp_init = xt_init + d0 * np.random.randn(3)

# Propagate both to build an initial ensemble pool
traj_true = LorenzSystems.generate_trajectory_fast('63', xt_init, dt, n_steps + 1)[1:, :]
traj_pert = LorenzSystems.generate_trajectory_fast('63', xp_init, dt, n_steps + 1)[1:, :]

xt = traj_true[-1, :]   # true state after propagation
xp = traj_pert[-1, :]   # perturbed state after propagation

# %% Build initial ensemble pool (propagated with each surrogate once)
Ne_total = 50
Nx = 3

# Ensemble centered on perturbed state
Xf_pool = np.outer(np.ones(Ne_total), xp) + d0 * np.random.randn(Ne_total, Nx)

# Propagate ensemble pool forward to decorrelate members
M_spinup = len(np.arange(0, 10, dt))
#M_spinup = 2

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
        enkf_type='stochastic',  # 'stochastic' or 'deterministic'
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
        self.enkf_type = enkf_type
# ============================================================
# Core EnKF cycle
# ============================================================
def make_surrogate_forecaster(surrogate, M):
    """Wrap a SurrogateModel into a forecast callable.
    Returns full trajectory (M+1, Nx) including the initial state."""
    def forecaster(state):
        sol = surrogate.rollout(state, num_steps=M)  # (M, Nx)
        sol = np.asarray(sol)
        # Prepend initial state for a continuous trajectory
        init = np.asarray(state).reshape(1, -1)
        return np.vstack([init, sol])  # (M+1, Nx)
    return forecaster

def make_lorenz_forecaster(dt, M):
    """Wrap the Lorenz solver into a forecast callable.
    Returns full trajectory (M+1, Nx) including the initial state."""
    def forecaster(state):
        traj = LorenzSystems.generate_trajectory_fast('63', np.asarray(state), dt, M + 1)
        return traj  # (M+1, Nx) — already includes initial state
    return forecaster

def run_enkf(forecast_fn, xt_0, Xf_pool, config: EnKFConfig):
    """
    Run a full EnKF assimilation experiment.

    Parameters
    ----------
    surrogate : SurrogateModel
        ML surrogate used as the forecast model.
    xt_0 : np.ndarray, shape (Nx,)
        Initial true state (propagated with the real Lorenz solver).
    Xf_pool : np.ndarray, shape (Ne_total, Nx)
        Pre-propagated ensemble pool from which Ne members are drawn.
    config : EnKFConfig
        Experiment configuration.

    Returns
    -------
    dict with keys:
        'errorf'  : np.ndarray (T,)  – forecast RMSE per cycle
        'errora'  : np.ndarray (T,)  – analysis RMSE per cycle
        'spread'  : np.ndarray (T,)  – ensemble spread per cycle
        'xt_traj' : np.ndarray (T, Nx) – true state at each cycle
        'xf_traj' : np.ndarray (T, Nx) – forecast mean at each cycle
        'xa_traj' : np.ndarray (T, Nx) – analysis mean at each cycle
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
    Xf_all = np.zeros((cfg.T, Nx, cfg.Ne))
    Xa_all = np.zeros((cfg.T, Nx, cfg.Ne))

    # Storage — full inter-cycle trajectories
    # ens_fcst_traj[k, e, :, :] = trajectory of member e from analysis at cycle k
    #                              to forecast at cycle k+1, shape (M+1, Nx)
    # truth_fcst_traj[k, :, :] = truth trajectory over the same window
    ens_fcst_traj = np.full((cfg.T, cfg.Ne, cfg.M + 1, Nx), np.nan)
    truth_fcst_traj = np.full((cfg.T, cfg.M + 1, Nx), np.nan)

    diverged = False

    for k in range(cfg.T):
        # --- Divergence check ---
        if np.any(np.isnan(Xf_k)) or np.any(np.abs(Xf_k) > 1e6):
            print(f"  WARNING: Ensemble diverged at cycle {k}")
            errorf[k:] = np.nan
            errora[k:] = np.nan
            spread[k:] = np.nan
            diverged = True
            break

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
        Xf_all[k] = Xf_k.copy()

        # --- Observations ---
        obs_comp = np.random.permutation(Nx)[:Ny]
        H_k = I[obs_comp, :]
        y_k = H_k @ xt_k + cfg.sig_obs * np.random.randn(Ny)

        if cfg.enkf_type == 'stochastic':
            # ========== Stochastic EnKF (perturbed observations) ==========
            Yobs_k = np.outer(y_k, np.ones(cfg.Ne)) + cfg.sig_obs * np.random.randn(Ny, cfg.Ne)

            D_k = Yobs_k - H_k @ Xf_k
            IN_k = R_k + H_k @ Pf_k @ H_k.T
            Z_k = np.linalg.solve(IN_k, D_k)
            Xa_k = Xf_k + Pf_k @ H_k.T @ Z_k

        elif cfg.enkf_type == 'deterministic':
            # ========== Deterministic EnKF (ETKF) =========================
            # Ensemble anomalies (raw, not normalized)
            Xf_prime = Xf_k - np.outer(xf_k, ones)          # [Nx, Ne]

            # Project anomalies into observation space
            S_k = H_k @ Xf_prime                              # [Ny, Ne]

            # Ensemble-space analysis covariance
            R_inv = np.linalg.inv(R_k)
            C_k = (cfg.Ne - 1) * np.eye(cfg.Ne) + S_k.T @ R_inv @ S_k  # [Ne, Ne]
            C_inv = np.linalg.inv(C_k)

            # Mean update weights
            d_k = y_k - H_k @ xf_k                            # innovation [Ny,]
            w_bar = C_inv @ S_k.T @ R_inv @ d_k               # [Ne,]

            # Analysis mean
            xa_k_det = xf_k + Xf_prime @ w_bar

            # Perturbation update via symmetric matrix square root
            W_k = np.real(sqrtm((cfg.Ne - 1) * C_inv))        # [Ne, Ne]

            # Rebuild full analysis ensemble
            Xa_k = np.outer(xa_k_det, ones) + Xf_prime @ W_k

        else:
            raise ValueError(f"Unknown enkf_type: '{cfg.enkf_type}'. Use 'stochastic' or 'deterministic'.")

        # --- Inflation ---
        xa_k = np.mean(Xa_k, axis=1)
        if cfg.use_inflation:
            DXa_k = Xa_k - np.outer(xa_k, ones)
            Xa_k = np.outer(xa_k, ones) + cfg.infl_factor * DXa_k

        # Analysis RMSE
        errora[k] = np.linalg.norm(xa_k - xt_k)
        xa_traj[k] = xa_k
        Xa_all[k] = Xa_k.copy()

        # --- Forecast to next cycle (store full trajectories) ---
        for e in range(cfg.Ne):
            traj_e = forecast_fn(Xa_k[:, e])       # (M+1, Nx)
            ens_fcst_traj[k, e, :, :] = traj_e
            Xf_k[:, e] = traj_e[-1, :]             # final state is the next forecast

        # Truth forward (always the real Lorenz solver)
        xt_next = LorenzSystems.generate_trajectory_fast('63', xt_k, cfg.dt, cfg.M + 1)
        truth_fcst_traj[k, :, :] = xt_next          # (M+1, Nx)
        xt_k = xt_next[-1, :]

    return {
        'errorf': errorf,
        'errora': errora,
        'spread': spread,
        'xt_traj': xt_traj,
        'xf_traj': xf_traj,
        'xa_traj': xa_traj,
        'Xf_all': Xf_all,               # [T, Nx, Ne]
        'Xa_all': Xa_all,                # [T, Nx, Ne]
        'ens_fcst_traj': ens_fcst_traj,  # [T, Ne, M+1, Nx]
        'truth_fcst_traj': truth_fcst_traj,  # [T, M+1, Nx]
        'diverged': diverged,
    }
# ============================================================
# Benchmark runner
# ============================================================
def run_all_models(cfg, surrogates, Xf_pools, xt_0, palette):
    """Run EnKF for Lorenz baseline + all surrogates under a given config."""
    results = {}

    # Lorenz63 baseline
    lorenz_fn = make_lorenz_forecaster(cfg.dt, cfg.M)
    res = run_enkf(lorenz_fn, xt_0.copy(), Xf_pools['Lorenz63'], cfg)
    results['Lorenz63'] = res
    print(f"  Lorenz63    | fRMSE={np.nanmean(res['errorf']):.3f}  aRMSE={np.nanmean(res['errora']):.3f}  spread={np.nanmean(res['spread']):.3f}  div={res['diverged']}")

    # Surrogates
    for name, model in surrogates.items():
        fn = make_surrogate_forecaster(model, cfg.M)
        res = run_enkf(fn, xt_0.copy(), Xf_pools[name], cfg)
        results[name] = res
        print(f"  {name:12s} | fRMSE={np.nanmean(res['errorf']):.3f}  aRMSE={np.nanmean(res['errora']):.3f}  spread={np.nanmean(res['spread']):.3f}  div={res['diverged']}")

    return results
# %%
# ============================================================
# Plotting functions
# ============================================================
def plot_rmse_comparison(results, palette, cfg, title_suffix=""):
    """Forecast & Analysis RMSE side by side for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    cycles = np.arange(cfg.T)
    for name, res in results.items():
        color = palette.get(name, '#888888')
        ls = '--' if name == 'Lorenz63' else '-'
        lw = 2.0 if name == 'Lorenz63' else 1.5
        axes[0].plot(cycles, res['errorf'], color=color, label=name, lw=lw, ls=ls)
        axes[1].plot(cycles, res['errora'], color=color, label=name, lw=lw, ls=ls)
    axes[0].set_title('Forecast RMSE')
    axes[0].set_xlabel('DA Cycle'); axes[0].set_ylabel('RMSE')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_title('Analysis RMSE')
    axes[1].set_xlabel('DA Cycle')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    loc_str = f"Loc r={cfg.loc_radius}" if cfg.use_localization else "No Loc"
    infl_str = f"Infl λ={cfg.infl_factor}" if cfg.use_inflation else "No Infl"
    fig.suptitle(f'EnKF Comparison  |  {loc_str}  |  {infl_str}  |  Ne={cfg.Ne}  |  M={cfg.M}  |  p={cfg.p}  |  σ_obs={cfg.sig_obs}{title_suffix}', fontsize=11)
    plt.tight_layout()
    _savefig("rmse_comparison")

def plot_spread_comparison(results, palette, cfg, title_suffix=""):
    """Ensemble spread time series."""
    fig, ax = plt.subplots(figsize=(10, 5))
    cycles = np.arange(cfg.T)
    for name, res in results.items():
        color = palette.get(name, '#888888')
        ls = '--' if name == 'Lorenz63' else '-'
        lw = 2.0 if name == 'Lorenz63' else 1.5
        ax.plot(cycles, res['spread'], color=color, label=name, lw=lw, ls=ls)
    ax.set_title(f'Ensemble Spread per Cycle{title_suffix}')
    ax.set_xlabel('DA Cycle'); ax.set_ylabel('Mean Std')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig("spread_comparison")

def plot_spread_vs_rmse(results, palette, cfg, title_suffix=""):
    """Scatter: mean spread vs mean forecast RMSE — reveals calibration."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, res in results.items():
        ms = np.nanmean(res['spread'])
        mf = np.nanmean(res['errorf'])
        color = palette.get(name, '#888888')
        marker = 's' if name == 'Lorenz63' else 'o'
        ax.scatter(ms, mf, color=color, s=120, marker=marker, edgecolors='k', zorder=5)
        ax.annotate(name, (ms, mf), textcoords="offset points", xytext=(8, 5), fontsize=9)
    lims = ax.get_xlim()
    ax.plot(lims, lims, 'k--', alpha=0.3, label='Spread = RMSE (ideal)')
    ax.set_xlabel('Mean Ensemble Spread')
    ax.set_ylabel('Mean Forecast RMSE')
    ax.set_title(f'Spread vs RMSE {title_suffix}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _savefig("spread_vs_rmse")

def compute_rank_histogram(Xf_all, xt_traj):
    """
    Compute rank histograms (Talagrand diagrams) from stored ensembles.

    Parameters
    ----------
    Xf_all : np.ndarray, shape (T, Nx, Ne)
        Forecast ensemble at each DA cycle.
    xt_traj : np.ndarray, shape (T, Nx)
        True state at each DA cycle.

    Returns
    -------
    ranks : np.ndarray, shape (T * Nx,)
        Rank of the truth within the ensemble (0 to Ne inclusive).
    """
    T, Nx, Ne = Xf_all.shape
    ranks = np.zeros(T * Nx, dtype=int)
    idx = 0
    for k in range(T):
        if np.any(np.isnan(Xf_all[k])):
            ranks[idx:idx + Nx] = -1  # mark invalid
            idx += Nx
            continue
        for v in range(Nx):
            sorted_ens = np.sort(Xf_all[k, v, :])
            rank = np.searchsorted(sorted_ens, xt_traj[k, v])
            ranks[idx] = rank
            idx += 1
    # Remove invalid entries
    ranks = ranks[ranks >= 0]
    return ranks

def compute_rank_histogram_per_var(Xf_all, xt_traj):
    """Compute rank histograms separately for each state variable."""
    T, Nx, Ne = Xf_all.shape
    ranks_per_var = {v: [] for v in range(Nx)}
    for k in range(T):
        if np.any(np.isnan(Xf_all[k])):
            continue
        for v in range(Nx):
            sorted_ens = np.sort(Xf_all[k, v, :])
            rank = np.searchsorted(sorted_ens, xt_traj[k, v])
            ranks_per_var[v].append(rank)
    return {v: np.array(r) for v, r in ranks_per_var.items()}

def plot_talagrand_grid(results, palette, cfg, ensemble_key='Xf_all', title_suffix=""):
    """
    Talagrand rank histograms: one row per model, one column per state variable,
    plus one column for the aggregated histogram.
    """
    model_names = list(results.keys())
    n_models = len(model_names)
    Nx = results[model_names[0]]['xt_traj'].shape[1]
    Ne = cfg.Ne

    fig, axes = plt.subplots(n_models, Nx + 1, figsize=(4 * (Nx + 1), 3 * n_models),
                              squeeze=False)
    bins = np.arange(Ne + 2) - 0.5  # bin edges for ranks 0..Ne

    for i, name in enumerate(model_names):
        res = results[name]
        color = palette.get(name, '#888888')

        # Per-variable ranks
        ranks_pv = compute_rank_histogram_per_var(res[ensemble_key], res['xt_traj'])
        for v in range(Nx):
            ax = axes[i, v]
            if len(ranks_pv[v]) > 0:
                ax.hist(ranks_pv[v], bins=bins, color=color, edgecolor='black', alpha=0.8, density=True)
                ax.axhline(1.0 / (Ne + 1), color='k', ls='--', alpha=0.5, label='Uniform')
            ax.set_xlim(-0.5, Ne + 0.5)
            if i == 0:
                ax.set_title(f'{VAR_NAMES[v]}', fontsize=11)
            if v == 0:
                ax.set_ylabel(name, fontsize=10, fontweight='bold')

        # Aggregated ranks
        ax_agg = axes[i, Nx]
        ranks_all = compute_rank_histogram(res[ensemble_key], res['xt_traj'])
        if len(ranks_all) > 0:
            ax_agg.hist(ranks_all, bins=bins, color=color, edgecolor='black', alpha=0.8, density=True)
            ax_agg.axhline(1.0 / (Ne + 1), color='k', ls='--', alpha=0.5)
        ax_agg.set_xlim(-0.5, Ne + 0.5)
        if i == 0:
            ax_agg.set_title('All variables', fontsize=11)

    ens_label = "Forecast" if ensemble_key == 'Xf_all' else "Analysis"
    fig.suptitle(f'Talagrand Diagrams ({ens_label} Ensemble)  |  Ne={Ne}  |  M={cfg.M}  |  p={cfg.p}{title_suffix}', fontsize=13)
    plt.tight_layout()
    _savefig("talagrand")

def plot_ensemble_spaghetti(res, name, palette, cfg, cycle_range=None, var_indices=None):
    """
    Spaghetti plot: full trajectory of every ensemble member between DA cycles,
    plus ensemble mean and truth. Analysis corrections are visible as jumps.

    Parameters
    ----------
    res : dict — single model result from run_enkf
    name : str — model name (for title and color)
    palette : dict
    cfg : EnKFConfig
    cycle_range : tuple (start, end) — which cycles to plot (default: all)
    var_indices : list of int — which state variables to plot (default: all)
    """
    ens_traj = res['ens_fcst_traj']    # (T, Ne, M+1, Nx)
    truth_traj = res['truth_fcst_traj']  # (T, M+1, Nx)
    T, Ne, Mp1, Nx = ens_traj.shape
    M = Mp1 - 1

    if cycle_range is None:
        cycle_range = (0, T)
    k_start, k_end = cycle_range
    k_end = min(k_end, T)

    if var_indices is None:
        var_indices = list(range(Nx))

    n_vars = len(var_indices)
    fig, axes = plt.subplots(n_vars, 1, figsize=(16, 4 * n_vars), sharex=True,
                              squeeze=False, dpi=600)
    color = palette.get(name, '#888888')

    for row, v in enumerate(var_indices):
        ax = axes[row, 0]

        for k in range(k_start, k_end):
            if np.any(np.isnan(ens_traj[k])):
                break
            # Time axis: each cycle spans M+1 points starting at k*M
            t_offset = k * M
            t_local = np.arange(Mp1) + t_offset

            # Individual members — thin, transparent
            for e in range(Ne):
                ax.plot(t_local, ens_traj[k, e, :, v],
                        color=color, alpha=0.6, linewidth=0.5)

            # Ensemble mean trajectory
            ens_mean = np.mean(ens_traj[k, :, :, v], axis=0)
            ax.plot(t_local, ens_mean, color='red', linewidth=2.0, alpha=0.8)

            # Truth
            ax.plot(t_local, truth_traj[k, :, v],
                    color='black', linewidth=1.2, alpha=0.9)

            # Mark analysis time (start of each forecast window)
            ax.axvline(t_offset, color='gray', linewidth=0.4, alpha=0.4, linestyle=':')

        ax.set_ylabel(f'{VAR_NAMES[v]}', fontsize=12)
        ax.grid(True, alpha=0.2)

    # Legend (only on first axis)
    legend_elements = [
        Line2D([0], [0], color='black', linewidth=1.2, label='Truth'),
        Line2D([0], [0], color='red', linewidth=1.5, label=f'{name} (ens. mean)'),
        Line2D([0], [0], color=color, linewidth=0.5, alpha=0.4, label=f'{name} members'),
    ]
    axes[0, 0].legend(handles=legend_elements, loc='upper right')

    axes[-1, 0].set_xlabel('Forecast step (cumulative)')
    fig.suptitle(f'Ensemble — {name}  |  Ne={cfg.Ne}  |  M={cfg.M}  |  cycles {k_start}–{k_end-1}',
                 fontsize=13)
    plt.tight_layout()
    _savefig(f"spaghetti_{name}")

def plot_ensemble_spaghetti_multi(results, palette, cfg, cycle_range=None, var_idx=0):
    """
    Compare spaghetti plots across all architectures for a single state variable.
    One subplot row per model.
    """
    model_names = list(results.keys())
    n_models = len(model_names)

    T = cfg.T
    M = cfg.M
    if cycle_range is None:
        cycle_range = (0, T)
    k_start, k_end = cycle_range
    k_end = min(k_end, T)

    fig, axes = plt.subplots(n_models, 1, figsize=(16, 3 * n_models), sharex=True,
                              squeeze=False, dpi=600)

    for i, name in enumerate(model_names):
        ax = axes[i, 0]
        res = results[name]
        ens_traj = res['ens_fcst_traj']
        truth_traj = res['truth_fcst_traj']
        Ne = ens_traj.shape[1]
        Mp1 = ens_traj.shape[2]
        color = palette.get(name, '#888888')

        for k in range(k_start, k_end):
            if np.any(np.isnan(ens_traj[k])):
                break
            t_offset = k * M
            t_local = np.arange(Mp1) + t_offset

            for e in range(Ne):
                ax.plot(t_local, ens_traj[k, e, :, var_idx],
                        color=color, alpha=0.6, linewidth=0.5)
            ens_mean = np.mean(ens_traj[k, :, :, var_idx], axis=0)
            ax.plot(t_local, ens_mean, color="red", linewidth=2.0, alpha=0.8)
            ax.plot(t_local, truth_traj[k, :, var_idx],
                    color='black', linewidth=1.0, alpha=0.9)
            ax.axvline(t_offset, color='gray', linewidth=0.3, alpha=0.3, linestyle=':')

        ax.set_ylabel(name, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.2)

    axes[-1, 0].set_xlabel('Forecast step (cumulative)')
    fig.suptitle(f'Ensemble Comparison — {VAR_NAMES[var_idx]}  |  Ne={cfg.Ne}  |  M={cfg.M}',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    _savefig(f"spaghetti_multi_{VAR_NAMES[var_idx]}")

def plot_spread_reduction(res, name, palette, cfg, cycle_range=None, var_idx=0):
    """
    For each cycle, show how the ensemble std evolves within the forecast window.
    This reveals whether spread grows (healthy divergence) or collapses (variance death).
    Also marks the analysis correction as a vertical jump.
    """
    ens_traj = res['ens_fcst_traj']    # (T, Ne, M+1, Nx)
    T, Ne, Mp1, Nx = ens_traj.shape
    M = Mp1 - 1

    if cycle_range is None:
        cycle_range = (0, T)
    k_start, k_end = cycle_range
    k_end = min(k_end, T)

    color = palette.get(name, '#888888')

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False, dpi=600)

    # Compute per-cycle spreads for the full run (bars should not depend on the selected window)
    analysis_spread_all = []
    forecast_spread_all = []
    valid_T = 0
    for k in range(T):
        if np.any(np.isnan(ens_traj[k])):
            break
        std_over_time = np.std(ens_traj[k, :, :, var_idx], axis=0)  # (M+1,)
        analysis_spread_all.append(std_over_time[0])
        forecast_spread_all.append(std_over_time[-1])
        valid_T += 1

    # Top: spread over time within each selected forecast window (x-axis is cumulative model time steps)
    last_t = None
    for k in range(k_start, min(k_end, valid_T)):
        t_offset = k * M
        t_local = np.arange(Mp1) + t_offset
        std_over_time = np.std(ens_traj[k, :, :, var_idx], axis=0)  # (M+1,)
        axes[0].plot(t_local, std_over_time, color=color, linewidth=1.0, alpha=0.6)
        axes[0].axvline(t_offset, color='gray', linewidth=0.3, alpha=0.3, linestyle=':')
        last_t = t_local[-1]

    axes[0].set_ylabel(f'Ensemble Std ({VAR_NAMES[var_idx]})')
    axes[0].set_title(f'Intra-cycle spread evolution — {name}')
    axes[0].grid(True, alpha=0.2)
    axes[0].set_xlabel('Model time step (cumulative)')
    if last_t is not None:
        axes[0].set_xlim(k_start * M, last_t)

    # Bottom: bar chart of analysis vs forecast spread per cycle
    cycles_plot = np.arange(valid_T)
    width = 0.35
    axes[1].bar(cycles_plot - width/2, analysis_spread_all, width,
                label='After analysis', color=color, alpha=0.5, edgecolor='black')
    axes[1].bar(cycles_plot + width/2, forecast_spread_all, width,
                label='Before next analysis', color=color, alpha=1.0, edgecolor='black')
    axes[1].set_xlabel('DA Cycle')
    axes[1].set_ylabel(f'Ensemble Std ({VAR_NAMES[var_idx]})')
    axes[1].set_title('Spread: post-analysis vs end-of-forecast')
    axes[1].legend()
    axes[1].grid(True, alpha=0.2, axis='y')

    # Visually indicate which cycles are shown in the top panel
    if valid_T > 0:
        sel_start = max(0, min(k_start, valid_T - 1))
        sel_end = max(0, min(k_end, valid_T))
        if sel_end > sel_start:
            axes[1].axvspan(sel_start - 0.5, sel_end - 0.5, color='gray', alpha=0.12, zorder=0)

    fig.suptitle(f'Spread Reduction Diagnostic — {name}  |  Ne={cfg.Ne}  |  M={cfg.M}', fontsize=13)
    plt.tight_layout()
    _savefig(f"spread_reduction_{name}")


def plot_spread_reduction_multi(results, palette, cfg, cycle_range=None, var_idx=0):
    """
    Multi-model version of `plot_spread_reduction`.

    - Top: for each model, overlay the intra-cycle spread evolution for the selected cycle window
           (x-axis in cumulative model time steps).
    - Bottom: for each model, show full-run per-cycle spread after analysis vs end-of-forecast
              (x-axis is DA cycle index, not timesteps).
    """
    if not results:
        return

    # Infer global T, M from cfg/first entry, but compute per-model valid_T independently.
    any_res = next(iter(results.values()))
    ens0 = any_res['ens_fcst_traj']
    T0, _, Mp1, _ = ens0.shape
    M = Mp1 - 1

    if cycle_range is None:
        cycle_range = (0, T0)
    k_start, k_end = cycle_range
    k_end = min(k_end, T0)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False, dpi=600)

    max_valid_T = 0
    for name, res in results.items():
        ens_traj = res['ens_fcst_traj']  # (T, Ne, M+1, Nx)
        T, _, Mp1_here, _ = ens_traj.shape
        M_here = Mp1_here - 1
        if M_here != M:
            # If different M per model, the cumulative-time overlay becomes ambiguous.
            # In that case, fall back to each model's own time axis (still decoupled from cycles below).
            M_used = M_here
        else:
            M_used = M

        color = palette.get(name, '#888888')

        # Determine valid cycles until NaNs (divergence)
        valid_T = 0
        analysis_spread_all = []
        forecast_spread_all = []
        for k in range(T):
            if np.any(np.isnan(ens_traj[k])):
                break
            std_over_time = np.std(ens_traj[k, :, :, var_idx], axis=0)  # (M+1,)
            analysis_spread_all.append(std_over_time[0])
            forecast_spread_all.append(std_over_time[-1])
            valid_T += 1
        max_valid_T = max(max_valid_T, valid_T)

        # Top: selected window only
        for k in range(k_start, min(k_end, valid_T)):
            t_offset = k * M_used
            t_local = np.arange(Mp1_here) + t_offset
            std_over_time = np.std(ens_traj[k, :, :, var_idx], axis=0)  # (M+1,)
            axes[0].plot(t_local, std_over_time, color=color, linewidth=0.8, alpha=0.60)

        # Bottom: full-run per-cycle series (use lines to avoid unreadable grouped bars)
        cycles = np.arange(valid_T)
        axes[1].plot(cycles, analysis_spread_all, color=color, alpha=0.6, lw=1.0, ls='--',
                     label=f'{name} (post)')
        axes[1].plot(cycles, forecast_spread_all, color=color, alpha=0.6, lw=1.0, ls='-',
                     label=f'{name} (end)')

    # Top styling
    axes[0].set_ylabel(f'Ensemble Std ({VAR_NAMES[var_idx]})')
    axes[0].set_title('Intra-cycle spread evolution (selected window)')
    axes[0].grid(True, alpha=0.2)
    axes[0].set_xlabel('Model time step (cumulative)')
    axes[0].axvline(k_start * M, color='gray', linewidth=0.6, alpha=0.5, linestyle=':')
    axes[0].axvline(k_end * M, color='gray', linewidth=0.6, alpha=0.5, linestyle=':')
    axes[0].set_xlim(k_start * M, k_end * M)

    # Bottom styling
    axes[1].set_xlabel('DA Cycle')
    axes[1].set_ylabel(f'Ensemble Std ({VAR_NAMES[var_idx]})')
    axes[1].set_title('Spread per cycle (full run): post-analysis vs end-of-forecast')
    axes[1].grid(True, alpha=0.2, axis='y')
    if max_valid_T > 0:
        sel_start = max(0, min(k_start, max_valid_T - 1))
        sel_end = max(0, min(k_end, max_valid_T))
        if sel_end > sel_start:
            axes[1].axvspan(sel_start - 0.5, sel_end - 0.5, color='gray', alpha=0.10, zorder=0)
        axes[1].set_xlim(-0.5, max_valid_T - 0.5)

    # Legend (keep it compact)
    axes[1].legend(loc='upper right', fontsize=7, ncol=2, frameon=True)

    fig.suptitle(f'Spread Reduction Diagnostic (multi)  |  Ne={cfg.Ne}  |  M={cfg.M}', fontsize=13)
    plt.tight_layout()
    _savefig("spread_reduction_multi")


def plot_trajectory_comparison(results, palette, cfg, spinup=0, title_suffix=""):
    """
    Compare ensemble forecast mean trajectories against truth for each
    state variable (x, y, z), one subplot per variable.

    Parameters
    ----------
    results      : dict from run_all_models
    palette      : colour dict
    cfg          : EnKFConfig
    spinup       : cycles to skip at the start of the plot (not trimmed
                   from data, just shifts x-axis start for clarity)
    title_suffix : str appended to figure title
    """
    cycles = np.arange(cfg.T)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    for var_idx, (ax, var_name) in enumerate(zip(axes, VAR_NAMES)):

        # --- truth (same for all models) ---
        xt = results['Lorenz63']['xt_traj'][spinup:, var_idx]
        ax.plot(cycles[spinup:], xt,
                color='black', lw=1.5, ls='--',
                label='truth', zorder=10)

        # --- each model's forecast mean ---
        for name, res in results.items():
            if res['diverged']:
                # plot only up to divergence point, then stop
                valid = ~np.isnan(res['xf_traj'][:, var_idx])
                c = cycles[spinup:][valid[spinup:]]
                v = res['xf_traj'][spinup:, var_idx][valid[spinup:]]
            else:
                c = cycles[spinup:]
                v = res['xf_traj'][spinup:, var_idx]

            color = palette.get(name, '#888888')
            ls    = '--' if name == 'Lorenz63' else '-'
            lw    = 2.0  if name == 'Lorenz63' else 1.2
            alpha = 0.6  if name == 'Lorenz63' else 0.85

            ax.plot(c, v, color=color, lw=lw, ls=ls,
                    alpha=alpha, label=name if var_idx == 0 else '_nolegend_')

        ax.set_ylabel(f'${var_name}$', fontsize=11)
        ax.grid(True, alpha=0.2)

        # mark spinup boundary
        if spinup > 0:
            ax.axvline(spinup, color='gray', lw=0.8,
                       ls=':', alpha=0.6, label='spinup end' if var_idx == 0 else '_nolegend_')

    axes[0].legend(loc='upper right', fontsize=8, ncol=4)
    axes[-1].set_xlabel('DA cycle', fontsize=11)

    fig.suptitle(
        f'Forecast ensemble mean vs truth  |  '
        f'Ne={cfg.Ne}  |  M={cfg.M}  |  σ_obs={cfg.sig_obs}'
        f'{title_suffix}',
        fontsize=11
    )
    plt.tight_layout()
    _savefig("trajectory_comparison")

# %% Start experiments
# ============================================================
# Benchmark 1: Baseline (reference conditions)
# ============================================================
print("\n" + "="*60)
print("BENCHMARK 1: Baseline (reference conditions)")
print("="*60)
generic_inflation_factor = 1.05

cfg_base = EnKFConfig(
    T=50, M=10, Ne=20, dt=0.01,
    p=1.0, sig_obs=1.0,
    use_localization=False, use_inflation=False, infl_factor=generic_inflation_factor,
    seed=10,
    enkf_type='deterministic_',  # 'stochastic' or 'deterministic'
)

results_base = run_all_models(cfg_base, surrogates, Xf_pools, xt, surrogates_palette)
_print_benchmark_table(results_base, "BENCHMARK 1 — Baseline Summary (T=10, M=10, Ne=20, p=1.0, σ=1.0)")

plot_rmse_comparison(results_base, surrogates_palette, cfg_base, f" — Baseline_{cfg_base.enkf_type}")
plot_spread_comparison(results_base, surrogates_palette, cfg_base, f" — Baseline_{cfg_base.enkf_type}")
# plot_talagrand_grid(results_base, surrogates_palette, cfg_base, ensemble_key='Xf_all', title_suffix=" — Baseline")
plot_spread_vs_rmse(results_base, surrogates_palette, cfg_base, f" — Baseline_{cfg_base.enkf_type}")
#plot_trajectory_comparison(results_base, surrogates_palette, cfg_base,
#                           spinup=0, title_suffix=f" — Baseline_{cfg_base.enkf_type}")

# Spaghetti: compare all models for each variable
for v in range(Nx):
    plot_ensemble_spaghetti_multi(results_base, surrogates_palette, cfg_base,
                                  cycle_range=(0, 10), var_idx=v)

# Detailed spaghetti + spread reduction for each architecture
plot_spread_reduction_multi(results_base, surrogates_palette, cfg_base,
                             cycle_range=(0, 10), var_idx=0)
#for name in results_base:
    #plot_ensemble_spaghetti(results_base[name], name, surrogates_palette, cfg_base,
    #                        cycle_range=(3, 8))
#    plot_spread_reduction(results_base[name], name, surrogates_palette, cfg_base,
#                          cycle_range=(3, 8), var_idx=0)
# ============================================================
# BENCHMARK 2: Stress test — longer forecast window (M)
# ============================================================
# %% Benchmark 2 — Forecast window sweep
print("\n" + "="*60)
print("BENCHMARK 2: Forecast window sweep (M)")
print("="*60)

M_values = [5, 10, 20, 50, 100, 200]
bench_M = {}
exp_M = {}
for M_val in M_values:
    print(f"\n--- M = {M_val} ---")
    cfg_m = EnKFConfig(
        T=50, M=M_val, Ne=20, dt=0.01,
        p=1.0, sig_obs=1.0,
        use_localization=False, use_inflation=False,
        infl_factor=generic_inflation_factor,
        seed=10, enkf_type='deterministic',
    )
    bench_M[M_val] = run_all_models(cfg_m, surrogates, Xf_pools, xt, surrogates_palette)
    exp_M[M_val] = cfg_m

# Summary: mean RMSE vs M for each architecture
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
all_names = ['Lorenz63'] + list(surrogates.keys())
for name in all_names:
    color = surrogates_palette.get(name, '#888888')
    ls = '--' if name == 'Lorenz63' else '-'
    frmse = [np.nanmean(bench_M[m][name]['errorf']) for m in M_values]
    armse = [np.nanmean(bench_M[m][name]['errora']) for m in M_values]
    axes[0].plot(M_values, frmse, 'o-', color=color, label=name, ls=ls)
    axes[1].plot(M_values, armse, 'o-', color=color, label=name, ls=ls)
axes[0].set_xlabel('Forecast steps (M)'); axes[0].set_ylabel('Mean Forecast RMSE')
axes[0].set_title('Forecast RMSE vs Forecast Window'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].set_xlabel('Forecast steps (M)'); axes[1].set_ylabel('Mean Analysis RMSE')
axes[1].set_title('Analysis RMSE vs Forecast Window'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
_savefig("b2_M_sweep")

# Pivot table: aRMSE per model × M
print("\nBENCHMARK 2 — Mean aRMSE by model and forecast window M:")
pivot_rows = []
for name in all_names:
    row = {'Model': name}
    for m in M_values:
        row[f'M={m}'] = round(np.nanmean(bench_M[m][name]['errora']), 4)
    pivot_rows.append(row)
print(pd.DataFrame(pivot_rows).to_string(index=False))

print("\nBENCHMARK 2 — Mean Spread by model and forecast window M:")
pivot_rows2 = []
for name in all_names:
    row = {'Model': name}
    for m in M_values:
        row[f'M={m}'] = round(np.nanmean(bench_M[m][name]['spread']), 4)
    pivot_rows2.append(row)
print(pd.DataFrame(pivot_rows2).to_string(index=False))

print("\nBENCHMARK 2 — Diverged per model and forecast window M:")
pivot_rows3 = []
for name in all_names:
    row = {'Model': name}
    for m in M_values:
        row[f'M={m}'] = bench_M[m][name]['diverged']
    pivot_rows3.append(row)
print(pd.DataFrame(pivot_rows3).to_string(index=False))

# Create spaghetti plots and spread reduction for the hardest M
m_hard = M_values[-1]

plot_spread_reduction_multi(bench_M[m_hard], surrogates_palette, exp_M[m_hard],
                             cycle_range=(0, 10), var_idx=0)
for v in range(Nx):
    plot_ensemble_spaghetti_multi(bench_M[m_hard], surrogates_palette, exp_M[m_hard],
                                  cycle_range=(0, 5), var_idx=v)

#for name in bench_M[m_hard]:
#    plot_ensemble_spaghetti(bench_M[m_hard][name], name, surrogates_palette, exp_M[m_hard],
#                            cycle_range=(0, 5))
#    plot_spread_reduction(bench_M[m_hard][name], name, surrogates_palette, exp_M[m_hard],
#                          cycle_range=(0, 5), var_idx=0)

# ============================================================
# BENCHMARK 3: Stress test — ensemble size (Ne)
# ============================================================
# %% Benchmark 3 — Ensemble size sweep
print("\n" + "="*60)
print("BENCHMARK 3: Ensemble size sweep (Ne)")
print("="*60)

Ne_values = [5, 10, 20, 50]
bench_Ne = {}
exp_Ne = {}
for Ne_val in Ne_values:
    print(f"\n--- Ne = {Ne_val} ---")
    cfg_ne = EnKFConfig(
        T=50, M=50, Ne=Ne_val, dt=0.01,
        p=1.0, sig_obs=1.0,
        use_localization=False, use_inflation=False,
        infl_factor=generic_inflation_factor,
        seed=10, enkf_type='deterministic',
    )
    bench_Ne[Ne_val] = run_all_models(cfg_ne, surrogates, Xf_pools, xt, surrogates_palette)
    exp_Ne[Ne_val] = cfg_ne

# Summary: mean RMSE vs Ne
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for name in all_names:
    color = surrogates_palette.get(name, '#888888')
    ls = '--' if name == 'Lorenz63' else '-'
    frmse = [np.nanmean(bench_Ne[n][name]['errorf']) for n in Ne_values]
    armse = [np.nanmean(bench_Ne[n][name]['errora']) for n in Ne_values]
    axes[0].plot(Ne_values, frmse, 'o-', color=color, label=name, ls=ls)
    axes[1].plot(Ne_values, armse, 'o-', color=color, label=name, ls=ls)
axes[0].set_xlabel('Ensemble size (Ne)'); axes[0].set_ylabel('Mean Forecast RMSE')
axes[0].set_title('Forecast RMSE vs Ensemble Size'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].set_xlabel('Ensemble size (Ne)'); axes[1].set_ylabel('Mean Analysis RMSE')
axes[1].set_title('Analysis RMSE vs Ensemble Size'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
_savefig("b3_Ne_sweep")

print("\nBENCHMARK 3 — Mean aRMSE by model and ensemble size Ne:")
pivot_rows = []
for name in all_names:
    row = {'Model': name}
    for n in Ne_values:
        row[f'Ne={n}'] = round(np.nanmean(bench_Ne[n][name]['errora']), 4)
    pivot_rows.append(row)
print(pd.DataFrame(pivot_rows).to_string(index=False))

print("\nBENCHMARK 3 — Mean Spread by model and ensemble size Ne:")
pivot_rows2 = []
for name in all_names:
    row = {'Model': name}
    for n in Ne_values:
        row[f'Ne={n}'] = round(np.nanmean(bench_Ne[n][name]['spread']), 4)
    pivot_rows2.append(row)
print(pd.DataFrame(pivot_rows2).to_string(index=False))

# Spaghetti for the hardest Ne
ne_hard = Ne_values[0]
plot_spread_reduction_multi(bench_Ne[ne_hard], surrogates_palette, exp_Ne[ne_hard],
                             cycle_range=(0, 10), var_idx=0)
for v in range(Nx):
    plot_ensemble_spaghetti_multi(bench_Ne[ne_hard], surrogates_palette, exp_Ne[ne_hard],
                                  cycle_range=(0, 5), var_idx=v)
#for name in bench_Ne[ne_hard]:
#    plot_ensemble_spaghetti(bench_Ne[ne_hard][name], name, surrogates_palette, exp_Ne[ne_hard],
#                            cycle_range=(0, 5))
#    plot_spread_reduction(bench_Ne[ne_hard][name], name, surrogates_palette, exp_Ne[ne_hard],
#                          cycle_range=(0, 5), var_idx=0)

# ============================================================
# BENCHMARK 4: Stress test — partial observations (p)
# ============================================================
# %% Benchmark 4 — Observation sparsity sweep
print("\n" + "="*60)
print("BENCHMARK 4: Observation sparsity sweep (p)")
print("="*60)

# For L63 with Nx=3: p=1.0 -> 3 obs, p=0.67 -> 2 obs, p=0.33 -> 1 obs
p_values = [1.0, 0.67, 0.33]
bench_p = {}
exp_p = {}
for p_val in p_values:
    Ny_actual = int(round(p_val * Nx))
    print(f"\n--- p = {p_val} (Ny = {Ny_actual}) ---")
    cfg_p = EnKFConfig(
        T=50, M=50, Ne=20, dt=0.01,
        p=p_val, sig_obs=1.0,
        use_localization=False, use_inflation=False,
        infl_factor=generic_inflation_factor,
        seed=10, enkf_type='deterministic',
    )
    bench_p[p_val] = run_all_models(cfg_p, surrogates, Xf_pools, xt, surrogates_palette)
    exp_p[p_val] = cfg_p

# Summary: mean RMSE vs p
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for name in all_names:
    color = surrogates_palette.get(name, '#888888')
    ls = '--' if name == 'Lorenz63' else '-'
    frmse = [np.nanmean(bench_p[p][name]['errorf']) for p in p_values]
    armse = [np.nanmean(bench_p[p][name]['errora']) for p in p_values]
    axes[0].plot(p_values, frmse, 'o-', color=color, label=name, ls=ls)
    axes[1].plot(p_values, armse, 'o-', color=color, label=name, ls=ls)
axes[0].set_xlabel('Observation fraction (p)'); axes[0].set_ylabel('Mean Forecast RMSE')
axes[0].set_title('Forecast RMSE vs Obs Fraction'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[0].invert_xaxis()
axes[1].set_xlabel('Observation fraction (p)'); axes[1].set_ylabel('Mean Analysis RMSE')
axes[1].set_title('Analysis RMSE vs Obs Fraction'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
axes[1].invert_xaxis()
plt.tight_layout()
_savefig("b4_p_sweep")

print("\nBENCHMARK 4 — Mean aRMSE by model and observation fraction p:")
pivot_rows = []
for name in all_names:
    row = {'Model': name}
    for p in p_values:
        row[f'p={p}'] = round(np.nanmean(bench_p[p][name]['errora']), 4)
    pivot_rows.append(row)
print(pd.DataFrame(pivot_rows).to_string(index=False))

print("\nBENCHMARK 4 — Mean Spread by model and observation fraction p:")
pivot_rows2 = []
for name in all_names:
    row = {'Model': name}
    for p in p_values:
        row[f'p={p}'] = round(np.nanmean(bench_p[p][name]['spread']), 4)
    pivot_rows2.append(row)
print(pd.DataFrame(pivot_rows2).to_string(index=False))

p_hard = p_values[-1]
plot_spread_reduction_multi(bench_p[p_hard], surrogates_palette, exp_p[p_hard],
                             cycle_range=(0, 10), var_idx=0)
for v in range(Nx):
    plot_ensemble_spaghetti_multi(bench_p[p_hard], surrogates_palette, exp_p[p_hard],
                                  cycle_range=(0, 5), var_idx=v)

# ============================================================
# BENCHMARK 5: Stress test — observation noise (sig_obs)
# ============================================================
# %% Benchmark 5 — Observation noise sweep
print("\n" + "="*60)
print("BENCHMARK 5: Observation noise sweep (sig_obs)")
print("="*60)

sig_values = [0.1, 0.5, 1.0, 2.0, 3.0]
bench_sig = {}
exp_sig = {}
for sig_val in sig_values:
    print(f"\n--- sig_obs = {sig_val} ---")
    cfg_sig = EnKFConfig(
        T=50, M=50, Ne=20, dt=0.01,
        p=1.0, sig_obs=sig_val,
        use_localization=False, use_inflation=False,
        infl_factor=generic_inflation_factor,
        seed=10, enkf_type='deterministic',
    )
    bench_sig[sig_val] = run_all_models(cfg_sig, surrogates, Xf_pools, xt, surrogates_palette)
    exp_sig[sig_val] = cfg_sig
    
# Summary: mean RMSE vs sig_obs
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for name in all_names:
    color = surrogates_palette.get(name, '#888888')
    ls = '--' if name == 'Lorenz63' else '-'
    frmse = [np.nanmean(bench_sig[s][name]['errorf']) for s in sig_values]
    armse = [np.nanmean(bench_sig[s][name]['errora']) for s in sig_values]
    axes[0].plot(sig_values, frmse, 'o-', color=color, label=name, ls=ls)
    axes[1].plot(sig_values, armse, 'o-', color=color, label=name, ls=ls)
axes[0].set_xlabel('Observation noise (σ_obs)'); axes[0].set_ylabel('Mean Forecast RMSE')
axes[0].set_title('Forecast RMSE vs Obs Noise'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].set_xlabel('Observation noise (σ_obs)'); axes[1].set_ylabel('Mean Analysis RMSE')
axes[1].set_title('Analysis RMSE vs Obs Noise'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
_savefig("b5_sig_sweep")

print("\nBENCHMARK 5 — Mean aRMSE by model and observation noise σ_obs:")
pivot_rows = []
for name in all_names:
    row = {'Model': name}
    for s in sig_values:
        row[f'σ={s}'] = round(np.nanmean(bench_sig[s][name]['errora']), 4)
    pivot_rows.append(row)
print(pd.DataFrame(pivot_rows).to_string(index=False))

print("\nBENCHMARK 5 — Mean Spread by model and observation noise σ_obs:")
pivot_rows2 = []
for name in all_names:
    row = {'Model': name}
    for s in sig_values:
        row[f'σ={s}'] = round(np.nanmean(bench_sig[s][name]['spread']), 4)
    pivot_rows2.append(row)
print(pd.DataFrame(pivot_rows2).to_string(index=False))

print("\n" + "="*60)
print("ALL BENCHMARKS COMPLETE")
print("="*60)

sig_hard = sig_values[-1]
plot_spread_reduction_multi(bench_sig[sig_hard], surrogates_palette, exp_sig[sig_hard],
                             cycle_range=(0, 10), var_idx=0)
for v in range(Nx):
    plot_ensemble_spaghetti_multi(bench_sig[sig_hard], surrogates_palette, exp_sig[sig_hard],
                                  cycle_range=(0, 5), var_idx=v)
# %%
