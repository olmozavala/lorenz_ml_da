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
n_steps = len(np.arange(0, 30, dt))

traj = LorenzSystems.generate_trajectory_fast('63', x0, dt, spinup_steps + 1)
xt_init = traj[-1, :]  # on the attractor after spin-up

# Create a perturbed initial condition and propagate both
d0 = 0.05
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

    # For Talagrand diagrams: store forecast ensemble + truth at each cycle
    Xf_all = np.zeros((cfg.T, Nx, cfg.Ne))  # forecast ensembles
    Xa_all = np.zeros((cfg.T, Nx, cfg.Ne))  # analysis ensembles

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

        # Store trajectories and full ensembles
        xt_traj[k] = xt_k
        xf_traj[k] = xf_k
        Xf_all[k] = Xf_k.copy()

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
        Xa_all[k] = Xa_k.copy()

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
        'Xf_all': Xf_all,     # [T, Nx, Ne] forecast ensembles
        'Xa_all': Xa_all,      # [T, Nx, Ne] analysis ensembles
        'diverged': diverged,
    }

# ============================================================
# Talagrand (Rank Histogram) computation
# ============================================================
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

# ============================================================
# Plotting helpers
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
    plt.savefig(f'outputs/EnKF_RMSE_Comparison_p{cfg.p}_sig_obs{cfg.sig_obs}_M{cfg.M}_Ne{cfg.Ne}.png', dpi=300, bbox_inches='tight')
    plt.show()

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
    plt.savefig(f'outputs/EnKF_Spread_Comparison_p{cfg.p}_sig_obs{cfg.sig_obs}_M{cfg.M}_Ne{cfg.Ne}.png', dpi=300, bbox_inches='tight')
    plt.show()

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
    ax.set_xlabel('Mean Ensemble Spread'); ax.set_ylabel('Mean Forecast RMSE')
    ax.set_title(f'Spread vs RMSE (Calibration Diagnostic){title_suffix}')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'outputs/EnKF_Spread_vs_RMSE_p{cfg.p}_sig_obs{cfg.sig_obs}_M{cfg.M}_Ne{cfg.Ne}.png', dpi=300, bbox_inches='tight')
    plt.show()

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
    plt.savefig(f'outputs/Talagrand_Diagrams_{ens_label}_p{cfg.p}_sig_obs{cfg.sig_obs}_M{cfg.M}_Ne{cfg.Ne}.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_summary_bars(results, palette):
    """Bar chart: mean forecast and analysis RMSE per architecture."""
    model_names = list(results.keys())
    mean_f = [np.nanmean(results[n]['errorf']) for n in model_names]
    mean_a = [np.nanmean(results[n]['errora']) for n in model_names]
    colors = [palette.get(n, '#888888') for n in model_names]

    x_pos = np.arange(len(model_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x_pos - width/2, mean_f, width, label='Forecast', color=colors, alpha=0.7, edgecolor='black')
    ax.bar(x_pos + width/2, mean_a, width, label='Analysis', color=colors, alpha=1.0, edgecolor='black')
    ax.set_xticks(x_pos); ax.set_xticklabels(model_names, rotation=15)
    ax.set_ylabel('Mean RMSE'); ax.set_title('Mean RMSE by Architecture')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'outputs/EnKF_Mean_RMSE_by_Architecture.png', dpi=300, bbox_inches='tight')
    plt.show()

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

# ============================================================
# BENCHMARK 1: Baseline experiment (reference conditions)
# ============================================================
# %% Benchmark 1 — Baseline
print("\n" + "="*60)
print("BENCHMARK 1: Baseline (reference conditions)")
print("="*60)

cfg_base = EnKFConfig(
    T=50, M=10, Ne=20, dt=0.01,
    p=1.0, sig_obs=1.1,
    use_localization=False, use_inflation=False,
    seed=10,
)

results_base = run_all_models(cfg_base, surrogates, Xf_pools, xt, surrogates_palette)

plot_rmse_comparison(results_base, surrogates_palette, cfg_base, " — Baseline")
plot_spread_comparison(results_base, surrogates_palette, cfg_base, " — Baseline")
plot_spread_vs_rmse(results_base, surrogates_palette, cfg_base, " — Baseline")
plot_talagrand_grid(results_base, surrogates_palette, cfg_base, ensemble_key='Xf_all', title_suffix=" — Baseline")
plot_talagrand_grid(results_base, surrogates_palette, cfg_base, ensemble_key='Xa_all', title_suffix=" — Baseline")
plot_summary_bars(results_base, surrogates_palette)

# ============================================================
# BENCHMARK 2: Stress test — longer forecast window (M)
# ============================================================
# %% Benchmark 2 — Forecast window sweep
print("\n" + "="*60)
print("BENCHMARK 2: Forecast window sweep (M)")
print("="*60)

M_values = [5, 10, 20, 40]
bench_M = {}
for M_val in M_values:
    print(f"\n--- M = {M_val} ---")
    cfg_m = EnKFConfig(
        T=50, M=M_val, Ne=20, dt=0.01,
        p=1.0, sig_obs=1.1,
        use_localization=False, use_inflation=False,
        seed=10,
    )
    bench_M[M_val] = run_all_models(cfg_m, surrogates, Xf_pools, xt, surrogates_palette)

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
plt.savefig(f'outputs/EnKF_RMSE_vs_M.png', dpi=300, bbox_inches='tight')
plt.show()

# Talagrand for hardest M
m_hard = M_values[-1]
cfg_hard_m = EnKFConfig(T=50, M=m_hard, Ne=20, dt=0.01, p=1.0, sig_obs=1.1,
                         use_localization=False, use_inflation=False, seed=10)
plot_talagrand_grid(bench_M[m_hard], surrogates_palette, cfg_hard_m,
                    title_suffix=f" — M={m_hard} (hardest)")

# ============================================================
# BENCHMARK 3: Stress test — ensemble size (Ne)
# ============================================================
# %% Benchmark 3 — Ensemble size sweep
print("\n" + "="*60)
print("BENCHMARK 3: Ensemble size sweep (Ne)")
print("="*60)

Ne_values = [5, 10, 20, 50]
bench_Ne = {}
for Ne_val in Ne_values:
    print(f"\n--- Ne = {Ne_val} ---")
    cfg_ne = EnKFConfig(
        T=50, M=10, Ne=Ne_val, dt=0.01,
        p=1.0, sig_obs=1.1,
        use_localization=False, use_inflation=False,
        seed=10,
    )
    bench_Ne[Ne_val] = run_all_models(cfg_ne, surrogates, Xf_pools, xt, surrogates_palette)

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
plt.savefig(f'outputs/EnKF_RMSE_vs_Ne.png', dpi=300, bbox_inches='tight')
plt.show()

# Talagrand for smallest Ne
ne_hard = Ne_values[0]
cfg_hard_ne = EnKFConfig(T=50, M=10, Ne=ne_hard, dt=0.01, p=1.0, sig_obs=1.1,
                          use_localization=False, use_inflation=False, seed=10)
plot_talagrand_grid(bench_Ne[ne_hard], surrogates_palette, cfg_hard_ne,
                    title_suffix=f" — Ne={ne_hard} (smallest)")

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
for p_val in p_values:
    Ny_actual = int(round(p_val * Nx))
    print(f"\n--- p = {p_val} (Ny = {Ny_actual}) ---")
    cfg_p = EnKFConfig(
        T=50, M=10, Ne=20, dt=0.01,
        p=p_val, sig_obs=1.1,
        use_localization=False, use_inflation=False,
        seed=10,
    )
    bench_p[p_val] = run_all_models(cfg_p, surrogates, Xf_pools, xt, surrogates_palette)

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
plt.savefig(f'outputs/EnKF_RMSE_vs_p.png', dpi=300, bbox_inches='tight')
plt.show()

# Talagrand for sparsest observations
p_hard = p_values[-1]
cfg_hard_p = EnKFConfig(T=50, M=10, Ne=20, dt=0.01, p=p_hard, sig_obs=1.1,
                         use_localization=False, use_inflation=False, seed=10)
plot_talagrand_grid(bench_p[p_hard], surrogates_palette, cfg_hard_p,
                    title_suffix=f" — p={p_hard} (1 obs only)")

# ============================================================
# BENCHMARK 5: Stress test — observation noise (sig_obs)
# ============================================================
# %% Benchmark 5 — Observation noise sweep
print("\n" + "="*60)
print("BENCHMARK 5: Observation noise sweep (sig_obs)")
print("="*60)

sig_values = [0.1, 0.5, 1.1, 3.0]
bench_sig = {}
for sig_val in sig_values:
    print(f"\n--- sig_obs = {sig_val} ---")
    cfg_sig = EnKFConfig(
        T=50, M=10, Ne=20, dt=0.01,
        p=1.0, sig_obs=sig_val,
        use_localization=False, use_inflation=False,
        seed=10,
    )
    bench_sig[sig_val] = run_all_models(cfg_sig, surrogates, Xf_pools, xt, surrogates_palette)

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
plt.tight_layout(); plt.show()

# Talagrand for noisiest observations
sig_hard = sig_values[-1]
cfg_hard_sig = EnKFConfig(T=50, M=10, Ne=20, dt=0.01, p=1.0, sig_obs=sig_hard,
                           use_localization=False, use_inflation=False, seed=10)
plot_talagrand_grid(bench_sig[sig_hard], surrogates_palette, cfg_hard_sig,
                    title_suffix=f" — σ_obs={sig_hard} (noisiest)")

# ============================================================
# BENCHMARK 6: Combined stress — worst case
# ============================================================
# %% Benchmark 6 — Combined stress test
print("\n" + "="*60)
print("BENCHMARK 6: Combined stress (M=40, Ne=10, p=0.33, σ=3.0)")
print("="*60)

cfg_stress = EnKFConfig(
    T=50, M=40, Ne=10, dt=0.01,
    p=0.33, sig_obs=3.0,
    use_localization=False, use_inflation=False,
    seed=10,
)
results_stress = run_all_models(cfg_stress, surrogates, Xf_pools, xt, surrogates_palette)

plot_rmse_comparison(results_stress, surrogates_palette, cfg_stress, " — Combined Stress")
plot_spread_comparison(results_stress, surrogates_palette, cfg_stress, " — Combined Stress")
plot_spread_vs_rmse(results_stress, surrogates_palette, cfg_stress, " — Combined Stress")
plot_talagrand_grid(results_stress, surrogates_palette, cfg_stress,
                    ensemble_key='Xf_all', title_suffix=" — Combined Stress (Forecast)")
plot_talagrand_grid(results_stress, surrogates_palette, cfg_stress,
                    ensemble_key='Xa_all', title_suffix=" — Combined Stress (Analysis)")
plot_summary_bars(results_stress, surrogates_palette)

# %%
