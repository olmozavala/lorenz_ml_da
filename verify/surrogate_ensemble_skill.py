# %%
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from os.path import join
from SurrogateModel import SurrogateModel
from lorenz.lorenz_systems import LorenzSystems

torch.set_default_dtype(torch.float64)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DT = 0.01
LAMBDA1 = 0.906            # L63 largest Lyapunov exponent (1/time)
CLIM_STD = 8.6              # climatological std (mean of per-var stds)
CLIM_STD_PER_VAR = np.array([7.93, 9.01, 8.59])
OBS_INTERVAL = 8            # observe every 8 steps
SIGMA_OBS = 1.0             # observation noise std
N_ENS = 50                  # ensemble size
DELTA_0 = 0.1               # initial perturbation magnitude
VAR_NAMES = ['x', 'y', 'z']
MODEL_NAMES = ['DenseNN', 'ResDenseNN', 'LSTMNN', 'RNN_relu', 'RNN_tanh']

model_dir = 'models'

# ---------------------------------------------------------------------------
# Model initialisation (unchanged)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def generate_centered_ensemble(ic, N_ens, delta_0):
    """Generate N_ens ICs centred exactly on *ic*."""
    perturbations = np.random.randn(N_ens, 3) * delta_0
    perturbations -= perturbations.mean(axis=0)
    return ic + perturbations


@torch.no_grad()
def batched_surrogate_rollout(surrogate, ics, K):
    """
    Batched autoregressive rollout — all members in a single forward pass.

    Parameters
    ----------
    surrogate : SurrogateModel
    ics : np.ndarray, shape (N_ens, 3)
    K : int  — number of forward steps

    Returns
    -------
    np.ndarray, shape (K, N_ens, 3)
    """
    assert surrogate.prev_time_steps == 1, "Batched rollout only supports prev_time_steps=1"
    model = surrogate.model
    mean = surrogate.train_mean          # (3,)
    std = surrogate.train_std            # (3,)
    device = surrogate.device
    dtype = next(model.parameters()).dtype

    states = ics.copy()                  # (N_ens, 3)
    trajectory = np.empty((K, ics.shape[0], 3), dtype=np.float64)

    for k in range(K):
        normed = (states - mean) / std
        x = torch.as_tensor(normed, dtype=dtype, device=device)
        pred = model(x).cpu().numpy().astype(np.float64)
        states = pred * std + mean
        trajectory[k] = states

    return trajectory


def true_l63_ensemble_rollout(ics, K, dt=DT):
    """Run RK45 L63 integrator for each ensemble member."""
    N_ens = ics.shape[0]
    trajectory = np.empty((K, N_ens, 3), dtype=np.float64)
    for j in range(N_ens):
        traj_j = LorenzSystems.generate_trajectory_fast('63', ics[j], dt, K + 1)
        trajectory[:, j, :] = traj_j[1:]
    return trajectory


# ---------------------------------------------------------------------------
# Initialise models and generate truth trajectory
# ---------------------------------------------------------------------------
init_steps = 1
surrogates, surrogates_palette = init_models(init_steps)

x0 = np.array([1.0, 1.0, 1.1], dtype=np.float64)
n_total = 20_000
print("Generating truth trajectory ...")
long_traj = LorenzSystems.generate_trajectory_fast('63', x0, DT, n_total)

# %%  ========================================================================
# Experiment B — free-run ensemble spread
# =========================================================================
print("\n=== Experiment B: free-run ensemble spread ===")

K_spread = 1000
n_ics = 30
ic_spacing = 500
ic_indices = np.arange(1000, 1000 + n_ics * ic_spacing, ic_spacing)

all_models = ['Truth'] + MODEL_NAMES

# Storage: model → (n_ics, K, 3) per-var spread and (n_ics, K) mean spread
all_spreads = {m: np.empty((n_ics, K_spread, 3)) for m in all_models}
all_spreads_mean = {m: np.empty((n_ics, K_spread)) for m in all_models}

for i, idx in enumerate(ic_indices):
    ic = long_traj[idx]
    np.random.seed(1000 + i)  # reproducible per-IC perturbations
    ens_ics = generate_centered_ensemble(ic, N_ENS, DELTA_0)

    # Truth
    traj_truth = true_l63_ensemble_rollout(ens_ics, K_spread, DT)
    sp = traj_truth.std(axis=1)           # (K, 3)
    all_spreads['Truth'][i] = sp
    all_spreads_mean['Truth'][i] = sp.mean(axis=1)

    # Surrogates
    for name in MODEL_NAMES:
        traj = batched_surrogate_rollout(surrogates[name], ens_ics, K_spread)
        sp = traj.std(axis=1)
        all_spreads[name][i] = sp
        all_spreads_mean[name][i] = sp.mean(axis=1)

    if (i + 1) % 10 == 0:
        print(f"  IC {i+1}/{n_ics} done")

# Summary across ICs
mean_spread = {m: all_spreads_mean[m].mean(axis=0) for m in all_models}
std_spread = {m: all_spreads_mean[m].std(axis=0) for m in all_models}
mean_spread_per_var = {m: all_spreads[m].mean(axis=0) for m in all_models}

# %%  ========================================================================
# Metric 1 — exp_B_spread_growth.png
# =========================================================================
print("\n--- Metric 1: spread growth plot ---")

steps = np.arange(1, K_spread + 1)
lyap_times = steps * DT * LAMBDA1
t_phys = steps * DT

fig, ax = plt.subplots(figsize=(10, 6))

for name in all_models:
    color = surrogates_palette[name]
    ax.semilogy(lyap_times, mean_spread[name], label=name, color=color, lw=1.5)
    upper = mean_spread[name] + std_spread[name]
    lower = mean_spread[name] - std_spread[name]
    # On log scale, clip lower bound to half the mean to avoid bands dominating
    lower = np.maximum(lower, mean_spread[name] * 0.5)
    ax.fill_between(lyap_times, lower, upper, color=color, alpha=0.10)

# Theoretical exponential growth
ax.semilogy(lyap_times, DELTA_0 * np.exp(LAMBDA1 * t_phys),
            '--', color='gray', lw=1.0, label=r'$\delta_0 \cdot e^{\lambda_1 t}$')

ax.axhline(CLIM_STD, ls='--', color='gray', lw=0.8, label=f'Clim $\\sigma$ = {CLIM_STD}')
ax.axhline(DELTA_0, ls='--', color='lightgray', lw=0.8, label=f'$\\delta_0$ = {DELTA_0}')

# Vertical dotted lines at observation intervals
obs_lt = OBS_INTERVAL * DT * LAMBDA1
n_obs_lines = int(lyap_times[-1] / obs_lt) + 1
for iv in range(1, n_obs_lines + 1):
    lbl = 'Obs interval' if iv == 1 else None
    ax.axvline(iv * obs_lt, ls=':', color='black', lw=0.5, alpha=0.3, label=lbl)

ax.set_xlabel('Lyapunov times')
ax.set_ylabel('Ensemble spread $\\sigma(t)$')
ax.set_title('Experiment B: Free-run ensemble spread growth')
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig('exp_B_spread_growth.png', dpi=300)
plt.close()
print("  Saved exp_B_spread_growth.png")

# %%  ========================================================================
# Metrics 2-4 — scalar summaries
# =========================================================================
print("\n--- Metric 2: Peak spread fraction of climatological variance ---")
print(f"{'Model':<12s}  {'Peak spread':>11s}  {'% of clim':>9s}")
peak_spread_pct = {}
for name in all_models:
    peak = mean_spread[name].max()
    pct = peak / CLIM_STD * 100
    peak_spread_pct[name] = pct
    print(f"  {name:<12s}  {peak:11.4f}  {pct:8.1f}%")

print("\n--- Metric 3: Spread at first observation interval (step 8) ---")
print(f"{'Model':<12s}  {'Spread(8)':>11s}  {'/ clim':>9s}")
spread_obs1 = {}
for name in all_models:
    s8 = mean_spread[name][OBS_INTERVAL - 1]  # 0-indexed
    spread_obs1[name] = s8
    print(f"  {name:<12s}  {s8:11.6f}  {s8/CLIM_STD:9.6f}")

print("\n--- Metric 4: Collapse time & peak time ---")
print(f"{'Model':<12s}  {'Collapse (LT)':>13s}  {'Peak (LT)':>10s}")
collapse_time_LT = {}
peak_time_LT = {}
for name in all_models:
    curve = mean_spread[name]
    threshold = 0.1 * CLIM_STD  # 0.86
    exceeds = np.where(curve > threshold)[0]
    if len(exceeds) == 0:
        # Never exceeded threshold → collapse time = 0
        ct_lt = 0.0
    else:
        # Check if spread falls back below threshold after first exceeding it
        first_exceed = exceeds[0]
        after_exceed = curve[first_exceed:]
        below_after = np.where(after_exceed < threshold)[0]
        if len(below_after) == 0:
            ct_lt = float('inf')  # never collapses
        else:
            ct_lt = (first_exceed + below_after[0] + 1) * DT * LAMBDA1
    collapse_time_LT[name] = ct_lt

    peak_step = np.argmax(curve)
    pt_lt = (peak_step + 1) * DT * LAMBDA1
    peak_time_LT[name] = pt_lt
    ct_str = f"{ct_lt:.4f}" if ct_lt < 1e6 else "inf"
    print(f"  {name:<12s}  {ct_str:>13s}  {pt_lt:10.4f}")

# %%  ========================================================================
# Metric 5 — per-variable spread: exp_B_spread_per_variable.png
# =========================================================================
print("\n--- Metric 5: per-variable spread plot ---")

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for i, (var_name, ax) in enumerate(zip(VAR_NAMES, axes)):
    for name in all_models:
        color = surrogates_palette[name]
        ax.semilogy(lyap_times, mean_spread_per_var[name][:, i],
                    label=name, color=color, lw=1.5)
    ax.axhline(CLIM_STD_PER_VAR[i], ls='--', color='gray', lw=0.8,
               label=f'Clim $\\sigma$ = {CLIM_STD_PER_VAR[i]:.2f}')
    ax.axhline(DELTA_0, ls='--', color='lightgray', lw=0.8)
    for iv in range(1, n_obs_lines + 1):
        ax.axvline(iv * obs_lt, ls=':', color='black', lw=0.5, alpha=0.3)
    ax.set_xlabel('Lyapunov times')
    ax.set_title(f'Variable {var_name}')

axes[0].set_ylabel('Ensemble spread $\\sigma(t)$')
axes[0].legend(fontsize=8)
plt.suptitle('Experiment B: Per-variable spread growth', y=1.02)
plt.tight_layout()
plt.savefig('exp_B_spread_per_variable.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved exp_B_spread_per_variable.png")

# %%  ========================================================================
# Experiment C — stochastic EnKF DA cycle
# =========================================================================
print("\n=== Experiment C: stochastic EnKF DA cycle ===")

N_DA_STEPS = 8000
N_OBS_TIMES = N_DA_STEPS // OBS_INTERVAL   # 1000

# Truth trajectory for DA (with spinup)
x0_da = np.array([1.0, 1.0, 1.1], dtype=np.float64)
spinup = 1000
truth_full = LorenzSystems.generate_trajectory_fast(
    '63', x0_da, DT, N_DA_STEPS + spinup + 1)
truth_da = truth_full[spinup:]   # shape (N_DA_STEPS + 1, 3)

# Observations: truth at obs times + noise
obs_step_indices = np.arange(1, N_OBS_TIMES + 1) * OBS_INTERVAL
np.random.seed(42)
obs_noise_all = np.random.randn(N_OBS_TIMES, 3) * SIGMA_OBS
observations = truth_da[obs_step_indices] + obs_noise_all

# Pre-generate shared random draws for reproducibility
initial_ensemble = generate_centered_ensemble(truth_da[0], N_ENS, DELTA_0)
# Observation perturbations for EnKF update at each cycle
enkf_obs_noise = np.random.randn(N_OBS_TIMES, N_ENS, 3) * SIGMA_OBS

R = SIGMA_OBS**2 * np.eye(3)


def run_enkf_da(forward_fn, truth_traj, observations, initial_ens,
                obs_perturbations, obs_interval, R):
    """
    Run a stochastic EnKF DA cycle.

    Parameters
    ----------
    forward_fn : callable  (ensemble (N,3)) -> (N,3) after obs_interval steps
    truth_traj : (T, 3)
    observations : (n_obs, 3)
    initial_ens : (N_ens, 3)
    obs_perturbations : (n_obs, N_ens, 3)
    obs_interval : int
    R : (3, 3)

    Returns
    -------
    spreads, rmses, ranks
    """
    n_obs = len(observations)
    N_ens = initial_ens.shape[0]
    ensemble = initial_ens.copy()

    spreads = np.empty(n_obs)
    rmses = np.empty(n_obs)
    ranks = np.empty((n_obs, 3), dtype=int)

    for t in range(n_obs):
        # --- Forecast ---
        ensemble = forward_fn(ensemble)

        # --- Diagnostics (on forecast ensemble, before update) ---
        ens_mean = ensemble.mean(axis=0)
        truth_now = truth_traj[(t + 1) * obs_interval]

        spreads[t] = ensemble.std(axis=0).mean()
        rmses[t] = np.sqrt(((ens_mean - truth_now) ** 2).mean())

        for v in range(3):
            sorted_ens = np.sort(ensemble[:, v])
            ranks[t, v] = np.searchsorted(sorted_ens, truth_now[v])

        # --- EnKF analysis update ---
        y_obs = observations[t]
        X_f = ensemble.T                               # (3, N_ens)
        x_bar = X_f.mean(axis=1, keepdims=True)        # (3, 1)
        A = X_f - x_bar                                # (3, N_ens)
        P_f = (1.0 / (N_ens - 1)) * (A @ A.T)         # (3, 3)
        K_gain = np.linalg.solve((P_f + R).T, P_f.T).T # (3, 3)

        y_pert = y_obs + obs_perturbations[t]           # (N_ens, 3)
        innovations = y_pert - ensemble                 # (N_ens, 3)
        ensemble = ensemble + (K_gain @ innovations.T).T

        if (t + 1) % 200 == 0:
            print(f"    cycle {t+1}/{n_obs}")

    return spreads, rmses, ranks


def make_surrogate_forward(surrogate, obs_interval):
    def forward(ensemble):
        traj = batched_surrogate_rollout(surrogate, ensemble, obs_interval)
        return traj[-1]
    return forward


def truth_forward(ensemble):
    traj = true_l63_ensemble_rollout(ensemble, OBS_INTERVAL, DT)
    return traj[-1]


# Run DA for each model
da_results = {}

for name in all_models:
    print(f"  Running EnKF for {name} ...")
    if name == 'Truth':
        fwd = truth_forward
    else:
        fwd = make_surrogate_forward(surrogates[name], OBS_INTERVAL)

    spreads, rmses, ranks = run_enkf_da(
        fwd, truth_da, observations, initial_ensemble,
        enkf_obs_noise, OBS_INTERVAL, R)
    da_results[name] = {'spreads': spreads, 'rmses': rmses, 'ranks': ranks}

# %%  ========================================================================
# Metric 6 — spread-to-error ratio: da_spread_error_ratio.png
# =========================================================================
print("\n--- Metric 6: spread-error ratio plot ---")

fig, ax = plt.subplots(figsize=(10, 5))
window = 50

for name in all_models:
    spreads = da_results[name]['spreads']
    rmses = da_results[name]['rmses']
    ratio = spreads / np.maximum(rmses, 1e-10)
    ratio_smooth = np.convolve(ratio, np.ones(window) / window, mode='valid')
    ax.plot(np.arange(len(ratio_smooth)), ratio_smooth,
            color=surrogates_palette[name], label=name, lw=1.0)

ax.axhline(1.0, ls='--', color='gray', lw=0.8, label='Ideal ratio = 1.0')
ax.set_xlabel('Assimilation cycle')
ax.set_ylabel('$\\sigma$ / RMSE')
ax.set_title('DA Spread-Error Ratio')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('da_spread_error_ratio.png', dpi=300)
plt.close()
print("  Saved da_spread_error_ratio.png")

# %%  ========================================================================
# Metric 7 — rank histograms: rank_histograms.png
# =========================================================================
print("\n--- Metric 7: rank histograms ---")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes_flat = axes.flatten()

for idx, name in enumerate(all_models):
    ax = axes_flat[idx]
    ranks = da_results[name]['ranks']
    all_ranks = ranks.flatten()

    bins = np.arange(N_ENS + 2) - 0.5
    ax.hist(all_ranks, bins=bins, density=True,
            color=surrogates_palette[name], alpha=0.7, edgecolor='black', lw=0.3)
    expected = 1.0 / (N_ENS + 1)
    ax.axhline(expected, ls='--', color='black', lw=0.8,
               label=f'Expected = {expected:.4f}')
    ax.set_title(name)
    ax.set_xlabel('Rank')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=8)

plt.suptitle('Rank Histograms (Talagrand Diagrams)')
plt.tight_layout()
plt.savefig('rank_histograms.png', dpi=300)
plt.close()
print("  Saved rank_histograms.png")

# %%  ========================================================================
# Summary CSV — tier2_summary.csv
# =========================================================================
print("\n--- Summary CSV ---")

rows = []
for name in all_models:
    # Rank histogram shape classification
    ranks = da_results[name]['ranks'].flatten()
    hist_counts, _ = np.histogram(ranks, bins=np.arange(N_ENS + 2))
    edge_bins = np.concatenate([hist_counts[:5], hist_counts[-5:]]).mean()
    center_bins = hist_counts[20:31].mean()
    ec_ratio = edge_bins / max(center_bins, 1e-10)

    if ec_ratio > 1.5:
        shape = 'U-shape'
    elif ec_ratio < 0.67:
        shape = 'dome'
    else:
        shape = 'flat'

    # Mean spread/RMSE ratio
    sp = da_results[name]['spreads']
    rm = da_results[name]['rmses']
    mean_ratio = (sp / np.maximum(rm, 1e-10)).mean()

    ct = collapse_time_LT[name]
    ct_str = f"{ct:.4f}" if ct < 1e6 else "inf"

    rows.append({
        'model': name,
        'peak_spread_pct': round(peak_spread_pct[name], 2),
        'spread_at_obs1': round(spread_obs1[name], 6),
        'collapse_time_LT': ct_str,
        'peak_time_LT': round(peak_time_LT[name], 4),
        'mean_sigma_rmse_ratio': round(mean_ratio, 4),
        'rank_hist_shape': shape,
    })

df = pd.DataFrame(rows)
df.to_csv('tier2_summary.csv', index=False)
print(df.to_string(index=False))
print("\nSaved tier2_summary.csv")
print("\nDone.")
