import numpy as np
from scipy.linalg import sqrtm
from lorenz.lorenz_systems import LorenzSystems
from SurrogateModel import SurrogateModel
from metrics import energy_score

# ============================================================
# EnKF configuration dataclass
# ============================================================
class EnKFConfig:
    """All tunable parameters for a single EnKF experiment.

    Random number streams
    ---------------------
    Two independent streams are exposed so that EnKF and PF runs can be
    compared on identical truth + observation realizations while keeping
    filter-internal stochasticity independent:

    - ``obs_seed``    : controls observation-component selection and
                        observation noise. Share this across filters
                        being compared head-to-head.
    - ``filter_seed`` : controls ensemble subset draws and any
                        filter-internal stochasticity (perturbed obs in
                        stochastic EnKF; resampling/jitter in PF).

    The legacy ``seed`` argument is retained as a convenience alias: when
    given, it sets both ``obs_seed`` and ``filter_seed`` to the same
    value. Explicit ``obs_seed`` / ``filter_seed`` take precedence.
    """
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
        seed=None,          # legacy alias: sets both seeds when provided
        obs_seed=None,      # truth/observation realization
        filter_seed=None,   # filter-internal stochasticity
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
        self.enkf_type = enkf_type

        # Resolve seeds: explicit values win; otherwise fall back to `seed`;
        # otherwise default to 10 (preserves prior default behaviour).
        if obs_seed is None:
            obs_seed = seed if seed is not None else 10
        if filter_seed is None:
            filter_seed = seed if seed is not None else 10
        self.obs_seed = obs_seed
        self.filter_seed = filter_seed
        # Keep `seed` accessible for any downstream code that inspects it.
        self.seed = seed if seed is not None else obs_seed

# ============================================================
# Core EnKF cycle and helper functions
# ============================================================
# Localization matrix
def create_localization_matrix(r, n):
    """Gaussian (Gaspari-Cohn-like) localization matrix with periodic distance."""
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            dij = min(abs(i - j), abs((n - 1) - j + i))
            L[i, j] = (dij ** 2) / (2 * r ** 2)
            L[j, i] = L[i, j]
    return np.exp(-L)

# Surrogate model forecaster
def make_surrogate_forecaster(surrogate, M):
    """Wrap a SurrogateModel into a forecast callable.
    Returns full trajectory (M+1, Nx) including the initial state."""
    def forecaster(state):
        sol = surrogate.rollout(state, num_steps=M)
        sol = np.asarray(sol)
        init = np.asarray(state).reshape(1, -1)
        return np.vstack([init, sol])
    return forecaster

# Lorenz system forecaster
def make_lorenz_forecaster(dt, M):
    """Wrap the Lorenz solver into a forecast callable.
    Returns full trajectory (M+1, Nx) including the initial state."""
    def forecaster(state):
        traj = LorenzSystems.generate_trajectory_fast('63', np.asarray(state), dt, M + 1)
        return traj
    return forecaster

# EnKF experiment runner
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
        'errorf'         : np.ndarray (T,)  – forecast RMSE per cycle
        'errora'         : np.ndarray (T,)  – analysis RMSE per cycle
        'spread'         : np.ndarray (T,)  – ensemble spread per cycle
        'errorf_es'      : np.ndarray (T,)  – forecast Energy Score per cycle
        'errora_es'      : np.ndarray (T,)  – analysis Energy Score per cycle
        'errorf_es_acc'  : np.ndarray (T,)  – forecast ES accuracy term
        'errorf_es_spr'  : np.ndarray (T,)  – forecast ES spread term
        'errora_es_acc'  : np.ndarray (T,)  – analysis ES accuracy term
        'errora_es_spr'  : np.ndarray (T,)  – analysis ES spread term
        'xt_traj'        : np.ndarray (T, Nx) – true state at each cycle
        'xf_traj'        : np.ndarray (T, Nx) – forecast mean at each cycle
        'xa_traj'        : np.ndarray (T, Nx) – analysis mean at each cycle
    """
    cfg = config

    # Split RNGs: see EnKFConfig docstring for the rationale.
    # `RandomState` is used (rather than the modern Generator API) to keep
    # the random sequence semantics close to the original implementation.
    obs_rng = np.random.RandomState(cfg.obs_seed)
    filt_rng = np.random.RandomState(cfg.filter_seed)

    Nx = xt_0.shape[0]
    I = np.eye(Nx)
    ones = np.ones(cfg.Ne)

    # Draw ensemble subset from pool (filter-side stochasticity)
    ind = filt_rng.permutation(Xf_pool.shape[0])[:cfg.Ne]
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
    # Energy Score and its decomposition (accuracy / spread terms)
    errorf_es = np.zeros(cfg.T)
    errora_es = np.zeros(cfg.T)
    errorf_es_acc = np.zeros(cfg.T)
    errorf_es_spr = np.zeros(cfg.T)
    errora_es_acc = np.zeros(cfg.T)
    errora_es_spr = np.zeros(cfg.T)
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
            errorf_es[k:] = np.nan
            errora_es[k:] = np.nan
            errorf_es_acc[k:] = np.nan
            errorf_es_spr[k:] = np.nan
            errora_es_acc[k:] = np.nan
            errora_es_spr[k:] = np.nan
            diverged = True
            break

        # --- Forecast statistics ---
        xf_k = np.mean(Xf_k, axis=1)
        Pf_k = L * np.cov(Xf_k)  # Schur product (localization or identity)

        # Ensemble spread: mean std across state variables
        spread[k] = np.mean(np.std(Xf_k, axis=1))

        # Forecast RMSE
        errorf[k] = np.linalg.norm(xf_k - xt_k)

        # Forecast Energy Score (uniform weights, full state)
        es_f, acc_f, spr_f = energy_score(Xf_k, xt_k)
        errorf_es[k] = es_f
        errorf_es_acc[k] = acc_f
        errorf_es_spr[k] = spr_f

        # Store trajectories
        xt_traj[k] = xt_k
        xf_traj[k] = xf_k
        Xf_all[k] = Xf_k.copy()

        # --- Observations (obs-side stochasticity) ---
        obs_comp = obs_rng.permutation(Nx)[:Ny]
        H_k = I[obs_comp, :]
        y_k = H_k @ xt_k + cfg.sig_obs * obs_rng.standard_normal(Ny)

        if cfg.enkf_type == 'stochastic':
            # ========== Stochastic EnKF (perturbed observations) ==========
            # Perturbations are filter-internal stochasticity → filt_rng
            Yobs_k = np.outer(y_k, np.ones(cfg.Ne)) + cfg.sig_obs * filt_rng.standard_normal((Ny, cfg.Ne))

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

        # Analysis Energy Score (uniform weights, full state)
        es_a, acc_a, spr_a = energy_score(Xa_k, xt_k)
        errora_es[k] = es_a
        errora_es_acc[k] = acc_a
        errora_es_spr[k] = spr_a

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
        'errorf_es': errorf_es,
        'errora_es': errora_es,
        'errorf_es_acc': errorf_es_acc,
        'errorf_es_spr': errorf_es_spr,
        'errora_es_acc': errora_es_acc,
        'errora_es_spr': errora_es_spr,
        'xt_traj': xt_traj,
        'xf_traj': xf_traj,
        'xa_traj': xa_traj,
        'Xf_all': Xf_all,               # [T, Nx, Ne]
        'Xa_all': Xa_all,                # [T, Nx, Ne]
        'ens_fcst_traj': ens_fcst_traj,  # [T, Ne, M+1, Nx]
        'truth_fcst_traj': truth_fcst_traj,  # [T, M+1, Nx]
        'diverged': diverged,
    }
