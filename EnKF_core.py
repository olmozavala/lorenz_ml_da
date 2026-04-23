import numpy as np
from scipy.linalg import sqrtm
from lorenz.lorenz_systems import LorenzSystems
from SurrogateModel import SurrogateModel

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
