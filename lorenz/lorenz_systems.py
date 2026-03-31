import numpy as np
from numbalsoda import solve_ivp as nb_solve_ivp, lsoda as nb_lsoda
from DAPyr.MODELS import make_rhs_l63, make_rhs_l96, make_rhs_l05, model as dapyr_model

# Module-level cache: (system_type, frozenset_of_params) -> (rhs_obj, funcptr_address)
# Keeping the rhs object alive prevents the cfunc from being garbage-collected,
# which would invalidate the funcptr address.
_rhs_cache = {}


class LorenzSystems:
    """
    Class to generate trajectories for Lorenz systems (63, 96, 05).

    Integration uses DAPyr's RK45/LSODA solver (via numbalsoda).

    State-dimension constraints imposed by DAPyr's compiled kernels:
        L96: N must be 40
        L05: N must be 480
    """

    @staticmethod
    def _get_funcptr(system_type, **params):
        """
        Returns a cached DAPyr function pointer for the given system and parameters.

        DAPyr's make_rhs_* functions trigger Numba JIT compilation on first call,
        so results are cached by (system_type, params) to avoid recompiling on
        every trajectory generation.

        Parameter mapping from the public API to DAPyr:
            L63:  sigma -> s,  rho -> r,  beta -> b
            L96:  F -> F
            L05:  F -> l05_F / l05_Fe,  K -> l05_K
        """
        cache_key = (system_type, frozenset(params.items()))
        if cache_key not in _rhs_cache:
            if system_type == '63':
                rhs = make_rhs_l63({
                    's': params.get('sigma', 10.0),
                    'r': params.get('rho', 28.0),
                    'b': params.get('beta', 8/3),
                })
            elif system_type == '96':
                rhs = make_rhs_l96({
                    'F': params.get('F', 8.0),
                })
            elif system_type == '05':
                rhs = make_rhs_l05({
                    'l05_K':  params.get('K', 32),
                    'l05_I':  params.get('l05_I', 12),
                    'l05_b':  params.get('l05_b', 10.0),
                    'l05_c':  params.get('l05_c', 0.6),
                    'l05_F':  params.get('F', 15.0),
                    'l05_Fe': params.get('F', 15.0),
                })
            else:
                raise ValueError(f"Unknown Lorenz system type: {system_type}")
            _rhs_cache[cache_key] = (rhs, rhs.address)
        return _rhs_cache[cache_key][1]

    @staticmethod
    def _validate_state(system_type, x):
        if system_type == '96' and len(x) != 40:
            raise ValueError(
                f"DAPyr's L96 kernel requires N=40, got N={len(x)}."
            )
        if system_type == '05' and len(x) != 480:
            raise ValueError(
                f"DAPyr's L05 kernel requires N=480, got N={len(x)}."
            )

    @classmethod
    def generate_trajectory(cls, system_type, x0, dt, n_steps, **params):
        """
        Generates a trajectory using DAPyr's RK45/LSODA integrator (per-step loop).

        This is the legacy method kept for compatibility.
        Prefer generate_trajectory_fast() for better performance.
        """
        x = np.array(x0, dtype=float)
        cls._validate_state(system_type, x)

        funcptr = cls._get_funcptr(system_type, **params)
        nx = len(x)
        trajectory = np.empty((n_steps, nx))
        trajectory[0] = x.copy()

        for i in range(1, n_steps):
            x, error = dapyr_model(x, dt, 1, funcptr)
            if error:
                raise RuntimeError(
                    f"DAPyr integration failed at step {i} for system '{system_type}'."
                )
            trajectory[i] = x

        return trajectory

    @classmethod
    def generate_trajectory_fast(cls, system_type, x0, dt, n_steps,
                                 t_eval=None, **params):
        """
        Generates a trajectory via a single numbalsoda solve_ivp call.

        Instead of looping in Python and calling the solver once per step,
        this passes the full t_eval array to the solver, eliminating per-step
        Python overhead (deepcopy, array allocation, solver setup).

        Parameters
        ----------
        system_type : str
            '63', '96', or '05'
        x0 : array-like
            Initial state.
        dt : float
            Integration time step.
        n_steps : int
            Number of output steps (including the initial state).
        t_eval : ndarray, optional
            Explicit output times.  When None, uses
            ``np.linspace(0, dt*(n_steps-1), n_steps)``.
        **params
            System parameters forwarded to the RHS.
        """
        x = np.ascontiguousarray(x0, dtype=np.float64)
        cls._validate_state(system_type, x)

        funcptr = cls._get_funcptr(system_type, **params)

        if t_eval is None:
            t_eval = np.linspace(0.0, dt * (n_steps - 1), n_steps)
        t_span = np.array([t_eval[0], t_eval[-1]])

        sol = nb_solve_ivp(funcptr, t_span, x.copy(), t_eval,
                           rtol=1e-9, atol=1e-30)

        if sol.success and not np.allclose(sol.y[-1], 0.0):
            return sol.y

        # Fallback to LSODA for stiff regions
        trajectory, success = nb_lsoda(funcptr, x.copy(), t_eval,
                                       rtol=1e-9, atol=1e-30)
        if not success or np.allclose(trajectory[-1], 0.0):
            raise RuntimeError(
                f"Integration failed for system '{system_type}' "
                f"(both DOP853 and LSODA)."
            )
        return trajectory

    @staticmethod
    def warmup(blocking=False):
        """Pre-compile DAPyr JIT kernels for all supported systems in background threads.

        Parameters
        ----------
        blocking : bool
            If True, wait for all compilations to finish before returning.

        Returns
        -------
        list[threading.Thread]
            The warmup threads (useful for joining later).
        """
        import threading

        def _compile(sys_type, x0, params):
            try:
                LorenzSystems.generate_trajectory(sys_type, x0, 0.01, 2, **params)
            except Exception:
                pass

        jobs = [
            threading.Thread(target=_compile, args=('63', np.ones(3), {}), daemon=True),
            threading.Thread(target=_compile, args=('96', np.ones(40) * 8.0, {'F': 8.0}), daemon=True),
            threading.Thread(target=_compile, args=('05', np.ones(480) * 15.0, {'F': 15.0, 'K': 32}), daemon=True),
        ]
        for j in jobs:
            j.start()
        if blocking:
            for j in jobs:
                j.join()
        return jobs
