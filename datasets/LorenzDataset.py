import os
import json
import hashlib

import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from lorenz.lorenz_systems import LorenzSystems

# Fixed state dimensions enforced by DAPyr compiled kernels.
# L96 kernel hardcodes Nx=40; L05 kernel hardcodes Nx=480.
_SYSTEM_DIMS = {'63': 3, '96': 40, '05': 480}

# Approximate attractor equilibrium used to seed random ICs.
# Starting near the forcing equilibrium reduces spin-up time.
_SYSTEM_IC_CENTER = {'63': 0.0, '96': 8.0, '05': 15.0}

class LorenzDataset(Dataset):
    # Number of future target steps returned by __getitem__.
    # Must stay in sync with Training.py (recursive_rollout target length).
    _MAX_FUTURE = 10

    def __init__(self, system_type='63', x0=None, dt=0.01, Ns=5000, save_Dt=10,
                 std=0.000, prev_time_steps=1, num_start_locations=1, ds_noise=False,
                 cache_dir=None, cache_enabled=True, **system_params):

        self.data_list = []
        self.target_list = []

        if system_type not in _SYSTEM_DIMS:
            raise ValueError(
                f"Unknown system_type '{system_type}'. "
                f"Expected one of {list(_SYSTEM_DIMS.keys())}."
            )
        nx = _SYSTEM_DIMS[system_type]

        def _generate_single_trajectory():
            if x0 is None or num_start_locations > 1:
                if system_type == '63':
                    current_x0 = np.random.normal(0, 10, nx)
                else:
                    center = _SYSTEM_IC_CENTER[system_type]
                    current_x0 = np.ones(nx) * center + np.random.normal(0, std, nx)
            else:
                current_x0 = np.asarray(x0, dtype=float)
                if len(current_x0) != nx:
                    raise ValueError(
                        f"Provided x0 has length {len(current_x0)}, "
                        f"but system '{system_type}' requires N={nx}."
                    )

            nt_truth = Ns * save_Dt + save_Dt

            print(f"Generating {system_type} trajectory...")
            x_truth = LorenzSystems.generate_trajectory_fast(
                system_type, current_x0, dt, nt_truth, **system_params
            )

            if ds_noise:
                x_perturbed = x_truth + np.random.normal(
                    loc=0, scale=std, size=x_truth.shape
                )
                d = x_perturbed[: nt_truth - save_Dt : save_Dt]
                t = x_perturbed[save_Dt:nt_truth:save_Dt]
            else:
                d = x_truth[: nt_truth - save_Dt : save_Dt]
                t = x_truth[save_Dt:nt_truth:save_Dt]

            return d, t

        # Build a deterministic cache key based on the base configuration
        if cache_dir is not None and cache_enabled:
            os.makedirs(cache_dir, exist_ok=True)

            cfg = {
                "system_type": system_type,
                "dt": float(dt),
                "Ns": int(Ns),
                "save_Dt": int(save_Dt),
                "std": float(std),
                "ds_noise": bool(ds_noise),
                "system_params": system_params or {},
            }
            if x0 is not None:
                cfg["x0"] = np.asarray(x0, dtype=float).tolist()

            key_str = json.dumps(cfg, sort_keys=True)
            key_hash = hashlib.sha1(key_str.encode("utf-8")).hexdigest()[:10]
            prefix = f"L{system_type}_{key_hash}_traj_"

            # Discover existing per-trajectory cache files
            traj_files = []
            for fname in os.listdir(cache_dir):
                if fname.startswith(prefix) and fname.endswith(".npz"):
                    idx_str = fname[len(prefix) : -4]
                    try:
                        idx = int(idx_str)
                    except ValueError:
                        continue
                    traj_files.append((idx, fname))

            traj_files.sort(key=lambda x: x[0])
            N_req = num_start_locations
            N_cached = len(traj_files)

            # Load as many cached trajectories as needed
            for _, fname in traj_files[: min(N_cached, N_req)]:
                arr = np.load(os.path.join(cache_dir, fname))
                d = arr["d"]
                t = arr["t"]
                self.data_list.append(d)
                self.target_list.append(t)

            next_index = traj_files[-1][0] if traj_files else 0
            n_new = max(0, N_req - N_cached)

            # Generate and cache any missing trajectories
            for k in range(n_new):
                d, t = _generate_single_trajectory()
                self.data_list.append(d)
                self.target_list.append(t)

                idx = next_index + k + 1
                fname = f"{prefix}{idx:04d}.npz"
                np.savez_compressed(
                    os.path.join(cache_dir, fname),
                    d=d,
                    t=t,
                )
        else:
            # No caching: generate all trajectories on the fly
            for _ in range(num_start_locations):
                d, t = _generate_single_trajectory()
                self.data_list.append(d)
                self.target_list.append(t)

        self.data = np.vstack(self.data_list)
        self.target = np.vstack(self.target_list)
        
        self.scaler = StandardScaler()
        self.prev_time_steps = prev_time_steps
        self.scaler.fit(self.data)
        
        self.data = self.transform(self.data)
        self.target = self.transform(self.target)

        # Precompute valid sample indices.
        #
        # After vstack, all locations are concatenated into one flat array.
        # A sample at position `i` uses:
        #   input  : self.data[i - prev_time_steps : i]          (history window)
        #   targets: self.target[i - 1 : i - 1 + _MAX_FUTURE]   (future window)
        #
        # Both windows must lie entirely within one trajectory segment
        # [k*Ns, (k+1)*Ns) to avoid mixing states from different ICs.
        #   lower bound: i >= k*Ns + prev_time_steps
        #   upper bound: i <= (k+1)*Ns - _MAX_FUTURE + 1
        self._valid_indices = []
        for k in range(num_start_locations):
            seg_start = k * Ns
            seg_end   = (k + 1) * Ns
            lo = seg_start + prev_time_steps
            hi = seg_end - self._MAX_FUTURE + 2   # +2 because range() is exclusive
            self._valid_indices.extend(range(lo, hi))

    def __len__(self):
        return len(self._valid_indices)

    def __getitem__(self, idx):
        i = self._valid_indices[idx]
        input_seq = torch.tensor(
            self.data[i - self.prev_time_steps : i].flatten(), dtype=torch.float32
        )
        targets = torch.tensor(
            self.target[i - 1 : i - 1 + self._MAX_FUTURE], dtype=torch.float32
        )
        return input_seq, targets

    def inverse_transform(self, x):
        return self.scaler.inverse_transform(x)

    def transform(self, x):
        return self.scaler.transform(x)
