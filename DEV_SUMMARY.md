# Developer Summary

Technical reference for contributors and maintainers of the Lorenz ML-DA codebase.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         Entry Points                              │
│  0_LorenzDashboard.py   1_SingleMLTraining.py   Main_ML.py       │
│  (Lorenz Explorer)      (Train + Eval GUI)      (CLI Train)       │
└──────────┬───────────────────┬───────────────────┬───────────────┘
           │                   │                   │
           ▼                   ▼                   ▼
┌──────────────────┐  ┌────────────────┐  ┌───────────────────┐
│  lorenz/         │  │ MachineLearning│  │ Training.py       │
│  lorenz_systems  │  │ .py            │  │                   │
│  • DAPyr RK45/   │  │ • DenseNN      │  │ • train_model()   │
│    LSODA backend │  │ • ResDenseNN   │  │ • EarlyStopping   │
│  • L63 / L96 /   │  │ • LSTMNN       │  │ • recursive_      │
│    L05 Model III │  │ • save/load    │  │   rollout()       │
│  • funcptr cache │  └────────────────┘  └───────────────────┘
└──────────────────┘           │
           │                   ▼
           │           ┌───────────────┐
           └──────────▶│ datasets/     │
                       │ LorenzDataset │
                       │ • PyTorch     │
                       │   Dataset     │
                       │ • StandardScaler
                       │ • Multi-loc   │
                       └───────────────┘
```

---

## Module Details

### `lorenz/lorenz_systems.py`

Integration engine for all Lorenz systems, backed by **DAPyr** (RK45 via `numbalsoda`/ARKODE, with LSODA fallback).

The previous Forward Euler loop (`x += dt * f(x)`) and all Python RHS methods (`lorenz63`, `lorenz96`, `lorenz05`, `get_system`) have been removed. Integration now delegates entirely to DAPyr's compiled Numba kernels.

#### Module-level `_rhs_cache`

```python
_rhs_cache: dict  # (system_type, frozenset(params)) → (rhs_obj, funcptr_address)
```

DAPyr's `make_rhs_*` functions trigger **Numba JIT compilation** on first call (a few seconds). The cache ensures each unique set of physics parameters is compiled exactly once. Holding a reference to `rhs_obj` keeps the C function pointer (`rhs.address`) alive and valid.

#### `LorenzSystems` class

| Method | Description |
|---|---|
| `_get_funcptr(system_type, **params)` | Returns a cached C function pointer for the DAPyr ODE integrator. Compiles on first call. |
| `generate_trajectory(system_type, x0, dt, n_steps, **params)` | Integrates the system for `n_steps` using `dapyr_model(x, dt, 1, funcptr)` at each step. Returns `(n_steps, nx)` array. |

#### Parameter name mapping (public API → DAPyr)

| System | Public param | DAPyr key |
|---|---|---|
| L63 | `sigma` | `s` |
| L63 | `rho` | `r` |
| L63 | `beta` | `b` |
| L96 | `F` | `F` |
| L05 | `F` | `l05_F`, `l05_Fe` |
| L05 | `K` | `l05_K` |

#### Fixed state dimensions (DAPyr constraint)

| System | N | Enforced by |
|---|---|---|
| L63 | 3 (from `x0`) | — |
| L96 | **40** | `make_rhs_l96` hardcodes `Nx=40` |
| L05 | **480** | `make_rhs_l05` hardcodes `Nx=480` |

A `ValueError` is raised in `generate_trajectory` if `len(x0)` does not match.

#### DAPyr integrator details

`dapyr_model(x, dt, 1, funcptr)` calls `numbalsoda.solve_ivp` (ARKODE RK45, `rtol=1e-9`, `atol=1e-30`) and falls back to `lsoda` for stiff regions. It returns `(final_state, error_flag)`. The function is called once per step; a `RuntimeError` is raised if `error_flag != 0`.

> **Note on L05 model variant**: DAPyr implements **Model III** from Lorenz (2005) — a two-scale model with large-scale (`X`) and small-scale (`Y = Z − X`) components, coupled via parameters `I` (smoothing half-width), `b` (amplitude ratio), and `c` (coupling). When `I=1` (default `l05_I=1`), it degenerates to single-scale behaviour (Model II). This differs from the original repo's simplified Model II implementation.

---

### `datasets/LorenzDataset.py`

PyTorch `Dataset` that generates and normalises Lorenz trajectory data.

**Constructor flow**:
1. For each of `num_start_locations` random initial conditions:
   - Generate a long trajectory via `LorenzSystems.generate_trajectory()` (DAPyr-backed)
   - Add Gaussian noise (`std` parameter)
   - Subsample at interval `save_Dt` to create input/target pairs offset by one step
2. Stack all locations' data into `self.data` and `self.target`
3. Fit a `StandardScaler` on `self.data`, transform both arrays to zero-mean / unit-variance

**State dimension for L96**: `LorenzDataset` reads `system_params.get('N', 40)` to size the random initial condition vector. Since DAPyr fixes L96 at N=40, callers must pass (or rely on the default of) `N=40`.

**Important attributes**:
- `data_list` — Raw (pre-normalisation) subsampled trajectories per location. Used for 3D preview in the training dashboard.
- `scaler` — Fitted `StandardScaler`. Its `mean_` and `scale_` are saved with every model checkpoint.
- `prev_time_steps` — Number of consecutive states forming one input sample.

**`__getitem__` output**:
- `input_seq`: flattened tensor of shape `(prev_time_steps * nx,)` — the history window
- `targets`: tensor of shape `(10, nx)` — the next 10 states for multi-step rollout training

---

### `MachineLearning.py`

Neural network architectures and model I/O utilities. Dead code removed: the old commented `DenseNN` block, the legacy `ml_step()` function (which assumed ensemble-shaped input), and the `realLoad()` function (hardcoded local path).

#### `DenseNN(nn.Module)`

Standard feedforward MLP.

```
Input: (batch, input_size * prev_time_steps)  →  hidden layers  →  Output: (batch, output_size)
```

#### `ResDenseNN(nn.Module)`

Residual feedforward network — predicts the **increment** Δ, not the full next state:

```python
def forward(self, x):
    current_state = x[:, -self.input_size:]  # most recent state in flattened history
    delta = self.network(x)
    return current_state + delta             # s_{t+1} = s_t + Δ
```

Preferred architecture: easier to optimise (network learns small corrections), more stable under long autoregressive rollouts.

#### `LSTMNN(nn.Module)`

```python
def forward(self, x):
    x = x.view(-1, self.prev_time_steps, self.input_size)  # unflatten history
    out, _ = self.lstm(x)
    return self.fc(out[:, -1, :])  # map last hidden state → next state
```

`hidden_size` is a single integer (set to `hidden_layers[0]` when instantiating from the dashboard or CLI). `num_layers` defaults to 1.

#### Model I/O

| Function | Description |
|---|---|
| `save_model(model, path, mean, std, architecture)` | Saves full checkpoint: `{model_state_dict, train_mean, train_std, architecture}` |
| `load_model(path, model_class, ...)` | Reconstructs model from a full checkpoint |
| `denormalize(predictions, mean, std)` | `pred * std + mean` |

**`architecture` dict** (stored in checkpoint and `.yml`):
```python
{
    'model_type':      str,       # 'Dense', 'ResDense', or 'LSTM'
    'input_size':      int,       # state dimension (3 for L63, 40 for L96)
    'prev_time_steps': int,       # history window size
    'hidden_layers':   list[int], # e.g. [64, 64, 32]
    'system':          str,       # '63', '96', or '05'
    'N':               int,       # state dimension (same as input_size; legacy field)
}
```

---

### `Training.py`

Training loop with progressive multi-step rollout.

#### `recursive_rollout(model, initial_input, num_steps, prev_time_steps, device)`

Autoregressively applies the model for `num_steps`:
1. Current history window `[s_{t-n}, …, s_{t-1}]` → model → `s_t`
2. Shift window: `[s_{t-n+1}, …, s_{t-1}, s_t]`
3. Repeat

Returns `(batch, num_steps, nx)` tensor of all predictions.

#### `train_model(...)`

**Progressive rollout schedule**:

| Epochs | Rollout Steps |
|---|---|
| 0 – 19 | 1 (single-step) |
| 20 – 59 | 2 |
| 60 – 119 | 3 |
| 120 – 199 | 4 |
| 200+ | 5 |

**Loss weighting**: geometric decay `γ^s` (γ = 0.9) — earlier predictions weighted more heavily.

**Early stopping**: Patience-based; counter resets when rollout depth increases. Patience is read from `EarlyStopping.patience`.

**Best model saving**: `model.state_dict()` saved as `{model_name}_best_model.pth` whenever validation loss improves. This is a raw `state_dict` with no metadata — the companion `.yml` config file holds architecture and normalisation info.

**Validation**: Always at 5-step rollout regardless of training phase, giving a consistent metric across the whole run.

**Stop signal**: `progress_callback` return value is checked; returning `True` breaks the loop (used by the dashboard's Stop button).

> ⚠️ **Known bug**: The early-stopping phase-skip logic (`epoch = 19`, `epoch = 59`, etc.) inside a `for … range()` loop is a **silent no-op** in Python — assigning to the loop variable does not affect the iterator. The early-stopping *reset* (counter = 0) still fires correctly at natural epoch boundaries, so training is not broken — just potentially slower than intended when a phase converges early.

---

### `0_LorenzDashboard.py`

Dash web application (port 5007) for interactive Lorenz system exploration.

**Systems**: L63, L96 (N=40 fixed), L05 Model III (N=480 fixed). N inputs have been removed from the UI; dimensions are hardcoded in `run_simulation`.

**Equation display**: Updated to reflect the correct model variants, including "N=40 (fixed)" for L96 and the full two-scale Model III equation for L05.

**Key design — split callbacks with `dcc.Store`**:

The original single callback computed and rendered everything on each button click, so changing display variable indices required a full recompute. This has been split into two callbacks:

```
Run button ──→ [run_simulation] ──→ traj-store (serialised trajectories)
                                           │
                               ┌───────────┘
Variable index inputs ────────→ [update_plots] ──→ 3d-plot / x-plot / y-plot / z-plot
```

- **`run_simulation`** (triggered by Run button only): integrates trajectories, serialises them to `dcc.Store` as JSON (`true_traj`, `ens_trajs`, `time`, `model`, `N`), returns stats text.
- **`update_plots`** (triggered by `traj-store` update **or** any index input change): reads stored arrays, selects columns by index, rebuilds all four figures. No integration cost.

**Ensemble logic**: one unperturbed truth trajectory + `ens_size` independent trajectories from pertured initial conditions (`x_init + Normal(0, pert)`). This is correct and unchanged.

---

### `1_SingleMLTraining.py`

Dash web application (port 8050) for ML training and evaluation.

**Changes from original**:
- Removed `l96-n-div` / `l96-n` UI element and `toggle_n_l96` callback — N=40 is hardcoded
- Dropdown label updated to "Lorenz 96 (N=40)" to communicate the constraint
- `sys_params['N'] = 40` hardcoded in `training_thread_func` (used by `LorenzDataset` to size the L96 initial condition)
- `input_size = 40` hardcoded for L96 in model construction
- `eval_params = {'F': 8.0}` for L96 in the evaluation callback (previously passed `{'N': meta['N']}`, which `_get_funcptr` ignores)
- L96 evaluation initial condition: `np.ones(40) * 8.0 + small_noise` — starts near the L96 forcing equilibrium rather than `Normal(0, 1)` which is unphysical

#### Training thread flow

```
start_tra() callback
  → resets training_state
  → starts threading.Thread(target=training_thread_func)

training_thread_func(config)
  → LorenzDataset(...)                   # generates + normalises data
  → random_split(train / val / test)
  → builds model (DenseNN / ResDenseNN / LSTMNN)
  → saves .yml config immediately        # available even if training stops early
  → train_model(..., progress_callback)
  → save_model(...)                      # saves full checkpoint

progress_callback(epoch, train_loss, val_loss, rollout)
  → appends to training_state['history']
  → returns True if stop_requested      # breaks train_model loop
```

`update_metrics()` fires every 1 s via `dcc.Interval`, reads `training_state`, updates loss graph and status text.

#### Evaluation flow

```
Load .pth checkpoint
  → if 'model_state_dict' in cp: full checkpoint (has arch + mean/std)
  → else: raw state_dict (_best_model.pth) → read arch + mean/std from .yml

Reconstruct model → model.eval()

Generate truth ensemble (ens_size trajectories, perturbed ICs, full Lorenz)
Generate ML ensemble:
  → share first prev_steps states with corresponding truth trajectory
  → normalize → recursive_rollout(model, …) → denormalize
  → prepend shared history → full ML trajectory

Plot truth (blue) vs ML (red) — 3D + 3 × 1D
```

---

### `Main_ML.py`

Command-line batch training. Reads `config.yml`, creates `LorenzDataset`, runs `n_trials` independent training runs, saves full checkpoints to `outputs/`.

---

### `ML_Model_Comparison.py`

Batch evaluation and visualisation. Iterates over trained models in `outputs/`, groups by architecture and `prev_time_steps`, evaluates at lead times [1, 4, 8, 16] steps, and generates RMSE comparison plots and 3D trajectory comparisons.

---

## Data Flow

### Training Pipeline

```
Random ICs
  → LorenzSystems.generate_trajectory()  [DAPyr RK45/LSODA]
  → add noise → subsample (save_Dt)
  → LorenzDataset: StandardScaler.fit_transform()
  → DataLoader (train / val / test split)
  → train_model() with recursive_rollout()
     • progressive rollout schedule (1→5 steps)
     • geometric loss weighting (γ=0.9)
     • early stopping with rollout-phase resets
  → save_model() → .pth (full checkpoint) + .yml (config + scaler stats)
```

### Evaluation Pipeline

```
Load .pth + .yml
  → reconstruct model from architecture metadata
  → generate truth ensemble (Lorenz, perturbed ICs)
  → extract first prev_steps as shared ML initial history
  → (history - mean) / std → recursive_rollout(model) → * std + mean
  → prepend shared history → full ML trajectory
  → plot truth vs ML ensemble
```

### Normalisation

```
raw_state  →  (raw_state - train_mean) / train_std  →  model input (normalised)
                                                              ↓
                                                         model forward
                                                              ↓
normalised output  →  output * train_std + train_mean  →  predicted raw state
```

`train_mean` and `train_std` come from `dataset.scaler.mean_` and `dataset.scaler.scale_` and are stored in both the `.pth` checkpoint and the `.yml` config.

---

## File Formats

### Model Checkpoint (`.pth`)

**Full checkpoint** (written by `save_model()`):
```python
{
    'model_state_dict': OrderedDict,   # PyTorch weights
    'train_mean':       np.ndarray,    # scaler mean  (nx,)
    'train_std':        np.ndarray,    # scaler std   (nx,)
    'architecture': {
        'model_type':      str,        # 'Dense', 'ResDense', or 'LSTM'
        'input_size':      int,        # state dimension (3 for L63, 40 for L96)
        'prev_time_steps': int,
        'hidden_layers':   list[int],
        'system':          str,        # '63', '96', or '05'
        'N':               int,        # same as input_size (legacy field)
    }
}
```

**Best model** (written by `Training.py` during training):
```python
OrderedDict  # raw state_dict only — no metadata
             # architecture + mean/std must be read from the companion .yml
```

### Model Config (`.yml`)

Saved at training start (available even if training is stopped early):
```yaml
model_type: ResDense
system_type: '63'
dt: 0.001
save_dt: 10
prev_steps: 3
hidden_layers: '64,64,32'
batch_size: 2048
patience: 20
loss_func: MSE
split_train: 70
split_val: 20
split_test: 10
architecture:
  model_type: ResDense
  input_size: 3
  prev_time_steps: 3
  hidden_layers: [64, 64, 32]
  system: '63'
  N: 3
train_mean: [0.123, ...]    # list, length nx
train_std:  [4.567, ...]    # list, length nx
```

---

## Dependencies

All dependencies are pinned in `requirements.txt` to match the `lorenzo` conda environment:

| Package | Version | Role |
|---|---|---|
| `numpy` | 2.3.5 | Array operations throughout |
| `torch` | 2.10.0 | Neural networks, DataLoader |
| `tensorboard` | — | TensorBoard logging in `Training.py` |
| `scikit-learn` | 1.8.0 | `StandardScaler` in `LorenzDataset` |
| `tqdm` | 4.67.3 | Progress bars |
| `PyYAML` | 6.0.3 | Config serialisation |
| `numba` | 0.64.0 | JIT compilation of Lorenz RHS (DAPyr) |
| `numbalsoda` | 0.3.4 | RK45 / LSODA ODE solver wrappers (DAPyr) |
| `DAPyr` | 1.0.0 | Lorenz model kernels + DA methods |
| `matplotlib` | — | Plots in `ML_Model_Comparison.py` |
| `dash` | 4.0.0 | Web dashboards |
| `dash-bootstrap-components` | 2.0.4 | Dashboard styling |
| `plotly` | 6.6.0 | Interactive figures |
| `pytest` | — | Test suite |

> **DAPyr is not on PyPI** — install from local source: `pip install <path-to-DAPyr>`

---

## Testing

```bash
conda activate lorenzo
pytest tests/
```

`tests/test_lorenz_systems.py` covers the DAPyr-backed integration layer:

| Test | What it checks |
|---|---|
| `test_generate_trajectory_l63` | L63 trajectory shape `(n_steps, 3)` and correct initial condition |
| `test_generate_trajectory_invalid_system` | `ValueError` raised for unknown system type |
| `test_generate_trajectory_l96_wrong_dim` | `ValueError` raised when `len(x0) != 40` for L96 |

> Tests require the `lorenzo` conda environment (for the DAPyr import). Running in a plain environment without DAPyr will fail at import time.
