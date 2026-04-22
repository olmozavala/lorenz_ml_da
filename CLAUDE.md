# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML surrogate modeling framework for Lorenz dynamical systems (L63, L96, L05 Model III). The goal is to train neural networks as fast replacements for numerical ODE integrators in Ensemble Data Assimilation (DA) workflows. Trained surrogates propagate ensemble members through forecast cycles instead of the expensive numerical solver.

## Setup

DAPyr (not on PyPI) must be installed locally **before** installing other dependencies:

```bash
pip install DAPyr
pip install -r requirements.txt
```

For GPU training, `torch` in `requirements.txt` is CPU-only by default — replace with the appropriate CUDA wheel.

## Commands

```bash
# Interactive Lorenz system simulator
python 0_LorenzDashboard.py       # http://localhost:5007

# Interactive ML training studio (dataset gen, training, live loss plots)
python 1_SingleMLTraining.py      # http://localhost:8050

# Batch training driven by config.yml
python Main_ML.py

# Comparative evaluation of saved models
python ML_Model_Comparison.py

# EnKF DA benchmarks (surrogate vs. truth); outputs go to figures/
python D_EnKF.py

# TensorBoard (logs written to runs/)
tensorboard --logdir runs/

# Tests
pytest tests/
pytest tests/test_lorenz_systems.py::test_l63_shape   # single test
```

## Architecture

### Entry Points

| File | Role |
|------|------|
| `0_LorenzDashboard.py` | Dash app — Lorenz trajectory simulator |
| `1_SingleMLTraining.py` | Dash app — full interactive training UI |
| `Main_ML.py` | CLI — batch training from `config.yml` |
| `ML_Model_Comparison.py` | CLI — evaluate and compare saved models |
| `D_EnKF.py` | Script — full EnKF DA experiments with surrogate vs. truth |

### Core Modules

**`MachineLearning.py`** — Four model architectures, all sharing the same interface:
- Input: `(batch, prev_time_steps × state_dim)` — flattened history window (oldest-first)
- Output: `(batch, state_dim)` — predicted next state

| Class | Notes |
|-------|-------|
| `DenseNN` | Standard MLP |
| `ResDenseNN` | Residual MLP; predicts Δ, returns `current + Δ` — more stable training |
| `LSTMNN` | LSTM; reshapes input to `(batch, prev_time_steps, state_dim)` internally |
| `RNN` | Vanilla RNN with same reshaping as LSTM |

**`Training.py`** — Training loop and rollout logic:
- `train_model(model, model_name, train_loader, val_loader, criterion, optimizer, num_epochs, early_stopping, ...)` — gradient-descent loop; returns `(trained_model, history)`
- `recursive_rollout(model, initial_input, num_steps, prev_time_steps, device)` → `(batch, num_steps, nx)` — autoregressive rollout used during training
- Multi-step loss: `Σ γ^s · loss_s` with γ=0.9
- `EarlyStopping` — patience-based; resets patience when rollout depth increases

Progressive rollout schedule (critical for DA accuracy):

| Epoch range | Rollout steps |
|-------------|--------------|
| 0–19 | 1 |
| 20–59 | 2 |
| 60–119 | 3 |
| 120–199 | 4 |
| 200+ | 5 |

**`datasets/LorenzDataset.py`** — PyTorch Dataset:
- Generates trajectories via `LorenzSystems.generate_trajectory_fast()` (DAPyr + numbalsoda)
- Caches per-trajectory `.npz` files in `dataset_cache/` keyed by a hash of the full dataset config — any config change invalidates the cache
- Fits `StandardScaler` on training data; normalization stats saved with model checkpoint
- Sample: input = `data[i-prev_time_steps:i].flatten()`, target = next `_MAX_FUTURE=10` steps
- Valid index caching prevents samples from crossing trajectory boundaries (different ICs)

**`lorenz/lorenz_systems.py`** — DAPyr integration layer:
- `LorenzSystems.generate_trajectory_fast()` — preferred; single batch call to SUNDIALS via numbalsoda
- `LorenzSystems.generate_trajectory()` — legacy per-step loop (slower, avoid)
- Caches DAPyr RHS function pointers by `(system_type, params)` to avoid Numba recompilation

**`SurrogateModel.py`** — Inference wrapper:
- Loads any checkpoint, reconstructs architecture from metadata
- Handles normalization/denormalization internally; users always work in physical space
- `predict(history)` → next state `(N,)`; `rollout(history, n_steps)` → `(n_steps, N)`; calling the object (`surrogate(history)`) aliases `predict`
- Auto-discovers `.yml` sidecar next to `.pth` for full config reconstruction

**`D_EnKF.py`** — Ensemble Kalman Filter:
- `run_enkf(forecast_fn, xt_0, Xf_pool, config)` — runs full DA cycle; `forecast_fn` wraps either `SurrogateModel` or ground-truth Lorenz
- `make_surrogate_forecaster(surrogate, M)` / `make_lorenz_forecaster(dt, M)` — factory functions for `forecast_fn`
- Supports stochastic EnKF and deterministic ETKF, multiplicative inflation, Gaspari-Cohn localization
- Runs 5 benchmark sweeps: baseline, M, Ne, p (observed fraction), σ_obs

**`EnKFConfig`** key fields:
```python
T=30          # DA cycles
M=10          # forecast steps between assimilation
Ne=20         # ensemble size
dt=0.01
p=1.0         # fraction of observed state vars
sig_obs=1.1   # observation error std
enkf_type='stochastic'  # or 'deterministic' (ETKF)
use_localization=False; loc_radius=3.0
use_inflation=False;    infl_factor=1.02
```

**`plotting_helpers.py`** — Matplotlib/Plotly utilities for EnKF output (RMSE curves, rank histograms, spread-skill plots). Consumed by `D_EnKF.py`; not called from training code.

### Data Flow

```
config.yml
  └─▶ LorenzDataset
        ├─ DAPyr RK45/LSODA integrator (numbalsoda)
        ├─ Cache .npz per trajectory (hash-keyed)
        └─ StandardScaler normalization
  └─▶ DataLoader (train/val/test 70/20/10)
  └─▶ Model (DenseNN | ResDenseNN | LSTMNN | RNN)
        ├─ recursive_rollout with progressive depth
        └─ Multi-step loss (γ=0.9 decay)
  └─▶ Checkpoint .pth + sidecar .yml
  └─▶ SurrogateModel (inference)
  └─▶ EnKF assimilation cycles
```

### Model Checkpoint Format

`.pth` contains: `model_state_dict`, `train_mean`, `train_std`, `architecture` dict.  
`.yml` sidecar stores the full training config for reproducibility.  
Filenames: `{ModelType}_L{system}_trial{n}_{timestamp}.pth`.

## Configuration (`config.yml`)

Controls all batch training parameters. `1_SingleMLTraining.py` GUI exposes the same knobs.

```yaml
dataset:
  system_type: '05'          # '63' | '96' | '05'
  dt: 0.001                  # integration time step
  ns: 5000                   # samples per start location
  save_dt: 10                # subsampling factor (effective Δt = dt × save_dt)
  prev_time_steps: 4         # history window size
  num_start_locations: 50    # independent ICs
  system_params: {F: 15.0, K: 32, l05_c: 0.6, l05_b: 10.0}

model:
  type: 'RNN'                # DenseNN | ResDenseNN | LSTMNN | RNN
  hidden_layers: [1024, 1024, 512]
  hidden_activation: 'ReLU'

training:
  num_epochs: 10000
  batch_size: 2048
  learning_rate: 0.001
  early_stopping_patience: 20
  loss_func: 'MSE'           # 'MSE' | 'Huber'
  split_train: 70
  split_val: 20
  split_test: 10

paths:
  outputs: 'models'
  runs: 'runs'
  dataset_cache: 'dataset_cache'
```

## Key Constraints

- **Fixed state dimensions**: L96 is always N=40, L05 is always N=480 — hardcoded in DAPyr's Numba kernels. L63 is N=3.
- **Normalization**: Always normalize inputs with saved `train_mean`/`train_std`; denormalize predictions. `SurrogateModel` does this automatically — raw models do not.
- **Thread management**: `1_SingleMLTraining.py` sets `OMP/MKL/NUMBA_NUM_THREADS=1` to prevent oversubscription with its `ThreadPoolExecutor`.
- **Validation rollout**: Always 5-step regardless of training phase — do not change the validation logic when modifying the rollout schedule.
- **Cache invalidation**: The dataset cache key covers the full dataset config dict. Changing any dataset parameter (dt, ns, system_params, etc.) will trigger regeneration of all trajectory files.
