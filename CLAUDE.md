# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML surrogate modeling framework for Lorenz dynamical systems (L63, L96, L05 Model III). The goal is to train neural networks as fast replacements for numerical ODE integrators in Ensemble Data Assimilation (DA) workflows. Trained surrogates propagate ensemble members through forecast cycles instead of the expensive numerical solver.

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

# TensorBoard (logs written to runs/)
tensorboard --logdir runs/

# Tests
pytest tests/
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

**`MachineLearning.py`** — Five model architectures, all sharing the same interface:
- Input: `(batch, prev_time_steps × state_dim)` — flattened history window (oldest-first)
- Output: `(batch, state_dim)` — predicted next state

| Class | Notes |
|-------|-------|
| `DenseNN` | Standard MLP |
| `ResDenseNN` | Residual MLP; predicts Δ, returns `current + Δ` — more stable training |
| `LSTMNN` | LSTM; reshapes input to `(batch, prev_time_steps, state_dim)` |
| `RNN` | Vanilla RNN with same reshaping as LSTM |

**`Training.py`** — Training loop and rollout logic:
- `train_model()` — standard gradient-descent loop with early stopping
- `recursive_rollout()` — autoregressive multi-step rollout used during training; multi-step loss `Σ γ^s · loss_s` with γ=0.9
- `EarlyStopping` — patience-based; resets patience each time rollout depth increases

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
- Caches per-trajectory `.npz` files in `dataset_cache/` keyed by config hash
- Fits `StandardScaler` on training data; normalization stats saved with model checkpoint
- Sample: input = `data[i-prev_time_steps:i].flatten()`, target = next `_MAX_FUTURE=10` steps

**`lorenz/lorenz_systems.py`** — DAPyr integration layer:
- `LorenzSystems.generate_trajectory_fast()` — preferred; single batch call to SUNDIALS
- Caches DAPyr RHS function pointers by `(system_type, params)` to avoid Numba recompilation

**`SurrogateModel.py`** — Inference wrapper:
- Loads any checkpoint, reconstructs architecture from metadata
- Handles normalization/denormalization internally
- `predict(history)` → next state; `rollout(history, n_steps)` → trajectory

**`D_EnKF.py`** — Ensemble Kalman Filter:
- `run_enkf(config, propagator)` — runs full DA cycle; `propagator` is either `SurrogateModel` or ground-truth Lorenz
- Supports stochastic EnKF and deterministic ETKF, optional multiplicative inflation, Gaspari-Cohn localization
- `EnKFConfig` dataclass holds all DA parameters

### Data Flow

```
config.yml
  └─▶ LorenzDataset
        ├─ DAPyr RK45/LSODA integrator (numbalsoda)
        ├─ Cache .npz per trajectory
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
- **DAPyr**: Not on PyPI; must be installed locally before anything runs.
- **Normalization**: Always normalize inputs with saved `train_mean`/`train_std`; denormalize predictions. `SurrogateModel` does this automatically — raw models do not.
- **Thread management**: `1_SingleMLTraining.py` sets `OMP/MKL/NUMBA_NUM_THREADS=1` to prevent oversubscription with its `ThreadPoolExecutor`.
- **Validation rollout**: Always 5-step regardless of training phase — do not change the validation logic when modifying the rollout schedule.
