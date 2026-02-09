# Developer Summary

Technical reference for contributors and maintainers of the Lorenz ML-DA codebase.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Entry Points                              │
│  0_LorenzDashboard.py   1_SingleMLTraining.py   Main_ML.py  │
│  (Lorenz Explorer)      (Train + Eval GUI)      (CLI Train) │
└──────────┬───────────────────┬───────────────────┬──────────┘
           │                   │                   │
           ▼                   ▼                   ▼
┌──────────────────┐  ┌────────────────┐  ┌───────────────────┐
│  lorenz/         │  │ MachineLearning│  │ Training.py       │
│  lorenz_systems  │  │ .py            │  │                   │
│  • L63 dynamics  │  │ • DenseNN      │  │ • train_model()   │
│  • L96 dynamics  │  │ • ResDenseNN   │  │ • EarlyStopping   │
│  • trajectory    │  │ • LSTMNN       │  │ • recursive_      │
│    generation    │  │ • save/load    │  │   rollout()       │
└──────────────────┘  └────────────────┘  └───────────────────┘
                              │
                              ▼
                      ┌───────────────┐
                      │ datasets/     │
                      │ LorenzDataset │
                      │ .py           │
                      │ • PyTorch     │
                      │   Dataset     │
                      │ • Scaler      │
                      │ • Multi-loc   │
                      └───────────────┘
```

---

## Module Details

### `lorenz/lorenz_systems.py`

Core physics engine. Contains `LorenzSystems` with static methods for L63 and L96 dynamics.

| Method | Description |
|---|---|
| `lorenz63(x, sigma, beta, rho)` | Returns `dx/dt` for the Lorenz-63 system |
| `lorenz96(x, F)` | Returns `dx/dt` for the Lorenz-96 system (N-dimensional) |
| `get_system(system_type)` | Factory method returning the appropriate dynamics function |
| `generate_trajectory(system_type, x0, dt, n_steps, **params)` | Forward Euler integration producing `(n_steps, nx)` array |

**Integration method**: Forward Euler (`x += dt * f(x)`). Simple but sufficient for the Lorenz attractor at small `dt`.

**Key parameter**: `**params` are passed directly to the dynamics function. For L63: `sigma`, `beta`, `rho`. For L96: `F`. The caller must ensure only relevant params are passed (e.g., don't pass `N` to L63).

---

### `datasets/LorenzDataset.py`

PyTorch `Dataset` that generates and normalizes Lorenz trajectory data.

**Constructor flow**:
1. For each of `num_start_locations` random initial conditions:
   - Generate a long trajectory using `LorenzSystems.generate_trajectory()`
   - Add Gaussian noise (`std` parameter)
   - Subsample at interval `save_Dt` to create input/target pairs offset by one step
2. Stack all locations' data into `self.data` and `self.target`
3. Fit a `StandardScaler` on `self.data`
4. Transform both `data` and `target` to zero-mean, unit-variance

**Important attributes**:
- `data_list` — Raw (pre-normalization) subsampled trajectories per location. Used for visualization.
- `scaler` — The fitted `StandardScaler`. Its `mean_` and `scale_` must be saved with the model.
- `prev_time_steps` — Number of consecutive states forming a single input sample.

**`__getitem__` output**:
- `input_seq`: Flattened tensor of shape `(prev_time_steps * nx,)` — the history window
- `targets`: Tensor of shape `(max_future=10, nx)` — the next 10 states for multi-step rollout training

---

### `MachineLearning.py`

Neural network architectures and model I/O utilities.

#### `DenseNN(nn.Module)`
Standard feedforward network. Input: flattened history `(batch, prev_time_steps * nx)` → Output: next state `(batch, nx)`.

#### `ResDenseNN(nn.Module)`
**Residual variant** — the network predicts the **increment** (delta) from the last input state:
```python
def forward(self, x):
    current_state = x[:, -self.input_size:]  # last state in history
    delta = self.network(x)
    return current_state + delta
```
This is generally more effective because the model only needs to learn the small correction rather than the full state magnitude.

#### `LSTMNN(nn.Module)`
Reshapes the flattened input into `(batch, prev_time_steps, nx)`, processes through LSTM layers, and maps the final hidden state to the output via a linear layer.

#### Model I/O
- **`save_model()`** saves a checkpoint dict: `{model_state_dict, train_mean, train_std, architecture}`
- **`load_model()`** reconstructs the model from a checkpoint
- **`architecture` dict** contains: `model_type`, `input_size`, `prev_time_steps`, `hidden_layers`, `system`, `N`

---

### `Training.py`

Training loop with progressive multi-step rollout.

#### `recursive_rollout(model, initial_input, num_steps, prev_time_steps, device)`
Autoregressively applies the model for `num_steps`:
1. Takes the current history window `[s_{t-n}, ..., s_{t-1}]`
2. Predicts `s_t`
3. Shifts the window: `[s_{t-n+1}, ..., s_{t-1}, s_t]`
4. Repeats

Returns `(batch, num_steps, nx)` tensor of all predictions.

#### `train_model(...)`
Main training loop with these key features:

**Progressive rollout schedule**:
| Epochs | Rollout Steps |
|---|---|
| 0–19 | 1 (single-step) |
| 20–59 | 2 |
| 60–119 | 3 |
| 120–199 | 4 |
| 200+ | 5 |

**Loss weighting**: Geometric decay `γ^s` (γ=0.9) across rollout steps — earlier predictions are weighted more heavily.

**Early stopping**: Resets when rollout depth increases. If patience is exhausted at a given depth, training advances to the next rollout phase.

**Best model saving**: `model.state_dict()` is saved as `{model_name}_best_model.pth` whenever validation loss improves. **Note**: This is a raw state_dict (no architecture metadata).

**Validation**: Always evaluated at 5-step rollout regardless of current training rollout depth, providing a consistent comparison metric.

---

### `0_LorenzDashboard.py`

Dash web application (port 5007) for interactive Lorenz system exploration.

**Features**:
- Dropdown to select L63 or L96 (toggles relevant parameter inputs)
- Physics parameter controls (σ, β, ρ for L63; F, N for L96)
- Adjustable `dt`, `steps`, ensemble size, and perturbation std
- 3D scatter plot + three 1D time series (x, y, z components)
- MathJax equation rendering

**Callbacks**:
- `toggle_ui(model)` — Shows/hides L63 vs L96 parameter divs
- `execute_sim(...)` — Generates truth + ensemble trajectories and plots

---

### `1_SingleMLTraining.py`

Dash web application (port 8050) for ML training and evaluation. Two tabs:

#### Training Tab
- Sidebar: Architecture config, data generation params, training hyperparams
- Start/Stop buttons with threaded training (doesn't block the UI)
- Real-time learning curves (train/val loss on log scale with `10^x` formatting)
- Sample trajectory visualization (3D plot of one training trajectory)
- Model configuration card showing selected model's details

**Training thread flow**:
1. `start_tra()` callback resets `training_state`, starts `threading.Thread(target=training_thread)`
2. `training_thread()` creates dataset, model, optimizer, then calls `train_model()` with a `progress_callback`
3. `progress_callback` appends loss values to `training_state['history']` each epoch
4. `update_metrics()` callback fires every 1s via `dcc.Interval`, reads `training_state`, updates plots

**Key global**: `training_state` dict holds all shared state between the training thread and Dash callbacks.

#### Evaluation Tab
- Model dropdown (auto-refreshes, shows both regular and `_best_model.pth` files)
- Forecast steps, ensemble size, IC noise (σ) controls
- Clicking "Run Eval" generates:
  1. **Truth ensemble**: `ens_size` trajectories from the true Lorenz system with perturbed ICs
  2. **ML ensemble**: Uses the first `prev_steps` of each truth trajectory as shared history, then rolls out the ML model
- Both ensembles share the same initial `prev_steps` states — they diverge only after the ML takes over
- 3D plot + three 1D component plots

**Model loading logic** (handles two checkpoint formats):
- **Full checkpoint** (`model_state_dict` key present): architecture, mean, std all in the `.pth`
- **Best model** (raw state_dict): Architecture and mean/std read from the companion `.yml` file

---

### `Main_ML.py`

Command-line training script. Reads `config.yml`, creates dataset, trains for `n_trials`, saves models to `outputs/`.

---

### `ML_Model_Comparison.py`

Batch evaluation and visualization script. Iterates over `prev_time_steps` ranges, loads trained models, evaluates at multiple lead times, and generates comparison plots (line plots + 3D trajectories).

---

## Data Flow

### Training Pipeline
```
Random ICs → LorenzSystems.generate_trajectory() → raw trajectory
  → add noise → subsample (save_Dt) → LorenzDataset
    → StandardScaler.fit_transform() → normalized (data, target) pairs
      → DataLoader → train_model() with recursive_rollout()
        → save checkpoint (.pth) + config (.yml)
```

### Evaluation Pipeline
```
Load checkpoint (.pth) + config (.yml)
  → Reconstruct model architecture from metadata
  → Generate truth ensemble (perturbed ICs through true Lorenz)
  → Extract first prev_steps from truth as shared ML history
  → Normalize history → recursive_rollout(model) → denormalize predictions
  → Plot truth vs ML ensemble (transparent blue vs red)
```

### Normalization
```
raw_state → (raw_state - train_mean) / train_std → normalized_input
                                                        ↓
                                                   model(normalized_input)
                                                        ↓
                                                   normalized_output
                                                        ↓
normalized_output * train_std + train_mean → predicted_raw_state
```

---

## File Formats

### Model Checkpoint (`.pth`)
**Full checkpoint** (saved by `save_model()`):
```python
{
    'model_state_dict': OrderedDict,   # PyTorch weights
    'train_mean': np.ndarray,          # Scaler mean (nx,)
    'train_std': np.ndarray,           # Scaler std (nx,)
    'architecture': {
        'model_type': str,             # 'Dense', 'ResDense', or 'LSTM'
        'input_size': int,             # State dimension (3 for L63)
        'prev_time_steps': int,        # History window size
        'hidden_layers': list[int],    # e.g. [64, 64, 32]
        'system': str,                 # '63' or '96'
        'N': int                       # L96 dimension (40 default)
    }
}
```

**Best model** (saved by `Training.py` during training):
```python
OrderedDict  # Raw state_dict only, no metadata
```

### Model Config (`.yml`)
Saved at training start (available even if training is stopped early):
```yaml
model_type: ResDense
system_type: '63'
dt: 0.001
save_dt: 10
prev_steps: 3
hidden_layers: 64,64,32
batch_size: 2048
patience: 20
loss_func: MSE
architecture:           # Same dict as in checkpoint
  model_type: ResDense
  input_size: 3
  # ...
train_mean: [...]       # Scaler mean as list
train_std: [...]        # Scaler std as list
```

---

## Testing

```bash
pytest tests/
```

Currently contains `test_lorenz_systems.py` with unit tests for the L63 and L96 trajectory generation.
