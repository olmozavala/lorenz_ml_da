# Lorenz ML-DA

Machine Learning surrogate models for Lorenz dynamical systems (L63, L96, and L05), with interactive dashboards for simulation, training, evaluation, and model comparison.

## Overview

This project trains neural networks to emulate the dynamics of chaotic Lorenz systems. The trained ML models are intended to serve as **fast surrogate replacements for the numerical model** inside **ensemble Data Assimilation (DA)** workflows — calling the neural network instead of the ODE integrator at each forecast step.

Numerical integration is provided by **[DAPyr](https://github.com/Dayton-DA/DAPyr)** (RK45 with LSODA fallback via `numbalsoda`/`sundials`), replacing the previous Forward Euler implementation.

### Supported Lorenz Systems

- **Lorenz 63** — Classic 3-variable chaotic attractor (N=3):

$$\frac{dx}{dt} = \sigma(y - x), \quad \frac{dy}{dt} = x(\rho - z) - y, \quad \frac{dz}{dt} = xy - \beta z$$

- **Lorenz 96** — N-dimensional weather-like model (**N=40, fixed**):

$$\frac{dx_i}{dt} = (x_{i+1} - x_{i-2})x_{i-1} - x_i + F$$

- **Lorenz 2005 Model III** — Two-scale model (**N=480, fixed**). The full state $Z_n$ is split into large-scale $X_n$ (smoothed with half-width $I$) and small-scale residual $Y_n = Z_n - X_n$:

$$\frac{dZ_n}{dt} = [X,X]_{K,n} + b^2[Y,Y]_n + c\,[Y,X]_n - X_n - bY_n + F$$

> **Note:** The state dimensions for L96 (N=40) and L05 (N=480) are fixed by DAPyr's compiled Numba kernels and cannot be changed without modifying DAPyr.

---

## Features

- 🌀 **Interactive Lorenz Simulator** — Explore L63/L96/L05 dynamics with adjustable physics parameters, ensemble perturbations, and 3D visualization. Display variables can be changed live without recomputing the trajectory.
- 🔥 **ML Training Dashboard** — Train Dense, Residual-Dense, or LSTM models with real-time learning curves and trajectory visualization.
- 🧪 **Model Evaluation** — Compare ML ensemble predictions against truth ensemble with shared initial history windows.
- 📊 **Batch Model Comparison** — Systematic evaluation across architectures and history-step configurations.

---

## Quick Start

### Prerequisites

This project requires **DAPyr** to be installed locally (it is not on PyPI):

```bash
pip install <path-to-DAPyr>
```

DAPyr in turn requires `numba` and `numbalsoda`. Both are included in `requirements.txt`.

### Installation

```bash
# Clone the repository
git clone https://github.com/olmozavala/lorenz_ml_da.git
cd lorenz_ml_da

# Install all dependencies (activate your environment first)
pip install -r requirements.txt
```

> **GPU / CUDA**: `torch` is listed without a CUDA suffix in `requirements.txt`. To use GPU acceleration, install the appropriate CUDA wheel separately:
> ```bash
> pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
> ```

### Running the Dashboards

**Lorenz Simulator** (port 5007):
```bash
python 0_LorenzDashboard.py
```
Explore Lorenz dynamics interactively. Adjust physics parameters (σ, β, ρ for L63; F for L96/L05; K for L05), run ensembles with perturbed initial conditions, and visualize trajectories in 3D. Change which state variables are shown in the 1D/3D plots without re-running the simulation.

**ML Training Studio** (port 8050):
```bash
python 1_SingleMLTraining.py
```
Train ML surrogate models through the browser. Configure architecture, data generation, and training hyperparameters in the sidebar. Monitor learning curves in real-time. Evaluate trained models against truth ensembles in the Evaluation tab.

### Command-Line Training

```bash
# Edit config.yml to set parameters, then:
python Main_ML.py
```

### Model Comparison

```bash
python ML_Model_Comparison.py
```

---

## Project Structure

```
lorenz_ml_da/
├── 0_LorenzDashboard.py      # Interactive Lorenz simulator (L63 / L96 / L05)
├── 1_SingleMLTraining.py     # ML training + evaluation dashboard
├── Main_ML.py                # Command-line batch training script
├── ML_Model_Comparison.py    # Batch model comparison & plotting
├── MachineLearning.py        # Neural network architectures (Dense, ResDense, LSTM)
├── Training.py               # Training loop, early stopping, recursive rollout
├── config.yml                # Configuration for command-line training
├── requirements.txt          # Python dependencies (pinned to lorenzo env versions)
├── lorenz/
│   └── lorenz_systems.py     # Lorenz 63/96/05 integration via DAPyr (RK45/LSODA)
├── datasets/
│   └── LorenzDataset.py      # PyTorch Dataset for Lorenz trajectory data
├── models/                   # Saved model checkpoints (.pth) and configs (.yml)
├── outputs/                  # Batch training outputs and comparison plots
├── runs/                     # TensorBoard training logs
└── tests/
    └── test_lorenz_systems.py # Unit tests for DAPyr-backed trajectory generation
```

---

## Model Architectures

| Architecture | Description |
|---|---|
| **DenseNN** | Standard feedforward network: flattened history → hidden layers → next state |
| **ResDenseNN** | Residual variant: predicts the *increment* Δ from the last state (`s_t + Δ`). Generally recommended — easier to optimise and more stable under long autoregressive rollouts. |
| **LSTMNN** | History is reshaped into `(batch, prev_time_steps, nx)` and processed by an LSTM; the last hidden state is mapped to the next state via a linear layer. |

All models take `prev_time_steps` consecutive states as input (flattened to `(batch, prev_time_steps × nx)`) and predict the next state `(batch, nx)`.

---

## Training Details

- **Data normalisation**: All data is `StandardScaler`-normalised (zero mean, unit variance). Scaler parameters (`mean_`, `scale_`) are saved with each model checkpoint for correct denormalisation at inference time.
- **Progressive multi-step rollout**: Training uses an escalating rollout schedule (1 → 2 → 3 → 4 → 5 steps) with geometric loss weighting (γ = 0.9). This encourages the model to remain accurate over longer horizons, which is critical for use inside a DA cycle.
- **Early stopping**: Patience-based, resets when the rollout depth increases so each phase gets a fair number of epochs.
- **Best model**: A raw `state_dict` is saved as `{model_name}_best_model.pth` whenever validation loss improves. The companion `.yml` config file holds architecture and normalisation metadata.

---

## Configuration

`config.yml` controls batch training via `Main_ML.py`. Every key maps 1-to-1 with the sidebar of `1_SingleMLTraining.py`.

```yaml
dataset:
  system_type: '63'          # '63' | '96' | '05'
  dt: 0.001                  # integration time step
  ns: 10000                  # samples per start location
  save_dt: 10                # subsampling factor  (effective Δt = dt × save_dt)
  std: 0.0                   # observation noise std (0 = clean)
  prev_time_steps: 3         # history window size fed to the model
  num_start_locations: 10    # number of independent random-IC trajectories

  # Physics parameters — leave empty {} for L63
  # L96:  {F: 8.0}    L05:  {F: 15.0, K: 32}
  system_params: {}

model:
  type: 'ResDenseNN'         # 'DenseNN' | 'ResDenseNN' | 'LSTMNN'
  hidden_layers: [64, 64, 32]
  hidden_activation: 'ReLU'  # 'ReLU' | 'Tanh' | 'Sigmoid'

training:
  num_epochs: 500
  batch_size: 2048
  learning_rate: 0.001
  n_trials: 1                # independent runs (each saved separately with timestamp)
  early_stopping_patience: 20
  loss_func: 'MSE'           # 'MSE' | 'Huber'
  split_train: 70
  split_val: 20
  split_test: 10

paths:
  outputs: 'outputs'         # full checkpoints + YAML configs saved here
  models: 'models'
  runs: 'runs'               # TensorBoard logs
```

Checkpoints written by `Main_ML.py` use the same full format as the dashboard (`model_state_dict` + `train_mean/std` + `architecture`), so they can be loaded directly in the **Evaluation** tab of `1_SingleMLTraining.py`.

---

## Research Context

The long-term goal of this project is to use the trained ML surrogates **inside an ensemble DA cycle** (e.g., EnSRF or Local Particle Filter as implemented in DAPyr), replacing the numerical Lorenz model with a neural network forward operator at each forecast step. The progressive rollout training strategy directly targets this use case by teaching the model to produce stable trajectories under repeated autoregressive application.

The three systems span a standard ladder of difficulty in the DA community: L63 (toy, easy to visualise) → L96 (canonical benchmark) → L05 Model III (multiscale, closer to realistic atmospheric dynamics).

---

## License

This project is part of research at COAPS/FSU.
