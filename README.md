# Lorenz ML-DA

Machine Learning surrogate models for Lorenz dynamical systems (L63 & L96), with interactive dashboards for simulation, training, evaluation, and model comparison.

## Overview

This project trains neural networks to emulate the dynamics of chaotic Lorenz systems. The trained ML models can then be used as surrogate models in **Data Assimilation (DA)** workflows, replacing expensive numerical integrations with fast neural network inference.

### Supported Lorenz Systems

- **Lorenz 63** — The classic 3-variable chaotic attractor:

$$\frac{dx}{dt} = \sigma(y - x), \quad \frac{dy}{dt} = x(\rho - z) - y, \quad \frac{dz}{dt} = xy - \beta z$$

- **Lorenz 96** — The N-dimensional weather-like model:

$$\frac{dx_i}{dt} = (x_{i+1} - x_{i-2})x_{i-1} - x_i + F$$

## Features

- 🌀 **Interactive Lorenz Simulator** — Explore L63/L96 dynamics with adjustable parameters, ensemble perturbations, and 3D visualization
- 🔥 **ML Training Dashboard** — Train Dense, Residual-Dense, or LSTM models with real-time learning curves and trajectory visualization
- 🧪 **Model Evaluation** — Compare ML predictions against truth with ensemble forecasts from perturbed initial conditions
- 📊 **Batch Model Comparison** — Systematic evaluation across architectures and history step configurations

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/olmozavala/lorenz_ml_da.git
cd lorenz_ml_da

# Install dependencies
pip install -r requirements.txt

# Additional dependencies for dashboards
pip install dash dash-bootstrap-components plotly scikit-learn pyyaml tensorboard
```

### Running the Dashboards

**Lorenz Simulator** (port 5007):
```bash
python 0_LorenzDashboard.py
```
Explore Lorenz dynamics interactively — adjust parameters (σ, β, ρ for L63; F, N for L96), run ensembles with perturbed initial conditions, and visualize trajectories in 3D.

**ML Training Studio** (port 8050):
```bash
python 1_SingleMLTraining.py
```
Train ML models through the browser. Configure architecture, data generation, and training hyperparameters in the sidebar. Monitor learning curves in real-time, then evaluate trained models against truth in the Evaluation tab.

### Command-Line Training

For batch training without the dashboard:

```bash
# Edit config.yml to set parameters, then:
python Main_ML.py
```

### Model Comparison

```bash
python ML_Model_Comparison.py
```

## Project Structure

```
lorenz_ml_da/
├── 0_LorenzDashboard.py      # Interactive Lorenz simulator dashboard
├── 1_SingleMLTraining.py      # ML training + evaluation dashboard
├── Main_ML.py                 # Command-line training script
├── ML_Model_Comparison.py     # Batch model comparison & plotting
├── MachineLearning.py         # Neural network architectures (Dense, ResDense, LSTM)
├── Training.py                # Training loop, early stopping, recursive rollout
├── config.yml                 # Configuration for command-line training
├── requirements.txt           # Python dependencies
├── lorenz/
│   └── lorenz_systems.py      # Lorenz 63 & 96 system implementations
├── datasets/
│   └── LorenzDataset.py       # PyTorch Dataset for Lorenz trajectory data
├── models/                    # Saved model checkpoints (.pth) and configs (.yml)
├── outputs/                   # Training outputs
├── runs/                      # TensorBoard logs
└── tests/
    └── test_lorenz_systems.py # Unit tests for Lorenz systems
```

## Model Architectures

| Architecture | Description |
|---|---|
| **DenseNN** | Standard feedforward network mapping flattened history → next state |
| **ResDenseNN** | Residual variant that predicts the *increment* (Δ) from the last input state |
| **LSTMNN** | LSTM-based model that processes the history as a temporal sequence |

All models take `prev_time_steps` history states as input (flattened) and predict the next state. The **ResDenseNN** is generally recommended as it learns the dynamics increment rather than the full state, which tends to train faster and generalize better.

## Training Details

- **Data normalization**: All data is StandardScaler-normalized (zero mean, unit variance). The scaler parameters are saved with the model for correct denormalization at inference.
- **Multi-step rollout**: Training uses a progressive rollout schedule (1→2→3→4→5 steps) with geometric loss weighting (γ=0.9), encouraging the model to remain accurate over longer horizons.
- **Early stopping**: Patience-based early stopping resets when the rollout depth increases, allowing the model to adapt to each new phase.

## Configuration

The `config.yml` file controls batch training via `Main_ML.py`:

```yaml
dataset:
  system_type: '63'        # '63' or '96'
  dt: 0.01                 # Integration time step
  ns: 5000                 # Number of integration steps per location
  save_dt: 10              # Subsampling factor (effective Δt = dt × save_dt)
  prev_time_steps: 4       # Number of history steps as model input

model:
  type: 'ResDenseNN'       # 'DenseNN' or 'ResDenseNN'
  hidden_layers: [64, 64, 32, 16]

training:
  num_epochs: 400
  batch_size: 64
  learning_rate: 0.001
  n_trials: 3
  early_stopping_patience: 10
```

## License

This project is part of research at COAPS/FSU.
