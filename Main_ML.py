import os
import torch
import torch.nn as nn
import yaml
import numpy as np
from torch.utils.data import DataLoader, random_split
from datasets.LorenzDataset import LorenzDataset
from MachineLearning import DenseNN, ResDenseNN, save_model
from Training import train_model, EarlyStopping

def load_config(config_path='config.yml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def cleanup_checkpoints(paths_config):
    """Removes old checkpoint files and tensorboard logs."""
    import glob
    import shutil
    for folder_key in ['outputs', 'models', 'runs']:
        folder = paths_config.get(folder_key)
        if not folder:
            continue
            
        if folder == 'runs' and os.path.exists(folder):
            shutil.rmtree(folder)
            os.makedirs(folder)
            continue
            
        if os.path.exists(folder):
            files = glob.glob(os.path.join(folder, '*.pth'))
            for f in files:
                try:
                    os.remove(f)
                except OSError as e:
                    print(f"Error removing {f}: {e.strerror}")

def run_training(config):
    # Parameters
    dataset_cfg = config['dataset']
    model_cfg = config['model']
    train_cfg = config['training']
    paths_cfg = config['paths']
    
    # Directories
    os.makedirs(paths_cfg['outputs'], exist_ok=True)
    os.makedirs(paths_cfg['models'], exist_ok=True)
    os.makedirs(paths_cfg['runs'], exist_ok=True)

    # Cleanup before starting new run
    cleanup_checkpoints(paths_cfg)

    # Create dataset
    dataset = LorenzDataset(
        system_type=dataset_cfg['system_type'],
        dt=dataset_cfg['dt'],
        Ns=dataset_cfg['ns'],
        save_Dt=dataset_cfg['save_dt'],
        std=dataset_cfg['std'],
        prev_time_steps=dataset_cfg['prev_time_steps']
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'], shuffle=False)
    
    # Get normalization constants
    train_mean = dataset.scaler.mean_
    train_std = dataset.scaler.scale_

    # Model Class selection
    if model_cfg['type'] == 'ResDenseNN':
        model_class = ResDenseNN
    else:
        model_class = DenseNN

    for trial in range(1, train_cfg['n_trials'] + 1):
        print(f"\n--- Trial {trial}/{train_cfg['n_trials']} ---")
        
        # Initialize model
        # Input size is determined by the system type (e.g., L63 has 3, L96 has whatever x0 has)
        # We can peek at the data to get input_size
        sample_input, _ = dataset[0]
        input_size_total = sample_input.shape[0]
        input_size_single = input_size_total // dataset_cfg['prev_time_steps']
        
        activation_dict = {'ReLU': nn.ReLU, 'Tanh': nn.Tanh, 'Sigmoid': nn.Sigmoid}
        hidden_act = activation_dict.get(model_cfg['hidden_activation'], nn.ReLU)
        
        model = model_class(
            input_size=input_size_single,
            prev_time_steps=dataset_cfg['prev_time_steps'],
            output_size=input_size_single,
            hidden_layers=model_cfg['hidden_layers'],
            hidden_activation=hidden_act,
            output_activation=None # Hardcoded for now as per legacy
        )
        
        model_name = f"{model_cfg['type']}_{dataset_cfg['system_type']}_trial{trial}"
        
        criterion = nn.MSELoss() # Defaulting to MSELoss for now
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['learning_rate'])
        early_stopping = EarlyStopping(patience=train_cfg['early_stopping_patience'])
        
        # Train model
        trained_model = train_model(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=train_cfg['num_epochs'],
            early_stopping=early_stopping
        )
        
        # Save model
        save_path = os.path.join(paths_cfg['outputs'], model_name)
        save_model(trained_model, save_path, train_mean, train_std)
        print(f"Saved model to {save_path}.pth")

if __name__ == "__main__":
    config = load_config()
    run_training(config)
