import os
import time
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split

from datasets.LorenzDataset import LorenzDataset, _SYSTEM_DIMS
from MachineLearning import DenseNN, ResDenseNN, LSTMNN, RNN, save_model
from Training import train_model, EarlyStopping

torch.set_default_dtype(torch.float64)


def load_config(config_path='config.yml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def cleanup_checkpoints(paths_config):
    """Removes .pth files from outputs/ and wipes the TensorBoard runs/ directory."""
    import glob
    import shutil
    for folder_key in ['outputs', 'models', 'runs']:
        folder = paths_config.get(folder_key)
        if not folder:
            continue
        if folder_key == 'runs' and os.path.exists(folder):
            shutil.rmtree(folder)
            os.makedirs(folder)
            continue
        if os.path.exists(folder):
            for f in glob.glob(os.path.join(folder, '*.pth')):
                try:
                    os.remove(f)
                except OSError as e:
                    print(f"Error removing {f}: {e.strerror}")


def run_training(config):
    dataset_cfg = config['dataset']
    model_cfg   = config['model']
    train_cfg   = config['training']
    paths_cfg   = config['paths']
    cleanup = config.get('cleanup', False)

    for path in paths_cfg.values():
        os.makedirs(path, exist_ok=True)

    if cleanup:
        cleanup_checkpoints(paths_cfg)

    sys_type = dataset_cfg['system_type']

    # --- Physics parameters (mirrors 1_SingleMLTraining.py training_thread_func) ---
    raw_sp = dataset_cfg.get('system_params', {}) or {}
    sys_params = {}
    if sys_type == '96':
        sys_params['F'] = float(raw_sp.get('F', 8.0))
    elif sys_type == '05':
        sys_params['F'] = float(raw_sp.get('F', 15.0))
        sys_params['K'] = int(raw_sp.get('K', 32))

    # --- Dataset ---
    print(f"Building LorenzDataset — system={sys_type}, "
          f"locs={dataset_cfg.get('num_start_locations', 1)}, "
          f"Ns={dataset_cfg['ns']}, save_Dt={dataset_cfg['save_dt']}")
    dataset = LorenzDataset(
        system_type=sys_type,
        dt=dataset_cfg['dt'],
        Ns=dataset_cfg['ns'],
        save_Dt=dataset_cfg['save_dt'],
        std=dataset_cfg.get('std', 0.0),
        prev_time_steps=dataset_cfg['prev_time_steps'],
        num_start_locations=dataset_cfg.get('num_start_locations', 1),
        ds_noise=dataset_cfg.get('ds_noise', False),
        cache_dir=paths_cfg.get('dataset_cache'),
        cache_enabled=True,
        **sys_params
    )

    # --- Train / val / test split ---
    split_train = train_cfg.get('split_train', 70)
    split_val   = train_cfg.get('split_val',   20)
    split_test  = train_cfg.get('split_test',  100 - split_train - split_val)
    total    = len(dataset)
    train_sz = int(total * split_train / 100)
    val_sz   = int(total * split_val   / 100)
    test_sz  = total - train_sz - val_sz
    train_set, val_set, _ = random_split(dataset, [train_sz, val_sz, test_sz])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  train={train_sz}  val={val_sz}  test={test_sz}")

    train_loader = DataLoader(
        train_set, batch_size=train_cfg['batch_size'],
        shuffle=True,  pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(
        val_set,   batch_size=train_cfg['batch_size'],
        shuffle=False, pin_memory=(device.type == 'cuda'))

    # --- Derived architecture constants ---
    input_size  = _SYSTEM_DIMS[sys_type]      # 3 / 40 / 480 — fixed by DAPyr
    prev_steps  = dataset_cfg['prev_time_steps']
    hidden_list = model_cfg['hidden_layers']
    model_type  = model_cfg['type']

    activation_map = {'ReLU': nn.ReLU, 'Tanh': nn.Tanh, 'Sigmoid': nn.Sigmoid}
    hidden_act = activation_map.get(model_cfg.get('hidden_activation', 'ReLU'), nn.ReLU)

    loss_name = train_cfg.get('loss_func', 'MSE')
    criterion = nn.MSELoss() if loss_name == 'MSE' else nn.HuberLoss()

    # Progressive rollout schedule + gradient clipping.
    # Defaults preserve the pre-refactor behaviour so older configs still work.
    rollout_schedule   = train_cfg.get('rollout_schedule',
                                       [[20, 1], [60, 2], [120, 3], [200, 4], [10000, 5]])
    val_rollout_steps  = train_cfg.get('val_rollout_steps', 5)
    grad_clip          = train_cfg.get('grad_clip', 1.0)

    # --- Trial loop ---
    for trial in range(1, train_cfg['n_trials'] + 1):
        print(f"\n{'='*60}")
        print(f"  Trial {trial} / {train_cfg['n_trials']}  —  {model_type}  L{sys_type}")
        print(f"{'='*60}")

        # Build model
        if model_type == 'DenseNN':
            model = DenseNN(input_size, prev_steps, input_size, hidden_list, hidden_act, None)
        elif model_type == 'ResDenseNN':
            model = ResDenseNN(input_size, prev_steps, input_size, hidden_list, hidden_act, None)
        elif model_type == 'LSTMNN':
            model = LSTMNN(input_size, prev_steps, input_size, hidden_list[0])
        elif model_type == 'RNN':
            rnn_nonlin = model_cfg.get('rnn_nonlinearity', 'tanh')
            model = RNN(input_size, prev_steps, input_size, hidden_list[0],
                        nonlinearity=rnn_nonlin)
        else:
            raise ValueError(
                f"Unknown model type '{model_type}'. "
                "Expected 'DenseNN', 'ResDenseNN', 'LSTMNN', or 'RNN'."
            )

        arch_meta = {
            'model_type':      model_type,
            'input_size':      input_size,
            'prev_time_steps': prev_steps,
            'hidden_layers':   hidden_list,
            'system':          sys_type,
            'N':               input_size,   # legacy field; same as input_size
            'system_params':   sys_params,   # stored for eval reconstruction
        }

        model_name = f"{model_type}_L{sys_type}_trial{trial}_{int(time.time())}"
        save_path  = os.path.join(paths_cfg['outputs'], model_name)

        # Save YAML config immediately — available even if training stops early
        run_config = {
            'model_type':      model_type,
            'system_type':     sys_type,
            'dt':              dataset_cfg['dt'],
            'save_dt':         dataset_cfg['save_dt'],
            'prev_steps':      prev_steps,
            'num_locs':        dataset_cfg.get('num_start_locations', 1),
            'samples_per_loc': dataset_cfg['ns'],
            'batch_size':      train_cfg['batch_size'],
            'patience':        train_cfg['early_stopping_patience'],
            'hidden_layers':   str(hidden_list),
            'rnn_nonlinearity': model_cfg.get('rnn_nonlinearity', None),
            'loss_func':       loss_name,
            'split_train':     split_train,
            'split_val':       split_val,
            'split_test':      split_test,
            'rollout_schedule':  rollout_schedule,
            'val_rollout_steps': val_rollout_steps,
            'grad_clip':         grad_clip,
            'architecture':    arch_meta,
            'train_mean':      dataset.scaler.mean_.tolist(),
            'train_std':       dataset.scaler.scale_.tolist(),
        }
        yml_path = os.path.join(paths_cfg['outputs'], f"{model_name}.yml")
        with open(yml_path, 'w') as f:
            yaml.dump(run_config, f, default_flow_style=False)
        print(f"Config saved → {yml_path}")

        # Train
        optimizer      = torch.optim.Adam(model.parameters(), lr=train_cfg['learning_rate'])
        early_stopping = EarlyStopping(patience=train_cfg['early_stopping_patience'])
        train_model(
            model=model,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=train_cfg['num_epochs'],
            early_stopping=early_stopping,
            rollout_schedule=rollout_schedule,
            val_rollout_steps=val_rollout_steps,
            grad_clip=grad_clip,
            device=device,
        )

        # Save full checkpoint (weights + arch + scaler stats)
        save_model(model, save_path, dataset.scaler.mean_, dataset.scaler.scale_, arch_meta)
        print(f"Saved model → {save_path}.pth")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch ML training for Lorenz surrogates.")
    parser.add_argument(
        'config',
        nargs='?',
        default='config.yml',
        help="Path to YAML config (default: config.yml)",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    print(f"Loaded config: {args.config}")
    run_training(config)
