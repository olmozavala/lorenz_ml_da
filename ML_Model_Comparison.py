import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datasets.LorenzDataset import LorenzDataset
from MachineLearning import DenseNN, ResDenseNN, load_model
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob

def autoregressive_predict(model, initial_input, lead_time, train_mean, train_std, prev_time_steps):
    """
    Perform multi-step prediction by feeding the model's output back as input.
    """
    current_input = initial_input.clone()
    predictions = []
    
    # current_input has shape (prev_time_steps * 3)
    # We need to keep track of the sequence of states
    sequence = current_input.view(prev_time_steps, 3)
    
    for _ in range(lead_time):
        with torch.no_grad():
            # Flatten sequence for the model input
            model_input = sequence.flatten().unsqueeze(0)
            output = model(model_input) # Output is (1, 3)
            
            predictions.append(output.squeeze(0))
            
            # Roll the sequence: remove the oldest state, add the new prediction
            sequence = torch.cat([sequence[1:], output], dim=0)
            
    return torch.stack(predictions)

def evaluate_models(prev_time_steps_range=range(1, 9), n_trials=5, outputs_dir='outputs', lead_times=[1, 4, 8, 16]):
    hidden_layers = [64, 64, 32, 16]
    
    # Results structure: results[model_type][prev_steps][lead_time] = [rmse_trial1, ...]
    results = {} 
    best_models = {} 

    # Find all models in outputs/
    existing_pth = glob.glob(os.path.join(outputs_dir, '*.pth'))
    model_configs = []
    for f in existing_pth:
        basename = os.path.basename(f)
        if basename.startswith('ResDense_'):
            m_type = "ResDense"
            rem = basename.replace('ResDense_', '')
        elif basename.startswith('Dense_'):
            m_type = "Dense"
            rem = basename.replace('Dense_', '')
        else:
            m_type = "Dense"
            rem = basename
            
        if 'steps' not in rem: continue
        steps = int(rem.split('steps')[0])
        trial = int(rem.split('trial')[1].split('.')[0])
        model_configs.append((m_type, steps, trial, basename))

    print("Generating validation dataset...")
    val_dataset_base = LorenzDataset(Ns=1000, std=0.0) 
    
    # Group and evaluate
    unique_types = sorted(list(set(c[0] for c in model_configs)))
    unique_steps = sorted(list(set(c[1] for c in model_configs)))

    for m_type in unique_types:
        results[m_type] = {steps: {lt: [] for lt in lead_times} for steps in unique_steps}
        best_models[m_type] = {}
        
        m_class = ResDenseNN if m_type == "ResDense" else DenseNN
        
        for prev_steps in unique_steps:
            configs = [c for c in model_configs if c[0] == m_type and c[1] == prev_steps]
            if not configs: continue
            
            print(f"\nEvaluating Type: {m_type} | Look-back: {prev_steps}")
            test_dataset = LorenzDataset(system_type=m_type.replace('ResDense', '63').replace('Dense', '63'), prev_time_steps=prev_steps, Ns=1000, std=0.0)
            
            best_rmse_1step = float('inf')
            
            for _, _, trial, f_name in configs:
                model_path = os.path.join(outputs_dir, f_name)
                
                try:
                    model, train_mean, train_std = load_model(
                        model_path, m_class, 3, prev_steps, 3, hidden_layers, nn.ReLU, None
                    )
                except Exception as e:
                    print(f"  Error loading {f_name}: {e}")
                    continue
                
                model.eval()
                all_preds = {lt: [] for lt in lead_times}
                all_targets = {lt: [] for lt in lead_times}
                max_lead = max(lead_times)
                eval_indices = range(prev_steps, len(test_dataset) - max_lead, 10) # Subsample for speed

                for idx in eval_indices:
                    initial_input_norm, _ = test_dataset[idx]
                    preds_norm = autoregressive_predict(model, initial_input_norm, max_lead, train_mean, train_std, prev_steps)
                    
                    for lt in lead_times:
                        pred_phys = preds_norm[lt-1].numpy() * train_std + train_mean
                        target_raw = test_dataset.inverse_transform(test_dataset.data[idx + lt - 1].reshape(1, -1)).flatten()
                        all_preds[lt].append(pred_phys)
                        all_targets[lt].append(target_raw)

                for lt in lead_times:
                    preds = np.array(all_preds[lt])
                    targets = np.array(all_targets[lt])
                    rmse = np.sqrt(np.mean((preds - targets)**2))
                    results[m_type][prev_steps][lt].append(rmse)
                    
                    if lt == 1 and rmse < best_rmse_1step:
                        best_rmse_1step = rmse
                        best_models[m_type][prev_steps] = (model, train_mean, train_std, trial)

                print(f"  Trial {trial} | 1-step RMSE: {results[m_type][prev_steps][1][-1]:.4f}")

    plot_comparison(results, lead_times)
    plot_trajectories(best_models, val_dataset_base)

def plot_comparison(results, lead_times):
    plt.figure(figsize=(14, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for i, (m_type, type_results) in enumerate(results.items()):
        steps_list = sorted(type_results.keys())
        for j, lt in enumerate(lead_times):
            means = [np.mean(type_results[s][lt]) if type_results[s][lt] else np.nan for s in steps_list]
            stds = [np.std(type_results[s][lt]) if type_results[s][lt] else np.nan for s in steps_list]
            
            line_style = '-' if m_type == "ResDense" else '--'
            plt.errorbar(steps_list, means, yerr=stds, label=f'{m_type} Lead:{lt}', 
                         capsize=5, marker='o', linestyle=line_style)

    plt.yscale('log')
    from matplotlib.ticker import ScalarFormatter
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())
    plt.gca().yaxis.set_minor_formatter(ScalarFormatter())
    
    plt.xlabel('Previous Time Steps (Look-back)')
    plt.ylabel('RMSE (Physical Units / Denormalized)')
    plt.title('ML Model Performance Comparison: Physical Space RMSE')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.savefig('outputs/rmse_multistep_comparison.png')
    plt.show()

def plot_3d_trajectories(best_models_wrapped, dataset, ensemble_size=20, perturbation_std=0.01):
    """
    Plots true vs ensemble predictions in 3D for both model types side-by-side.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    if not best_models_wrapped: return
    
    m_types = sorted([m for m in best_models_wrapped.keys() if best_models_wrapped[m]])
    if not m_types: return
    
    # Try to find a common highest look-back for fair comparison
    all_lbs = [set(best_models_wrapped[m].keys()) for m in m_types]
    common_lbs = set.intersection(*all_lbs) if all_lbs else set()
    
    if common_lbs:
        plot_lb = max(common_lbs)
    else:
        # Fallback to the individual best (highest available)
        plot_lb = None 

    fig = plt.figure(figsize=(9 * len(m_types), 10))
    start_idx = 100
    rollout_len = 50 
    
    for i, m_type in enumerate(m_types):
        lb = plot_lb if plot_lb in best_models_wrapped[m_type] else sorted(best_models_wrapped[m_type].keys())[-1]
        model, train_mean, train_std, trial = best_models_wrapped[m_type][lb]
        
        # Ground truth needs to cover both the input sequence and the rollout
        gt_data_norm = dataset.data[start_idx : start_idx + lb + rollout_len]
        gt_phys = dataset.inverse_transform(gt_data_norm)
        
        base_input_norm = torch.tensor((gt_phys[:lb] - train_mean) / train_std, dtype=torch.float32).flatten()
        
        ax = fig.add_subplot(1, len(m_types), i+1, projection='3d')
        
        # Plot Ensemble Members
        all_preds_phys = []
        for _ in range(ensemble_size):
            perturbed_input = base_input_norm + torch.randn_like(base_input_norm) * perturbation_std
            preds_norm = autoregressive_predict(model, perturbed_input, rollout_len, train_mean, train_std, lb)
            preds_phys = preds_norm.numpy() * train_std + train_mean
            all_preds_phys.append(preds_phys)
            ax.plot(preds_phys[:, 0], preds_phys[:, 1], preds_phys[:, 2], color='red', alpha=0.1, linewidth=0.5)
        
        ensemble_mean = np.mean(all_preds_phys, axis=0)
        
        # True trajectory for the rollout period
        ax.plot(gt_phys[lb:, 0], gt_phys[lb:, 1], gt_phys[lb:, 2], 'k--', label='True Trajectory', alpha=0.7)
        ax.plot(ensemble_mean[:, 0], ensemble_mean[:, 1], ensemble_mean[:, 2], 'r-', label=f'{m_type} Mean Forecast', alpha=0.9, linewidth=2)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{m_type} Model\nLook-back: {lb}, Trial {trial}')
        ax.legend()
    
    plt.suptitle(f'Lorenz63 Attractor: Dense vs ResDense Comparison (Rollout: {rollout_len} steps)', fontsize=16)
    plt.savefig('outputs/trajectory_3d_comparison.png')
    plt.show()

def plot_trajectories(best_models, dataset, ensemble_size=20, perturbation_std=0.01):
    """
    Plots ensemble predictions for each configuration in a 2-column layout.
    Left: Dense, Right: ResDense
    """
    start_idx = 100
    rollout_len = 50
    
    gt_data_norm = dataset.data[start_idx : start_idx + rollout_len]
    gt_phys = dataset.inverse_transform(gt_data_norm)
    
    m_types = sorted(best_models.keys()) # ['Dense', 'ResDense']
    all_lookbacks = set()
    for m_type in m_types:
        all_lookbacks.update(best_models[m_type].keys())
    sorted_lbs = sorted(list(all_lookbacks))
    
    if not sorted_lbs: return
    
    fig, axes = plt.subplots(len(sorted_lbs), len(m_types), figsize=(18, 4 * len(sorted_lbs)), sharex=False, sharey=True)
    if len(sorted_lbs) == 1: axes = np.array([axes])
    if len(m_types) == 1: axes = axes.reshape(-1, 1)

    for i, lb in enumerate(sorted_lbs):
        for j, m_type in enumerate(m_types):
            ax = axes[i, j]
            if lb not in best_models[m_type]:
                ax.text(0.5, 0.5, 'Model not found', ha='center', va='center')
                continue
                
            model, train_mean, train_std, trial = best_models[m_type][lb]
            
            # Ground truth for total window (Look-back + Rollout)
            gt_data_norm = dataset.data[start_idx : start_idx + lb + rollout_len]
            gt_phys = dataset.inverse_transform(gt_data_norm)
            
            raw_seq = gt_phys[:lb]
            base_input_norm = torch.tensor((raw_seq - train_mean) / train_std, dtype=torch.float32).flatten()
            
            all_preds_phys = []
            for _ in range(ensemble_size):
                perturbed_input = base_input_norm + torch.randn_like(base_input_norm) * perturbation_std
                preds_norm = autoregressive_predict(model, perturbed_input, rollout_len, train_mean, train_std, lb)
                preds_phys = preds_norm.numpy() * train_std + train_mean
                all_preds_phys.append(preds_phys)
                
                # Plot starting AFTER the look-back window
                ax.plot(range(lb, lb + rollout_len), preds_phys[:, 0], color='red', alpha=0.1, linewidth=0.5)
            
            ensemble_mean = np.mean(all_preds_phys, axis=0)
            
            # Plot the full Ground Truth (LB input + Rollout target)
            ax.plot(range(lb + rollout_len), gt_phys[:, 0], 'k--', label='True X', alpha=0.6)
            
            # Plot Ensemble Mean starting after LB
            ax.plot(range(lb, lb + rollout_len), ensemble_mean[:, 0], 'r-', label=f'{m_type} Mean Forecast', alpha=0.8, linewidth=2)
            
            ax.set_title(f'{m_type} | LB: {lb} (Trial {trial})')
            if j == 0: ax.set_ylabel(f'LB: {lb}\nX Position')
            ax.axvline(x=lb, color='blue', linestyle=':', alpha=0.5, label='Forecast Start')
            ax.legend(loc='upper right', fontsize='7')
            ax.set_xlabel('Time Steps')
            ax.grid(True)
    plt.tight_layout()
    plt.savefig('outputs/trajectory_comparison.png')
    plt.show()
    
    # 3D plot comparison
    plot_3d_trajectories(best_models, dataset, ensemble_size, perturbation_std)

if __name__ == "__main__":
    evaluate_models()
