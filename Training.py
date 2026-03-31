
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from os.path import join

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

# Training loop
def recursive_rollout(model, initial_input, num_steps, prev_time_steps, device):
    """
    Performs a recursive rollout of the model for a given number of steps.
    """
    batch_size = initial_input.shape[0]
    nx = initial_input.shape[1] // prev_time_steps
    
    # initial_input is (batch, prev_time_steps * nx)
    current_input = initial_input.view(batch_size, prev_time_steps, nx)
    outputs_list = []
    
    for _ in range(num_steps):
        # Flatten for the model input
        model_in = current_input.reshape(batch_size, -1)
        pred = model(model_in) # (batch, nx)
        outputs_list.append(pred)
        
        # Roll the sequence: [s1, s2, s3] -> [s2, s3, pred]
        # pred.unsqueeze(1) is (batch, 1, nx)
        current_input = torch.cat([current_input[:, 1:, :], pred.unsqueeze(1)], dim=1)
        
    return torch.stack(outputs_list, dim=1) # (batch, num_steps, nx)

# Training loop
def train_model(model, model_name, train_loader, val_loader, criterion, optimizer, num_epochs, early_stopping, progress_callback=None, device='cpu'):
    writer = SummaryWriter(log_dir=f'runs/{model_name}')
    best_val_loss = float('inf')
    best_model_name = join('models',f'{model_name}_best_model.pth')
    
    model = model.to(device)
    gamma = 0.9 # Decay factor for multi-step loss
    last_rollout_steps = 0
    
    history = {'train_loss': [], 'val_loss': [], 'epochs': []}
    
    for epoch in range(num_epochs):
        # Segmented schedule logic
        if epoch < 20:
            rollout_steps = 1
        elif epoch < 60:
            rollout_steps = 2
        elif epoch < 120:
            rollout_steps = 3
        elif epoch < 200:
            rollout_steps = 4
        else:
            rollout_steps = 5
            
        # Reset Early Stopping if rollout depth increases
        if rollout_steps > last_rollout_steps:
            early_stopping.counter = 0
            early_stopping.best_loss = None 
            last_rollout_steps = rollout_steps

        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            # Predict recursively
            preds = recursive_rollout(model, inputs, rollout_steps, model.prev_time_steps, device)
            
            # Weighted loss across rollout steps
            loss = 0
            for s in range(rollout_steps):
                step_loss = criterion(preds[:, s, :], targets[:, s, :])
                loss += (gamma ** s) * step_loss
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        writer.add_scalar('Loss/train', train_loss, epoch)

        model.eval()
        val_loss = 0.0
        val_rollout_steps = 5 
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                preds = recursive_rollout(model, inputs, val_rollout_steps, model.prev_time_steps, device)
                
                loss = 0
                for s in range(val_rollout_steps):
                    step_loss = criterion(preds[:, s, :], targets[:, s, :])
                    loss += (gamma ** s) * step_loss
                
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        writer.add_scalar('Loss/val', val_loss, epoch)

        history['epochs'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if progress_callback:
            if progress_callback(epoch + 1, train_loss, val_loss, rollout_steps):
                print("Training stop requested via callback.")
                break

        # Save model if validation loss improves.
        if early_stopping.best_loss is None or val_loss < early_stopping.best_loss:
             torch.save(model.state_dict(), best_model_name)

        if early_stopping(val_loss):
            if rollout_steps >= 5:
                break 
            # Advance to next phase
            if rollout_steps == 1: epoch = 19
            elif rollout_steps == 2: epoch = 59
            elif rollout_steps == 3: epoch = 119
            elif rollout_steps == 4: epoch = 199

    writer.close()
    model.load_state_dict(torch.load(best_model_name, weights_only=False))
    return model, history


def train_esn_ridge(model, model_name, train_loader, val_loader, device,
                    ridge_alpha=1e-6):
    """
    Train ESN readout via ridge regression (closed-form solution).

    Steps:
    1. Collect all reservoir states H and single-step targets Y from training set
    2. Solve readout weights: W = (H^T H + alpha I)^{-1} H^T Y
    3. Set model.readout weights to the solution
    4. Evaluate on validation set
    """
    writer = SummaryWriter(log_dir=f'runs/{model_name}')
    model = model.to(device)
    model.eval()

    # Step 1: Collect reservoir states and single-step targets
    all_states = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            h = model._run_reservoir(inputs)       # (batch, reservoir_size)
            all_states.append(h.cpu())
            all_targets.append(targets[:, 0, :].cpu())  # single-step target

    H = torch.cat(all_states, dim=0)    # (N_train, reservoir_size)
    Y = torch.cat(all_targets, dim=0)   # (N_train, output_size)

    # Step 2: Ridge regression closed-form
    # W = (H^T H + alpha * I)^{-1} H^T Y
    I = torch.eye(H.shape[1])
    W = torch.linalg.solve(H.T @ H + ridge_alpha * I, H.T @ Y)
    # W shape: (reservoir_size, output_size)

    # Step 3: Set readout weights
    with torch.no_grad():
        model.readout.weight.copy_(W.T)     # nn.Linear weight is (out, in)
        model.readout.bias.copy_((Y - H @ W).mean(dim=0))

    # Step 4: Compute training loss
    criterion = nn.MSELoss()
    train_loss = criterion(H @ W, Y).item()
    writer.add_scalar('Loss/train', train_loss, 0)

    # Step 5: Validate
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)
            val_loss += criterion(preds, targets[:, 0, :]).item() * inputs.size(0)
    val_loss /= len(val_loader.dataset)
    writer.add_scalar('Loss/val', val_loss, 0)

    writer.close()
    print(f"ESN ridge regression — train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

    history = {'train_loss': [train_loss], 'val_loss': [val_loss], 'epochs': [1]}
    return model, history