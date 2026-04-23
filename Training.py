
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from os.path import join

from MachineLearning import LSTMNN, RNN
_IS_RECURRENT = (LSTMNN, RNN)

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

    recurrent = isinstance(model, _IS_RECURRENT)
    hidden = None  # PyTorch auto-inits to zeros

    for _ in range(num_steps):
        # Flatten for the model input
        model_in = current_input.reshape(batch_size, -1)
        if recurrent:
            pred, hidden = model(model_in, hidden)
        else:
            pred = model(model_in)  # (batch, nx)
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
    
    epoch = 0
    while epoch < num_epochs:
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
            # Jump to first epoch of next phase
            if rollout_steps == 1:   epoch = 20
            elif rollout_steps == 2: epoch = 60
            elif rollout_steps == 3: epoch = 120
            elif rollout_steps == 4: epoch = 200
            continue  # skip epoch += 1

        epoch += 1

    writer.close()
    model.load_state_dict(torch.load(best_model_name, weights_only=False))
    return model, history
