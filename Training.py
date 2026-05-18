
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
def recursive_rollout(model, initial_input, num_steps, prev_time_steps, device, stateful_rollout=False):
    """
    Performs a recursive rollout of the model for a given number of steps.
    When stateful_rollout=True and model.stateful=True, hidden state is threaded
    across steps (proper BPTT within the rollout); hidden is never carried across batches.
    """
    batch_size = initial_input.shape[0]
    nx = initial_input.shape[1] // prev_time_steps
    current_input = initial_input.view(batch_size, prev_time_steps, nx)
    outputs_list = []
    hidden = None

    for _ in range(num_steps):
        model_in = current_input.reshape(batch_size, -1)
        if getattr(model, 'stateful', False):
            if stateful_rollout:
                pred, hidden = model(model_in, hidden)
            else:
                pred, _ = model(model_in)
        else:
            pred = model(model_in)
        outputs_list.append(pred)
        current_input = torch.cat([current_input[:, 1:, :], pred.unsqueeze(1)], dim=1)

    return torch.stack(outputs_list, dim=1)  # (batch, num_steps, nx)

# Training loop
def train_model(model, model_name, train_loader, val_loader, criterion, optimizer,
                num_epochs, early_stopping, progress_callback=None, device='cpu',
                max_rollout_steps=5, gamma=0.9, stateful_rollout=False,
                initial_lr=0.001, lr_phase_decay=1.0,
                lr_scheduler_patience=10, lr_scheduler_factor=0.5):
    writer = SummaryWriter(log_dir=f'runs/{model_name}')
    best_model_name = join('models', f'{model_name}_best_model.pth')

    model = model.to(device)
    history = {'train_loss': [], 'val_loss': [], 'epochs': []}

    epoch = 0
    rollout_steps = 1
    scheduler = ReduceLROnPlateau(optimizer, patience=lr_scheduler_patience, factor=lr_scheduler_factor)

    while epoch < num_epochs:
        # --- train ---
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = recursive_rollout(model, inputs, rollout_steps, model.prev_time_steps, device, stateful_rollout)
            loss = sum((gamma ** s) * criterion(preds[:, s, :], targets[:, s, :]) for s in range(rollout_steps))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)

        # --- validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                preds = recursive_rollout(model, inputs, rollout_steps, model.prev_time_steps, device, stateful_rollout)
                loss = sum((gamma ** s) * criterion(preds[:, s, :], targets[:, s, :]) for s in range(rollout_steps))
                #preds = recursive_rollout(model, inputs, max_rollout_steps, model.prev_time_steps, device, stateful_rollout)
                #loss = sum((gamma ** s) * criterion(preds[:, s, :], targets[:, s, :]) for s in range(max_rollout_steps))
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Rollout/steps', rollout_steps, epoch)

        history['epochs'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if progress_callback and progress_callback(epoch + 1, train_loss, val_loss, rollout_steps):
            print("Training stop requested via callback.")
            break

        # save best model within current phase
        if early_stopping.best_loss is None or val_loss < early_stopping.best_loss:
            torch.save(model.state_dict(), best_model_name)

        if early_stopping(val_loss):
            if rollout_steps >= max_rollout_steps:
                break
            # advance phase
            rollout_steps += 1
            early_stopping.counter = 0
            early_stopping.best_loss = None
            new_lr = initial_lr * (lr_phase_decay ** (rollout_steps - 1))
            optimizer.state.clear()
            for group in optimizer.param_groups:
                group['lr'] = new_lr
            scheduler = ReduceLROnPlateau(optimizer, patience=lr_scheduler_patience, factor=lr_scheduler_factor)

        epoch += 1

    writer.close()
    model.load_state_dict(torch.load(best_model_name, weights_only=False))
    return model, history
