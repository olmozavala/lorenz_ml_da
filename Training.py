
import torch
from torch.utils.tensorboard import SummaryWriter
from os.path import join

from MachineLearning import LSTMNN, RNN
_IS_RECURRENT = (LSTMNN, RNN)

# Default rollout schedule preserved for backward-compatible callers.
# Format: list of [epoch_cutoff, rollout_steps] pairs, scanned in order.
# The first entry whose cutoff is greater than the current epoch wins; if the
# epoch exceeds every cutoff, the last entry's step count is used.
_DEFAULT_ROLLOUT_SCHEDULE = [[20, 1], [60, 2], [120, 3], [200, 4], [10000, 5]]
_DEFAULT_VAL_ROLLOUT_STEPS = 5
_DEFAULT_GRAD_CLIP = 1.0


def _normalize_schedule(schedule):
    """Return the schedule as a list of (cutoff:int, steps:int) tuples, sorted
    by cutoff. Accepts list-of-lists or list-of-tuples."""
    if not schedule:
        raise ValueError("rollout_schedule must be non-empty")
    norm = [(int(c), int(s)) for c, s in schedule]
    norm.sort(key=lambda p: p[0])
    return norm


def _phase_index_for_epoch(epoch, schedule):
    """Return the schedule index active at `epoch`."""
    for i, (cutoff, _) in enumerate(schedule):
        if epoch < cutoff:
            return i
    return len(schedule) - 1


def _rollout_steps_for_epoch(epoch, schedule):
    return schedule[_phase_index_for_epoch(epoch, schedule)][1]


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
def train_model(model, model_name, train_loader, val_loader, criterion, optimizer,
                num_epochs, early_stopping,
                rollout_schedule=None, val_rollout_steps=None, grad_clip=None,
                progress_callback=None, device='cpu'):
    """
    Train a surrogate model with a progressive multi-step rollout schedule.

    Parameters
    ----------
    rollout_schedule : list[[int, int]] or None
        List of ``[epoch_cutoff, rollout_steps]`` pairs. For recurrent models
        (``LSTMNN``, ``RNN``), entries with ``rollout_steps == 1`` are filtered
        out — a single step with zero-init hidden state gives the gates no
        temporal signal, so that phase is skipped.
    val_rollout_steps : int or None
        Rollout depth used during validation (constant across the run).
    grad_clip : float or None
        Max L2 norm for gradient clipping; ``None`` or ``<= 0`` disables it.
    """
    schedule = _normalize_schedule(rollout_schedule or _DEFAULT_ROLLOUT_SCHEDULE)
    if val_rollout_steps is None:
        val_rollout_steps = _DEFAULT_VAL_ROLLOUT_STEPS
    if grad_clip is None:
        grad_clip = _DEFAULT_GRAD_CLIP

    if isinstance(model, _IS_RECURRENT):
        schedule = [(c, s) for c, s in schedule if s > 1]
        if not schedule:
            raise ValueError(
                "rollout_schedule contains no entries with rollout_steps > 1; "
                "recurrent models need at least one multi-step phase."
            )
        print(f"[Training] Recurrent model detected — skipping rollout_steps=1 phase. "
              f"Effective schedule: {schedule}")

    writer = SummaryWriter(log_dir=f'runs/{model_name}')
    best_model_name = join('models',f'{model_name}_best_model.pth')

    model = model.to(device)
    gamma = 0.9 # Decay factor for multi-step loss
    last_rollout_steps = 0

    history = {'train_loss': [], 'val_loss': [], 'epochs': []}

    epoch = 0
    while epoch < num_epochs:
        rollout_steps = _rollout_steps_for_epoch(epoch, schedule)

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
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        writer.add_scalar('Loss/train', train_loss, epoch)

        model.eval()
        val_loss = 0.0
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
            phase = _phase_index_for_epoch(epoch, schedule)
            if phase >= len(schedule) - 1:
                break
            # Jump to first epoch of next phase (== current phase's cutoff)
            epoch = schedule[phase][0]
            continue  # skip epoch += 1

        epoch += 1

    writer.close()
    model.load_state_dict(torch.load(best_model_name, weights_only=False))
    return model, history
