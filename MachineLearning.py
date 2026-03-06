import numpy as np
import torch
import torch.nn as nn


class DenseNN(nn.Module):
    """
    Feedforward network: flattened history window → hidden layers → next state.
    Input shape:  (batch, input_size * prev_time_steps)
    Output shape: (batch, output_size)
    """
    def __init__(self, input_size, prev_time_steps, output_size,
                 hidden_layers, hidden_activation, output_activation):
        super(DenseNN, self).__init__()
        self.prev_time_steps = prev_time_steps

        layers = []
        in_features = input_size * prev_time_steps

        for hidden in hidden_layers:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(hidden_activation())
            in_features = hidden

        layers.append(nn.Linear(in_features, output_size))

        if output_activation:
            layers.append(output_activation())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ResDenseNN(nn.Module):
    """
    Residual feedforward network: predicts the increment Δ between the most
    recent input state and the next state, then adds it back as a skip connection.

    Instead of learning  f(history) → s_{t+1}
    it learns            f(history) → Δ,  and returns  s_t + Δ

    This is easier to optimise (the network only needs to learn small corrections)
    and generally generalises better than the plain DenseNN.

    Input shape:  (batch, input_size * prev_time_steps)
    Output shape: (batch, output_size)
    """
    def __init__(self, input_size, prev_time_steps, output_size,
                 hidden_layers, hidden_activation, output_activation):
        super(ResDenseNN, self).__init__()
        self.input_size = input_size
        self.prev_time_steps = prev_time_steps

        layers = []
        in_features = input_size * prev_time_steps

        for hidden in hidden_layers:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(hidden_activation())
            in_features = hidden

        layers.append(nn.Linear(in_features, output_size))

        if output_activation:
            layers.append(output_activation())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, input_size * prev_time_steps)
        # The flattened input is ordered [s_{t-n}, …, s_{t-1}],
        # so the most recent state occupies the last input_size columns.
        current_state = x[:, -self.input_size:]
        delta = self.network(x)
        return current_state + delta


class LSTMNN(nn.Module):
    """
    LSTM-based model: treats the flattened history as a proper time sequence.

    The flattened input (batch, prev_time_steps * input_size) is first reshaped
    to (batch, prev_time_steps, input_size) before being fed to the LSTM.
    The hidden state at the last time step is mapped to the next state via a
    fully-connected layer.

    Input shape:  (batch, input_size * prev_time_steps)
    Output shape: (batch, output_size)
    """
    def __init__(self, input_size, prev_time_steps, output_size,
                 hidden_size, num_layers=1):
        super(LSTMNN, self).__init__()
        self.input_size = input_size
        self.prev_time_steps = prev_time_steps
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, input_size * prev_time_steps)
        x = x.view(-1, self.prev_time_steps, self.input_size)
        # out: (batch, prev_time_steps, hidden_size)
        out, _ = self.lstm(x)
        # Use only the last time-step's hidden state for prediction
        return self.fc(out[:, -1, :])


def save_model(model, model_path, train_mean, train_std, architecture):
    """
    Saves a full checkpoint containing model weights, normalisation constants,
    and architecture metadata needed to reconstruct the model at inference time.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'train_mean':       train_mean,
        'train_std':        train_std,
        'architecture':     architecture,
    }
    torch.save(checkpoint, f"{model_path}.pth")


def load_model(model_path, model_class, input_size, prev_time_steps,
               output_size, hidden_layers, hidden_activation, output_activation):
    """
    Reconstructs a model from a full checkpoint saved by save_model().
    Returns (model, train_mean, train_std).
    """
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'),
                            weights_only=False)
    model = model_class(input_size, prev_time_steps, output_size,
                        hidden_layers, hidden_activation, output_activation)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['train_mean'], checkpoint['train_std']


def denormalize(predictions, train_mean, train_std):
    """Reverses StandardScaler normalisation: pred * std + mean."""
    return predictions * train_std + train_mean
