import numpy as np

from dapper.stats import unbias_var, weight_degeneracy
from dapper.tools.linalg import mldiv, mrdiv, pad0, svd0, tinv
from dapper.tools.matrices import chol_reduce, funm_psd
from dapper.tools.progressbar import progbar
from dapper.tools.seeding import rng

import torch
import torch.nn as nn
# from . import da_method  # Removed since it's unused and causing ImportError



class DenseNN(nn.Module):
    def __init__(self, input_size, prev_time_steps, output_size, hidden_layers, hidden_activation, output_activation):
        # super(DenseNN, self).__init__()
        # layers = []
        # in_features = input_size * prev_time_steps  # Total input size (considering time steps)

        # for hidden in hidden_layers:
        #     layers.append(nn.Linear(in_features, hidden))
        #     layers.append(hidden_activation())  # Activation function
        #     in_features = hidden

        # layers.append(nn.Linear(in_features, output_size))  # Output layer
        # if output_activation:
        #     layers.append(output_activation())  # Output activation (if any)

        # self.model = nn.Sequential(*layers)
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
    A Dense Neural Network that learns the increment (Delta) between the last
    input state and the next state, rather than the full state.
    """
    def __init__(self, input_size, prev_time_steps, output_size, hidden_layers, hidden_activation, output_activation):
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
        
        # Optional: include output activation if provided
        if output_activation:
            layers.append(output_activation())
            
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch_size, input_size * prev_time_steps)
        # We assume the input is flattened as [state_{t-n}, ..., state_{t-1}]
        # The last state in the sequence is at the end of the flattened vector
        current_state = x[:, -self.input_size:]
        delta = self.network(x)
        return current_state + delta

class LSTMNN(nn.Module):
    def __init__(self, input_size, prev_time_steps, output_size, hidden_size, num_layers=1):
        super(LSTMNN, self).__init__()
        self.input_size = input_size
        self.prev_time_steps = prev_time_steps
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, input_size * prev_time_steps)
        # Reshape to (batch, prev_time_steps, input_size)
        x = x.view(-1, self.prev_time_steps, self.input_size)
        out, _ = self.lstm(x)
        # out: (batch, seq_len, hidden_size)
        # Take the last output
        out = out[:, -1, :]
        return self.fc(out)

def save_model(model, model_path, train_mean, train_std, architecture):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'train_mean': train_mean,
        'train_std': train_std,
        'architecture': architecture
    }
    torch.save(checkpoint, f"{model_path}.pth")

def load_model(model_path, model_class, input_size, prev_time_steps, output_size, hidden_layers, hidden_activation, output_activation):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)  # Load checkpoint
    model = model_class(input_size, prev_time_steps, output_size, hidden_layers, hidden_activation, output_activation)  # Initialize model
    model.load_state_dict(checkpoint['model_state_dict'])  # Load weights
    model.eval()  

    train_mean = checkpoint['train_mean']
    train_std = checkpoint['train_std']
    

    return model, train_mean, train_std
def denormalize(predictions, train_mean, train_std):
    return predictions * train_std + train_mean  # Reverse normalization



def ml_step(model,x,t,dt, model_mean, model_std):
	N, d = x.shape # N = ensemble size, d = dimension(3)
	x_norm = (x-model_mean)/model_std
	x_norm = torch.tensor(x_norm, dtype = torch.float32)
	with torch.no_grad():
		output = model(x_norm.unsqueeze(0))
		output = output.squeeze().numpy()
	x_next = output*model_std + model_mean
	return x_next


def realLoad():
	model_path = "/Users/katymerritt/Desktop/Python/DataSimulation/lorenz_ml_da/Test_8StepModel.pth"
	input_size = 3
	output_size = 3
	prev_time_steps = 8  # Adjust based on your training data
	hidden_layers = [64, 64, 32, 16]
	hidden_activation = nn.ReLU
	output_activation = None

	model, model_mean, model_std = load_model(
    	model_path, DenseNN, input_size, prev_time_steps, output_size, hidden_layers, hidden_activation, output_activation
	)
	return model, model_mean, model_std




