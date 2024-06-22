
import torch.nn as nn

class DenseNN(nn.Module):
    def __init__(self, input_size, prev_time_steps, output_size, hidden_layers, hidden_activation, output_activation):
        super(DenseNN, self).__init__()
        layers = []
        in_features = input_size*prev_time_steps

        for hidden_units in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(nn.BatchNorm1d(hidden_units))
            layers.append(hidden_activation())
            in_features = hidden_units

        layers.append(nn.Linear(in_features, output_size))
        if output_activation is not None:
            layers.append(output_activation())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
