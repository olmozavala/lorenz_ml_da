# %%
import torch
import numpy as np
from os.path import join
from DenseModel import DenseNN
import torch.nn as nn

folder = "/home/olmozavala/Dropbox/MyProjects/EOAS/COAPS/Katy_Hristo_DA_NN/models"
model_name = 'prevTimeS_8_std_0_saveDt_1_Ns_10000_hidden_layers_128_64_64_64_64_20240916_164633_best_model.pth'

input_size = 3  # Number of input features (always 3 for this Lorenz)
output_size = 3  # Number of output features (always 3 for this Lorenz)
prev_time_steps = 8 # Number of previous time steps to consider as input
# hidden_layers = [64, 64, 32, 16]
hidden_layers = [128, 64, 64, 64, 64]
hidden_activation = nn.ReLU
output_activation = None

model = DenseNN(input_size, prev_time_steps, output_size, hidden_layers, hidden_activation, output_activation)
model.eval()

# %%
model_file = join(folder, model_name)
model.load_state_dict(torch.load(model_file))

# %%Synthetic input data
input_tensor = torch.randn(3*8)  # Create a batch of 1 with 3 * 8 = 24 features
output = model(input_tensor.unsqueeze(0))
print(output)