# %%
# Code to reload modules when changes are made
# %load_ext autoreload
# %autoreload 2
from plot_helpers import plot_x_y_da, plot_x_3d
from datetime import datetime
# Import torch related libraries
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
# Import local modules
from DenseModel import DenseNN
from CustomLoss import RMSELoss
from Lorenz63Dataset import Lorenz63Dataset
from Training import train_model, EarlyStopping
import matplotlib.pyplot as plt
import numpy as np
from os.path import join

# %% Parameters
input_size = 3  # Number of input features (always 3 for this Lorenz)
output_size = 3  # Number of output features (always 3 for this Lorenz)
prev_time_steps = 8 # Number of previous time steps to consider as input
# hidden_layers = [64, 64, 32, 16]
hidden_layers = [128, 64, 64, 64, 64]
hidden_activation = nn.ReLU
output_activation = None
batch_size = 512
num_epochs = 500 # Max number of epochs
learning_rate = 0.001
patience = 30  # Early stopping patience
# Parameters for the Lorenz63 Dataset
std = 0 # Standard deviation of the noise
save_Dt = 1 # Save every save_Dt time steps from the Lorenz63 system
Ns = 10000 # Number of samples

# %% Create Dataset and DataLoader
dataset = Lorenz63Dataset(prev_time_steps=prev_time_steps, std=std, save_Dt=save_Dt, Ns=Ns)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# %% Plot training data
train_data, target_data = dataset.get_all_data()
# plot_x_3d(train_data, linewidth=0.5, title='Training Data')
# plot_x_3d(target_data, color='r')

# %% Model, Loss, Optimizer, EarlyStopping
model = DenseNN(input_size, prev_time_steps, output_size, hidden_layers, hidden_activation, output_activation)
criterion = RMSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
early_stopping = EarlyStopping(patience=patience, min_delta=0.01)

# %% Train Model
# Create a model name using the current date and time and the parameters
cur_time = datetime.now().strftime('%Y%m%d_%H%M%S')
model_name = f'prevTimeS_{prev_time_steps}_std_{std}_' +\
                f'saveDt_{save_Dt}_Ns_{Ns}_' +\
                    f'hidden_layers_{"_".join(map(str,hidden_layers))}_{cur_time}'

print(f'Model Name: {model_name}')
model = train_model(model, model_name, train_loader, val_loader, criterion, optimizer, num_epochs, early_stopping)

# %% ================ For debugging, evaluate on the full dataset
model_file = f'{model_name}_best_model.pth'
model.eval()
predictions = []
all_loss = 0.0
with torch.no_grad():
    for i in range(len(train_data)-prev_time_steps):
        input_tensor = torch.tensor(train_data[i:i+prev_time_steps].flatten(), dtype=torch.float32)
        target_tensor = torch.tensor(target_data[i], dtype=torch.float32)
        output = model(input_tensor.unsqueeze(0))
        loss = criterion(output, target_tensor.reshape(1, -1))
        all_loss += loss.item()
        predictions.append(output.numpy())
predictions = np.array(predictions).squeeze()
# %%
all_loss /= len(train_data)
print(f'All Loss: {all_loss:.5f}')
plot_x_3d(predictions, color='g', save_path=f'{model_name}_full_dataset.png',
          title=f'Predictions, full dataset from restart, loss: {all_loss:.5f} \n Model: {model_name}')

# %% Using only initial conditions to predict future states and plot error
start_time = 1
# start_time = len(train_dataset) 
input = train_data[start_time]
input_tensor = torch.tensor(train_data[start_time:start_time+prev_time_steps].flatten(), dtype=torch.float32)
time_steps = 1000
error = []
predictions = []
for i in range(time_steps):
    output = model(input_tensor.unsqueeze(0))

    target_tensor = torch.tensor(target_data[i+start_time], dtype=torch.float32)
    loss = criterion(output, target_tensor.reshape(1, -1))

    output = output.squeeze().detach().numpy()
    predictions.append(output)
    error.append(target_data[i+start_time] - output)
    all_loss += loss.item()
    keep = input_tensor[input_size:].clone()
    input_tensor = torch.cat((keep,torch.tensor(output, dtype=torch.float32)))

plt.plot(error)
plt.title(f'Error between predicted and true values start time: {start_time} \n {model_name}')
plt.xlabel(f'Time Steps with dt={save_Dt}')
plt.ylabel('Error')
plt.savefig(join('imgs',f'{model_name}_error.png'))
# plt.show()
# %%
show_figure = True
all_loss /= len(train_data)
title =f'Predictions, {time_steps} steps from initial condition, loss: {all_loss:.5f} \n Model: {model_name}'
plot_x_3d(np.array(predictions), color='g', save_path=f'{model_name}_error_IC.png',
          title=title, linewidth=0.5, show_figure=show_figure)

plot_x_y_da(np.array(predictions), 
            target_data[start_time:start_time+time_steps], title=title,
            save_path=f'{model_name}_pred_IC_.png', show_figure=show_figure)
# %%
