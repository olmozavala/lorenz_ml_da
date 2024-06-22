# %%
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
input_size = 3 
output_size = 3 
prev_time_steps = 10
hidden_layers = [64, 64, 64, 32, 16]
hidden_activation = nn.ReLU
output_activation = None
batch_size = 512
num_epochs = 500
learning_rate = 0.001
patience = 30
# Parameters for the Lorenz63 Dataset
std = 1 # Standard deviation of the noise
save_Dt = 10
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
model_name = f'LorenzDenseNN_prevTimeSteps_{prev_time_steps}_std_{std}_' +\
                f'saveDt_{save_Dt}_Ns_{Ns}_{cur_time}'

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
# %% Denormalize 
# predictions = np.array(dataset.inverse_transform(np.array(predictions)))
predictions = np.array(predictions).squeeze()
# %%
all_loss /= len(train_data)
print(f'All Loss: {all_loss:.5f}')
plot_x_3d(predictions, color='g', save_path=f'{model_name}_full_dataset.png',
          title=f'Predictions for the full dataset, loss: {all_loss:.5f} \n Model: {model_name}')

# %% Using only initial conditions to predict future states and plot error
start_time = 300
# start_time = len(train_dataset) 
input = train_data[start_time]
input_tensor = torch.tensor(train_data[start_time:start_time+prev_time_steps].flatten(), dtype=torch.float32)
time_steps = 100
error = []
for i in range(time_steps):
    output = model(input_tensor.unsqueeze(0))
    input = output.squeeze().detach().numpy()
    error.append(target_data[i+start_time] - input)
    keep = input_tensor[input_size:].clone()
    input_tensor = torch.cat((keep,torch.tensor(input, dtype=torch.float32)))

plt.plot(error)
plt.title(f'Error between predicted and true values start time: {start_time}')
plt.xlabel(f'Time Steps with dt={save_Dt}')
plt.ylabel('Error')
plt.savefig(join('imgs',f'{model_name}_error.png'))
# plt.show()

# %% Plot the test data
# model_file = f'{model_name}_best_model.pth'
# model.eval()
# test_loss = 0.0
# predictions = []
# train_data, target_data = dataset.get_all_data()
# with torch.no_grad():
#     # for inputs, targets in test_loader:
#     for inputs, targets in train_loader:
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         test_loss += loss.item() * inputs.size(0)
#         # for x in outputs:
#         for x in inputs:
#             predictions.append(x.numpy())
# predictions = np.array(dataset.inverse_transform(predictions))
# # %%
# test_loss /= len(test_loader.dataset)
# print(f'Test Loss: {test_loss:.4f}')
# plot_x_3d(predictions, color='g')
# # %%