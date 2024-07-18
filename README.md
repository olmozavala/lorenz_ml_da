
# README

## Overview

This repository contains Python scripts for implementing machine learning models and Data Assimilation using PyTorch. The focus is on the Lorenz63 dataset, which is a standard testbed for studying chaotic systems. The main script, `DapperML_Torch_Main.py`, is the starting file for the training and evaluation of models.

## File Descriptions

1. **CustomLoss.py**
   - This file contains custom loss functions tailored for specific requirements of the models used in this project. These loss functions are essential for optimizing the model's performance.

2. **DapperML_Torch_Main.py**
   - The main script that brings everything together. It handles the initialization, training, and evaluation of the machine learning models using the Lorenz63 dataset.

3. **DenseModel.py**
   - This file defines the dense (fully connected) neural network model architecture used for the experiments. It includes the necessary layers, forward pass, and other utilities required to build and train the model.

4. **Lorenz63Dataset.py**
   - This script is responsible for generating and managing the Lorenz63 dataset. It includes methods for data loading, preprocessing, and augmentation to prepare the data for training and evaluation.

5. **plot_helpers.py**
   - Contains helper functions for plotting and visualizing the results. This includes plotting loss curves, model predictions, and other relevant metrics to analyze the model's performance.

6. **Training.py**
   - Handles the training loop, including the setup of training parameters, optimization routines, and evaluation metrics. This script is crucial for training the model defined in `DenseModel.py` on the dataset provided by `Lorenz63Dataset.py`.

## How to Run the Main Script

To run the `DapperML_Torch_Main.py` script, follow these steps:

1. **Clone the Repository**
   - Clone this repository to your local machine using the following command:
     ```bash
     git clone git@github.com:olmozavala/lorenz_ml_da.git
     cd lorenz_ml_da
     ```

2. **Create Python environment and Install Dependencies**
   - Make sure you have Python installed. 
   - Create envirionment.
      ```
      conda create --name ml_da
      conda activate ml_da
      ```
   - Install the required dependencies using pip (since dapper can only be installed through pip):
     ```bash
     pip install -r requirements.txt
     ```

3. **Run the Main Script**
   - Execute the main script to start the training and evaluation process:
     ```bash
     python DapperML_Torch_Main.py
     ```

   - The script will initialize the dataset, build the model, and start the training process. It will also save the trained model and generate plots for visualizing the results.

## Customizing the Scripts

- You can modify the parameters in `DapperML_Torch_Main.py` to change the dataset size, model architecture, training epochs, and other hyperparameters.

## Contact

For any questions or issues, please raise an issue in the repository or contact the maintainer at osz09@fsu.edu
