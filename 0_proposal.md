# Project Proposal 

## Motivations

In modern geosciences, it is the trend if replacing computationally expensive numerical solvers with machine Learning (ML) surrogate models. However, ML models predicting chaotic systems, tend to rapidly diverge from the true solution and collapse and become unreliable.

A critical open question is how different ML architectures capture the underlying variance of chaotic systems. If an ML surrogate collapses the physical variance, ensemble-based DA techniques will suffer from "filter divergence," ignoring true observations in favor of an overly confident, inaccurate ML forecast.

## Objective

This project aims to investigate how architecture variation in Neural Networks (NN) affects their performance as Forward Dynamical Models subject to Data Assimilation Techniques. The core idea is to evaluate their capacity to capture enough system variability to allow for meaningful DA corrections.

## Methodology

In this project we will use idealized controlled Machine Learning Experiment.

1. As ground truth we will utilize a real numerical formulation of the Loren 63 equation. using a high accuracy integration method to generate the true trajectory, we will use this synthetic observations, by adding gaussian noise to the true trajectory at discrete intervals.
2. We will train some machine learning surrogate toy models to learn the dynamical operator. We want to  explore the following 3 architectures:
	- Multi-Layer Perceptron (MLP)
	- Residual Multi-Layer Perceptron (ResMLP)
	- Recurrent Neural Network (RNN)
	- Long Short-Term memory (LSTM)

3. Data assimilation cycle. A Deterministic Ensemble Kalman Filter (EnKF) DA technique will be implemented using the trained ML models as the forward forecast step. Using an ensemble of $N$ members of slightly perturbed initial conditions to initialize our NN forward models and at given number of integrations steps we will apply the Data Assimilation technique.

## Evaluation and Expected Outcomes

Firstly, we will focus on understand how this toy machine learning models are able to capture the variability by analyzing the ensemble spread. We will primarily focus on use the Root Mean Square Error (RMSE) as evaluation metric between the DA-corrected ML trajectory and the ground truth. We believe that the LSTM and RNN model should be able to capture a better variability in the ensemble spread compared to the MLP resulting in a more stable long-term DA cycle, however. We want to dig deeper on the characteristic of these architectures to understand the properties that allow for effective Data Assimilation corrections.
