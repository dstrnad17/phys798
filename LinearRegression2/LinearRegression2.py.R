#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 19:32:49 2024

@author: dunnchadnstrnad
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle
from tabulate import tabulate

# Generate synthetic data
torch.manual_seed(0)
X = torch.randn(100, 1)  # 100 data points with a single feature
y = 3 * X + 2 + torch.randn(100, 1) * 0.5  # y = 3X + 2 + noise

### Define the PyTorch linear regression model ###
class LRTorch(nn.Module):
    def __init__(self):
        super(LRTorch, self).__init__()
        self.linear = nn.Linear(1, 1)  # input and output features are 1

    def forward(self, x):
        return self.linear(x)

model_torch = LRTorch()

# Define the loss function and optimizer
crit = nn.MSELoss()
opt = optim.SGD(model_torch.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass: compute predicted y
    y_pred = model_torch(X)

    # Compute and print loss
    loss = crit(y_pred, y)

    # Zero gradients, perform a backward pass, and update the weights
    opt.zero_grad()
    loss.backward()
    opt.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Print learned parameters
for name, param in model_torch.named_parameters():
    if name == 'linear.weight':
        w_torch = param.data.item()
        print(f'PyTorch model weight: {param.data.item():.7f}')
    if name == 'linear.bias':
        bias_torch = param.data.item()
        print(f'PyTorch model bias: {param.data.item():.7f}')

# Make predictions
x = 4.0         # Input test value

X_new_torch = torch.tensor([[x]])
y_new_torch = model_torch(X_new_torch)

# Calculate error
y_true = 3 * x + 2
error_torch = (y_new_torch.item() - y_true) / y_true

# Redefine data with NumPy
X_np = X.numpy()
y_np = y.numpy()
X_b = np.c_[np.ones((100, 1)), X_np]        # Concatenation with bias to create augmented matrix

### Define the basic linear regression model ###
theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_np)

# Make predictions
X_new_np = np.array([[0], [x]])
X_new_b = np.c_[np.ones((2, 1)), X_new_np]  # Add x0 = 1 to each instance
y_predict = X_new_b.dot(theta)

# Calculate error
error_np = (y_predict[1] - y_true) / y_true

# Display results
print(f'Numpy Weight: {theta[1].item():.7f}')
print(f'Numpy Bias: {theta[0].item():.7f}')
print(f'True value for input x = {x} (not accounting for noise): {y_true:.7f}')
print(f'Prediction for input x = {x} (NumPy): {y_predict[1].item():.7f}')
print(f'Error (NumPy): {error_np.item():.7f}')
print(f'Prediction for input x = {x} (PyTorch): {y_new_torch.item():.7f}')
print(f'Error (PyTorch): {error_torch:.7f}')

# Create data file

data = [
    ["True", 3, 2, "N/A"],
    ["Numpy", theta[1].item(), theta[0].item(), error_np.item()],
    ["Pytorch", w_torch, bias_torch, error_torch]
]
headers = ["Method", "Weight", "Bias", "Error"]

table = (tabulate(data, headers, tablefmt="grid"))
print(table)
with open('LRtable.txt', 'w') as f:
    f.write(table)
    
# Plot
y_torch_plot = w_torch * X_np + bias_torch
y_np_plot = theta[1] * X_np + theta[0]
y_true_plot = 3 * X_np + 2

plt.scatter(X_np, y_np, color='blue', label='Data')
plt.plot(X_np, y_torch_plot, color='red', label='Pytorch, err = {:.5f}'.format(error_torch), linewidth=5.0)
plt.plot(X_np, y_np_plot, color = 'green', label='Numpy, err = {:.5f}'.format(error_np.item()))
plt.plot(X_np, y_true_plot, color = 'black', label='True')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression with PyTorch and Numpy')
plt.legend()
plt.savefig('LRplot.pdf')
plt.savefig('LRplot.png')