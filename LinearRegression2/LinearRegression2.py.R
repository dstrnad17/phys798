#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 19:32:49 2024

@author: dunnchadnstrnad
"""

import torch
import torch.nn as nn
import torch.optim as optim

# Generate synthetic data
torch.manual_seed(0)
X = torch.randn(100, 1)  # 100 data points with a single feature
y = 3 * X + 2 + torch.randn(100, 1) * 0.5  # y = 3X + 2 + noise

# Define the linear regression model
class LRTorch(nn.Module):
    def __init__(self):
        super(LRTorch, self).__init__()
        self.linear = nn.Linear(1, 1)  # input and output features are 1

    def forward(self, x):
        return self.linear(x)

model = LRTorch()

# Define the loss function and optimizer
crit = nn.MSELoss()
opt = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass: compute predicted y
    y_pred = model(X)

    # Compute and print loss
    loss = crit(y_pred, y)

    # Zero gradients, perform a backward pass, and update the weights
    opt.zero_grad()
    loss.backward()
    opt.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Print learned parameters
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

# Make predictions
X_new = torch.tensor([[4.0]])
y_new = model(X_new)
print(f'Prediction for input 4.0: {y_new.item():.4f}')