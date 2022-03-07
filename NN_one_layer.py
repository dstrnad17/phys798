# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 20:03:19 2022

@author: erikj
"""
    
# Import Numpy & PyTorch
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging, sys, os

import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Import nn.functional
import torch.nn.functional as F
from torch.autograd import Variable




# Define the data
T = np.array([1,1])
T = T.reshape(2,1)
n = 100


class SimpleNet(nn.Module):
    # Initialize the layers
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 2)
        self.act1 = nn.ReLU() # Activation function
        self.linear2 = nn.Linear(2, 1)
    
    # Perform the computation
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        return x
 
            
# Define model 3 (Neural Network)
model3 = SimpleNet()
opt_n = torch.optim.SGD(model3.parameters(), .003)
loss_fn = F.mse_loss

# # Define a utility function to train the model
# def fit(num_epochs, model, loss_fn, opt, inputs):
#     for epoch in range(num_epochs):
#         for xb,yb in train_dl:
#             # Generate predictions
#             pred = model(xb)
#             loss = loss_fn(pred, yb)
#             # Perform gradient descent
#             loss.backward()
#             opt.step()
#             opt.zero_grad()
#     return model(inputs)

# Create empty file
logfile = os.path.realpath(__file__)[0:-2] + "log"
with open(logfile, "w") as f: pass


def xprint(msg):
    print(msg)
    f = open(logfile, "a")
    f.write(msg + "\n")
    f.close()
    
def reg_compare(T,n, model, opt, loss_fn):
    # Define Data
    X = 10 * np.random.rand(n,1)
    b = np.ones_like(X)
    X1 = np.hstack((b,X))
    X1_d = X1.astype(np.float32)
    F1 = np.dot(X1, T)
    eps = np.random.randn(n,1)
    Y1 = F1 + eps
    Y1_d = Y1.astype(np.float32)
    
    
    # Define PyTorch tensors 
    inputs = torch.from_numpy(X1_d)
    targets2 = torch.from_numpy(Y1_d)
    
    train_ds = TensorDataset(inputs, targets2)

    # Define data loader
    batch_size = 5
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    
    def fit(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_dl):
            data, target = Variable(data), Variable(target)
            opt.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            xprint('Iteration loss = '+ str(loss.detach().numpy()))
            loss.backward()
            opt.step()
        return model(inputs)

    
    # Train model 3 for 100 epochs
    y3_network = fit(20)
    e_network_sum = np.sqrt(1/n*np.dot((F1-y3_network.detach().numpy()).transpose(),(F1-y3_network.detach().numpy())))
    data_return = [X, Y1, y3_network, F1]
    return e_network_sum, data_return


error_network_sum = 0

# Set number of iterations
N = 100
for j in range(N):
    e_network_sum, data_return = reg_compare(T,n, model3, opt_n, loss_fn)
    error_network_sum += e_network_sum
    if j < N-1:
        del data_return

# Calculate average error
error_network = error_network_sum / N


xprint('Average Single layer neural network solution error = '+ str(error_network))


# Plot data points
fig, bx = plt.subplots()
bx.scatter(data_return[0],data_return[1])
bx.plot(data_return[0],data_return[2].detach().numpy(), color='m')
bx.plot(data_return[0],data_return[3], color='r')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Regression")
bx.legend(['Single Layer Neural Net', 'True','Data'])
plt.show

plt.savefig('NN_one_layer.pdf') 