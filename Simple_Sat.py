# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 21:26:04 2022

@author: erikj
"""
# Import Numpy & PyTorch
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging, sys, os
from sklearn.model_selection import train_test_split




import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Import nn.functional
import torch.nn.functional as F
from torch.autograd import Variable

# Create empty file
logfile = os.path.realpath(__file__)[0:-2] + "log"
with open(logfile, "w") as f: pass


def xprint(msg):
    print(msg)
    f = open(logfile, "a")
    f.write(msg + "\n")
    f.close()


sat_data_df = pd.read_pickle("./sat_data/sat_data.pkl")


class SatSimpleNet(nn.Module):
    # Initialize the layers
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 3)
        self.act1 = nn.ReLU() # Activation function
        self.linear2 = nn.Linear(3, 3)
    
    # Perform the computation
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        return x
    
train, test = train_test_split(sat_data_df, test_size=0.05)
Position_train = train[["x[km]", "y[km]", "z[km]"]]
Field_train = train[["bx[nT]", "by[nT]", "bz[nT]"]]
Position_test = test[["x[km]", "y[km]", "z[km]"]]
Field_test = test[["bx[nT]", "by[nT]", "bz[nT]"]]

# Define PyTorch tensors 
train_inputs = torch.from_numpy(np.float32(Position_train.iloc[:10000,:].values))
train_targets = torch.from_numpy(np.float32(Field_train.iloc[:10000,:].values))
test_inputs = torch.from_numpy(np.float32(Position_test.iloc[:10000,:].values))
test_targets = torch.from_numpy(np.float32(Field_test.iloc[:10000,:].values))

train_ds = TensorDataset(train_inputs, train_targets)
test_ds = TensorDataset(test_inputs, test_targets)

# Define data loader
batch_size = 50
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
test_dl = DataLoader(test_ds, batch_size, shuffle=True)

    
# Define model 3 (Neural Network)
sat_model = SatSimpleNet()
opt_n = torch.optim.SGD(sat_model.parameters(), .00001)
loss_fn = F.mse_loss


def train(epoch):
    sat_model.train()
    for batch_idx, (data, target) in enumerate(train_dl):
        data, target = Variable(data), Variable(target)
        opt_n.zero_grad()
        output = sat_model(data)
        loss = loss_fn(output, target)
        xprint('Training iteration loss = '+ str(loss.detach().numpy()) + '\n')
        loss.backward()
        opt_n.step()
    return loss_fn(sat_model(data),target).detach().numpy()

def test():
    sat_model.eval()
    for batch_idx, (data, target) in enumerate(test_dl):
        data, target = Variable(data), Variable(target)
        output = sat_model(data)
    return loss_fn(output,target).detach().numpy()

# Train model 3 for 100 epochs
sat_nn_regresion_train = train(5)
#error_train = np.sqrt(np.dot((train_targets.detach().numpy()-sat_nn_regresion_train.detach().numpy()).transpose(),(train_targets.detach().numpy()-sat_nn_regresion_train.detach().numpy())))
sat_nn_regresion_test = test()
#error_test = np.sqrt(np.dot((test_targets.detach().numpy()-sat_nn_regresion_test.detach().numpy()).transpose(),(test_targets.detach().numpy()-sat_nn_regresion_test.detach().numpy())))

xprint('Single layer neural network training data error = '+ str(sat_nn_regresion_train) + '\n')
xprint('Single layer neural network test data error = '+ str(sat_nn_regresion_test))

