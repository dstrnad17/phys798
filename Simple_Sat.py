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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer




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
sat_data_df = sat_data_df.sort_values(by='epoch', ascending=True)

position = ["x[km]", "y[km]", "z[km]"]
field = ["bx[nT]", "by[nT]", "bz[nT]"]

pipe = Pipeline(steps=[("scaler", StandardScaler()), ("unit", MaxAbsScaler())])

preprocessor_pos = ColumnTransformer(transformers=[("num_pos", pipe, position)])
preprocessor_field = ColumnTransformer(transformers=[("num_field", pipe, field)])

class SatSimpleNet(nn.Module):
    # Initialize the layers
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 100)
        self.act1 = nn.Tanh() # Activation function
        self.linear2 = nn.Linear(100, 100)
        self.act2 = nn.Tanh() # Activation function
        self.linear3 = nn.Linear(100, 3)
    
    # Perform the computation
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        x = self.linear3(x)
        return x
    
train, test = train_test_split(sat_data_df, test_size=0.05)
Position_train = train[["x[km]", "y[km]", "z[km]"]]
Field_train = train[["bx[nT]", "by[nT]", "bz[nT]"]]
Position_test = test[["x[km]", "y[km]", "z[km]"]]
Field_test = test[["bx[nT]", "by[nT]", "bz[nT]"]]

preprocessor_pos.fit(Position_train)
preprocessor_field.fit(Field_train)


# Define PyTorch tensors 
# train_inputs = torch.from_numpy(np.float32(Position_train.iloc[:100000,:].values))
# train_targets = torch.from_numpy(np.float32(Field_train.iloc[:100000,:].values))
# test_inputs = torch.from_numpy(np.float32(Position_test.iloc[:100000,:].values))
# test_targets = torch.from_numpy(np.float32(Field_test.iloc[:100000,:].values))

train_inputs = torch.from_numpy(np.float32(Position_train))
train_targets = torch.from_numpy(np.float32(Field_train))
test_inputs = torch.from_numpy(np.float32(Position_test))
test_targets = torch.from_numpy(np.float32(Field_test))

train_ds = TensorDataset(train_inputs, train_targets)
test_ds = TensorDataset(test_inputs, test_targets)

# Define data loader
batch_size = 25
train_dl = DataLoader(train_ds, batch_size, shuffle=False)
test_dl = DataLoader(test_ds, batch_size, shuffle=False)

    
# Define model (Neural Network)
sat_model = SatSimpleNet()
opt_n = torch.optim.Adam(sat_model.parameters(), .005)
loss_fn = F.mse_loss


np.random.seed(123)
torch.manual_seed(123)
loss_saved = []
def train(epoch):
    sat_model.train()
    for batch_idx, (data, target) in enumerate(train_dl):
        data, target = Variable(data), Variable(target)
        opt_n.zero_grad()
        output = sat_model(data)
        loss = loss_fn(output, target)
        loss_saved.append(loss.detach().numpy())
        xprint('Training iteration loss = '+ str(loss.detach().numpy()) + '\n')
        loss.backward()
        opt_n.step()
    return loss_fn(sat_model(data),target).detach().numpy()

def test():
    sat_model.eval()
    for batch_idx, (data, target) in enumerate(test_dl):
        data, target = Variable(data), Variable(target)
        output = sat_model(data)
    return loss_fn(output,target).detach().numpy(), output.detach().numpy(), target.detach().numpy()

# Train model 3 for 100 epochs
sat_nn_regresion_train = train(5)
#error_train = np.sqrt(np.dot((train_targets.detach().numpy()-sat_nn_regresion_train.detach().numpy()).transpose(),(train_targets.detach().numpy()-sat_nn_regresion_train.detach().numpy())))
sat_nn_regresion_test, output, target = test()
#error_test = np.sqrt(np.dot((test_targets.detach().numpy()-sat_nn_regresion_test.detach().numpy()).transpose(),(test_targets.detach().numpy()-sat_nn_regresion_test.detach().numpy())))

xprint('Single layer neural network training data error = '+ str(sat_nn_regresion_train) + '\n')
xprint('Single layer neural network test data error = '+ str(sat_nn_regresion_test))

# Plot data points
fig, ax = plt.subplots()
ax.plot(loss_saved)
plt.xlabel("Batch Number")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show

plt.savefig('Simple_sat_loss_All_Data.pdf') 

fig, bx = plt.subplots()
bx.plot(target[0])
bx.plot(output[0])
bx.legend(['Bx_true', 'output 1','output 2', 'output 3'])
plt.ylabel("B_x")
plt.title("Measured Bx versus Model Output")
plt.show

plt.savefig('True_vs_Model_Bx_All_Data.pdf') 

fig, by = plt.subplots()
by.plot(target[1])
by.plot(output[1])
by.legend(['By_true', 'output 1','output 2', 'output 3'])
plt.ylabel("B_y")
plt.title("Measured By versus Model Output")
plt.show

plt.savefig('True_vs_Model_By_All_Data.pdf') 

fig, bz = plt.subplots()
bz.plot(target[2])
bz.plot(output[2])
bz.legend(['Bz_true', 'output 1','output 2', 'output 3'])
plt.ylabel("B_z")
plt.title("Measured Bz versus Model Output")
plt.show

plt.savefig('True_vs_Model_Bz_All_Data.pdf') 