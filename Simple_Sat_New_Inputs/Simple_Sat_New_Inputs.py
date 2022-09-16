# Import Numpy & PyTorch
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline
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

# Read in satellite data
sat_data_df = pd.read_pickle("./sat_data/sat_data_first_file.pkl")



inputs = ["x[km]", "y[km]", "z[km]", "vsw[km/s]", "ey[mV/m]", "imfbz[nT]", "nsw[1/cm^3]"]
field = ["bx[nT]", "by[nT]", "bz[nT]"]


pipe = Pipeline(steps=[("scaler", StandardScaler()), ("unit", MaxAbsScaler())])

preprocessor_inputs = ColumnTransformer(transformers=[("num_pos", pipe, inputs)])
preprocessor_field = ColumnTransformer(transformers=[("num_field", pipe, field)])
Inputs_trans = pd.DataFrame(preprocessor_inputs.fit_transform(sat_data_df[inputs]))
Inputs_trans.columns = inputs
Field_trans = pd.DataFrame(preprocessor_field.fit_transform(sat_data_df[field]))
Field_trans.columns = field
df_input_field_transformed = pd.concat([Inputs_trans, Field_trans], axis = 1, join = 'inner')


class SatSimpleNet(nn.Module):
    # Initialize the layers
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(7, 50)
        self.act1 = nn.Tanh() # Activation function
        self.linear2 = nn.Linear(50, 3)
    
    # Perform the computation
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        return x

## Split training and test data

train, test = train_test_split(df_input_field_transformed, test_size=0.20)
Input_train = train[["x[km]", "y[km]", "z[km]", "vsw[km/s]", "ey[mV/m]", "imfbz[nT]", "nsw[1/cm^3]"]]
Field_train = train[["bx[nT]", "by[nT]", "bz[nT]"]]
Input_test = test[["x[km]", "y[km]", "z[km]", "vsw[km/s]", "ey[mV/m]", "imfbz[nT]", "nsw[1/cm^3]"]]
Field_test = test[["bx[nT]", "by[nT]", "bz[nT]"]]

## Convert to tensor
train_inputs = torch.from_numpy(np.float32(Input_train))
train_targets = torch.from_numpy(np.float32(Field_train))
test_inputs = torch.from_numpy(np.float32(Input_test))
test_targets = torch.from_numpy(np.float32(Field_test))


train_ds = TensorDataset(train_inputs, train_targets)
test_ds = TensorDataset(test_inputs, test_targets)

# Define data loader
batch_size = 128
train_dl = DataLoader(train_ds, batch_size, shuffle=False)
test_dl = DataLoader(test_ds, batch_size, shuffle=False)

    
# Define model (Neural Network)
sat_model = SatSimpleNet()
opt_n = torch.optim.Adam(sat_model.parameters(), .005)
loss_fn = F.mse_loss

## Define the training function
np.random.seed(123)
torch.manual_seed(123)
training_loss_saved = []
test_loss = []
def train(num_epochs):
    sat_model.train()
    inter_loss = 0
    for epoch in range (num_epochs):
        for batch_idx, (data, target) in enumerate(train_dl):
            result = 0
            data, target = Variable(data), Variable(target)
            opt_n.zero_grad()
            output = sat_model(data)
            loss = loss_fn(output, target)
            xprint('Training iteration loss = '+ str(loss.detach().numpy()) + '\n')
            loss.backward()
            opt_n.step()
            for test_batch_idx, (data, target) in enumerate(test_dl):
                test_data, test_target = Variable(data), Variable(target)
                result = result + loss_fn(sat_model(test_data), test_target)
            inter_loss = inter_loss + loss
        inter_loss = inter_loss/len(train_dl)
        training_loss_saved.append(inter_loss.detach().numpy())
        result = result/len(test_dl)
        test_loss.append(result)
    return loss.detach().numpy()


## Define the test function
def test():
    sat_model.eval()
    output = []
    for batch_idx, (data, target) in enumerate(test_dl):
        data, target = Variable(data), Variable(target)
        output.append(sat_model(data))
    return loss_fn(sat_model(data),target).detach().numpy(), output

# Train model for 100 epochs
sat_nn_regresion_train = train(100)
sat_nn_regresion_test, output = test()


xprint('Single layer neural network training data error = '+ str(sat_nn_regresion_train) + '\n')
xprint('Single layer neural network test data error = '+ str(sat_nn_regresion_test))

# Plot data points
test_loss_saved = ([i.detach().numpy() for i in test_loss])
fig, bx = plt.subplots()
bx.loglog(training_loss_saved)
bx.loglog(test_loss_saved)
plt.legend(['Training Loss', 'Test Loss'], loc = 'upper right')
plt.text(.5, 0, 'Single layer neural network training data error = '+ str(sat_nn_regresion_train) + '\n' + '\n', horizontalalignment='center',
     verticalalignment='bottom', transform=bx.transAxes, fontsize = 7)
plt.text(.5, 0, 'Single layer neural network test data error = '+ str(sat_nn_regresion_test) + '\n', horizontalalignment='center',
     verticalalignment='bottom', transform=bx.transAxes, fontsize = 7)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss (Seven Inputs)")
plt.show

plt.savefig('Simple_sat_loss_Seven_Inputs.pdf') 

output_array = np.concatenate([i.detach().numpy() for i in output])
title = ['output 1', 'output 2', 'output 3']
Odf = Field_test
Odf[title] = output_array
Odf = Odf.sort_index()


Odf.plot(y = ['bx[nT]','output 1'], kind = 'line')
plt.legend(['Bx_true', 'Bx_predicted'])
plt.ylabel("B_x")
plt.title("Measured Bx versus Model Prediction (Seven Inputs)")
plt.show

plt.savefig('True_vs_Model_Bx.pdf') 

Odf.plot(y = ['by[nT]','output 2'], kind = 'line')
plt.legend(['By_true', 'By_Predicted'])
plt.ylabel("B_y")
plt.title("Measured By versus Model Prediction (Seven Inputs)")
plt.show

plt.savefig('True_vs_Model_By.pdf') 

Odf.plot(y = ['bz[nT]','output 3'], kind = 'line')
plt.legend(['Bz_true', 'Bz_Predicted'])
plt.ylabel("B_z")
plt.title("Measured Bz versus Model Prediction (Seven Inputs)")
plt.show

plt.savefig('True_vs_Model_Bz.pdf')