
# Import Numpy & PyTorch
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Read in satellite data
sat_data_df = pd.read_pickle("./sat_data/sat_data_GOES8_first_file.pkl")

def appendSpherical_np(xyz):
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2)
    #ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    ptsnew[:,1] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew


position_cart = ["x[km]", "y[km]", "z[km]"]
position_sph = ["r[km]", "theta[deg]", "phi[deg]"]
cartesian = sat_data_df[position_cart].to_numpy()
spherical = appendSpherical_np(cartesian)
spherical[:,1:3] = spherical[:,1:3] * 180/np.pi
a = -spherical[:,2]
a[a<0] = a[a<0] + 360
spherical[:,2] = a
sat_data_df[position_sph] = spherical



class SatSimpleNet_7(nn.Module):
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
    
class SatSimpleNet_6(nn.Module):
    # Initialize the layers
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(6, 50)
        self.act1 = nn.Tanh() # Activation function
        self.linear2 = nn.Linear(50, 3)
    
    # Perform the computation
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        return x

## Define the train function    
def model_train(num_epochs):
    sat_model.train()
    for epoch in range (num_epochs):
        for batch_idx, (data, target) in enumerate(train_dl):
            data, target = Variable(data), Variable(target)
            opt_n.zero_grad()
            output = sat_model(data)
            loss = loss_fn(output, target)
            loss.backward()
            opt_n.step()
            
    return loss.cpu().detach().numpy()



## Define the test function
def model_test():
    sat_model.eval()
    for batch_idx, (data, target) in enumerate(test_dl):
        data, target = Variable(data), Variable(target)
    return loss_fn(sat_model(data),target).cpu().detach().numpy()

def ave_rel_var(expected, predicted):
    error = expected.cpu().detach().numpy() - predicted.cpu().detach().numpy()
    error_var = np.var(error, axis=0)
    exp_var = np.var(expected.cpu().detach().numpy(), axis=0)
    ARV = error_var/exp_var
    return ARV
    
#inputs = ["x[km]", "y[km]", "z[km]", "vsw[km/s]", "ey[mV/m]", "imfbz[nT]", "nsw[1/cm^3]"]
inputs = ["r[km]", "theta[deg]", "phi[deg]", "vsw[km/s]", "ey[mV/m]", "imfbz[nT]", "nsw[1/cm^3]"]
field = ["bx[nT]", "by[nT]", "bz[nT]"]

pipe = Pipeline(steps=[("scaler", StandardScaler()), ("unit", MaxAbsScaler())])

preprocessor_inputs = ColumnTransformer(transformers=[("num_pos", pipe, inputs)])
preprocessor_field = ColumnTransformer(transformers=[("num_field", pipe, field)])
Inputs_trans = pd.DataFrame(preprocessor_inputs.fit_transform(sat_data_df[inputs]))
Inputs_trans.columns = inputs
Field_trans = pd.DataFrame(preprocessor_field.fit_transform(sat_data_df[field]))
Field_trans.columns = field
df_input_field_transformed = pd.concat([Inputs_trans, Field_trans], axis = 1, join = 'inner')

## Split training and test data
for n in range(8):
    prediction_efficiency = []
    for i in range(10):
        train = df_input_field_transformed.sample(frac=.7)
        test = df_input_field_transformed.drop(train.index)
        
        if n==0:
            Input_train = train[inputs]
            Field_train = train[field]
            Input_test = test[inputs]
            Field_test = test[field]
        else:
            Input_train = train[inputs].drop(inputs[n-1], axis=1)
            Field_train = train[field]
            Input_test = test[inputs].drop(inputs[n-1], axis=1)
            Field_test = test[field]   
        
        
        ## Convert to tensor
        train_inputs = torch.from_numpy(np.float32(Input_train))
        train_targets = torch.from_numpy(np.float32(Field_train))
        test_inputs = torch.from_numpy(np.float32(Input_test))
        test_targets = torch.from_numpy(np.float32(Field_test))
        
        
        train_ds = TensorDataset(train_inputs.to(device), train_targets.to(device))
        test_ds = TensorDataset(test_inputs.to(device), test_targets.to(device))
    
        # Define data loader
        batch_size = 128
        train_dl = DataLoader(train_ds, batch_size, shuffle=False)
        test_dl = DataLoader(test_ds, batch_size, shuffle=False)
    
        
        # Define model (Neural Network)
        if n == 0:
            sat_model = SatSimpleNet_7()
            sat_model.to(device)
            
        else:
            sat_model = SatSimpleNet_6()
            sat_model.to(device) 
            
        opt_n = torch.optim.Adam(sat_model.parameters(), .005)
        loss_fn = F.mse_loss
        
        ## Define the training function
        torch.manual_seed(123)
        


        # Train model for 100 epochs
        model_train(20)
       # xprint('Single layer neural network training data error = '+ str(sat_nn_regresion_train) + '\n')
       # sat_nn_regresion_test = model_test()
       # xprint('Single layer neural network test data error = '+ str(sat_nn_regresion_test))
        
        expected = torch.from_numpy(np.float32(Field_test)).to(device)
        predicted = sat_model(torch.from_numpy(np.float32(Input_test)).to(device))

        average_relative_variance = ave_rel_var(expected, predicted)
        prediction_efficiency.append(1 - average_relative_variance)
    ave_pred_eff = np.mean(prediction_efficiency, axis=0)
    if n==0:    
        xprint('Model Prediction Efficiancy All Inputs ')
        [xprint(str(j)) for j in ave_pred_eff]
    else:
        xprint('Model Predicion Efficiency Remove '+ inputs[n-1])
        [xprint(str(j)) for j in ave_pred_eff]
        





