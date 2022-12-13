# In[1]:
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
import glob



# Import nn.functional
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

# Create empty file
logfile = os.path.realpath(__file__)[0:-2] + "log"
with open(logfile, "w") as f: pass


def xprint(msg):
    print(msg)
    f = open(logfile, "a")
    f.write(msg + "\n")
    f.close()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize Tensorboard summary writer
tb = SummaryWriter()


# In[2]:
    
# Define 7 input Neural Network model with one hidden layer of 50 neurons and 3 outputs
class SatSimpleNet_7(nn.Module):
    # Initialize the layers
    def __init__(self):
    # Initialize the layers
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

# Define 7 input Linear model with 3 outputs
class Linear_model_7(nn.Module):
    # Initialize the layers
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(7, 3)
    
    # Perform the computation
    def forward(self, x):
        return self.linear1(x)

# Define 6 input Neural Network model with one hidden layer of 50 neurons and 3 outputs    
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

# Define 6 input Linear model with 3 outputs    
class Linear_model_6(nn.Module):
    # Initialize the layers
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(6, 3)
    
    # Perform the computation
    def forward(self, x):
        return self.linear1(x)

## Define the train function    
def model_train(model, num_epochs):
    model.train()
    for epoch in range (num_epochs):
        loss_sum = 0
        for batch_idx, (data, target) in enumerate(train_dl):
            data, target = Variable(data), Variable(target)
            opt_n.zero_grad()
            output = sat_model(data)
            loss = loss_fn(output, target)
            loss_sum += loss
            loss.backward()
            opt_n.step()
        
        # Save training and test losses at each epoch to Tensorboard
        test_loss, pred_effic = model_test()
        tb.add_scalar("Training Loss", loss_sum/batch_idx, epoch)
        tb.add_scalar("Test Loss", test_loss, epoch)
        tb.add_scalars("Prediction Efficiency", pred_effic, epoch)
        
            
    return loss.cpu().detach().numpy()


## Define the test function
def model_test():
    sat_model.eval()
    predicted = sat_model(test_inputs)
    test_loss = loss_fn(predicted,test_targets)
    efficiency = 1-ave_rel_var(test_targets, predicted)
    scalars = {
    'scalar_1': round(efficiency[0], 3),
    'scalar_2': round(efficiency[1], 3),
    'scalar_3': round(efficiency[2], 3)
    }
    # for batch_idx, (data, target) in enumerate(test_dl):
    #     data, target = Variable(data), Variable(target)
    return test_loss, scalars
# .cpu().detach().numpy()

def ave_rel_var(expected, predicted):
    error = expected.cpu().detach().numpy() - predicted.cpu().detach().numpy()
    error_var = np.var(error, axis=0)
    exp_var = np.var(expected.cpu().detach().numpy(), axis=0)
    ARV = error_var/exp_var
    return ARV

def appendSpherical_np(xyz):
    ptsnew = np.zeros(xyz.shape)
    xy = xyz[:,0]**2 + xyz[:,1]**2
    ptsnew[:,0] = np.sqrt(xy + xyz[:,2]**2)
    #ptsnew[:,4] = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    ptsnew[:,1] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:,2] = np.arctan2(xyz[:,1], xyz[:,0])
    return ptsnew
    
# In[3]:   


inputs = ["r[km]", "theta[deg]", "phi[deg]", "vsw[km/s]", "ey[mV/m]", "imfbz[nT]", "nsw[1/cm^3]"]
field = ["bx[nT]", "by[nT]", "bz[nT]"]
position_cart = ["x[km]", "y[km]", "z[km]"]
position_sph = ["r[km]", "theta[deg]", "phi[deg]"]
    
for file in glob.glob("./generated_pkl_files/*.pkl"):
    # Read in satellite data
    sat_data_df = pd.read_pickle(file)
    name = os.path.splitext(os.path.basename(file))[0]
    
    # Convert from cartesion to spherical coordinates and store in dataframe
    cartesian = sat_data_df[position_cart].to_numpy()
    spherical = appendSpherical_np(cartesian)
    spherical[:,1:3] = spherical[:,1:3] * 180/np.pi
    a = -spherical[:,2]
    a[a<0] = a[a<0] + 360
    spherical[:,2] = a
    sat_data_df[position_sph] = spherical

    # Plot spherical coordinates over time
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle('Position vs Time')
    sat_data_df.plot(y = ['r[km]'], kind = 'line', ax = ax1, ylabel = 'Radius', legend=False)
    sat_data_df.plot(y = ['theta[deg]'], kind = 'line', ax = ax2, ylabel = 'Theta', legend=False)
    sat_data_df.plot(y = ['phi[deg]'], kind = 'line', ax = ax3, ylabel = 'Phi', legend=False)
    plt.xlabel('Measurement Sample (5 minute interval)')
    plt.savefig('./plots/' + name + '_Position_Spherical.pdf')
    plt.show()
    
    # Plot magnetic field over time
    fig, (bx1, bx2, bx3) = plt.subplots(3, sharex=True)
    fig.suptitle('Magenetic Field vs Time')
    sat_data_df["bx[nT]"].plot(kind = 'line', ax = bx1, ylabel = 'Bx [nT]')
    sat_data_df["by[nT]"].plot(kind = 'line', ax = bx2, ylabel = 'By [nT]')
    sat_data_df["bz[nT]"].plot(kind = 'line', ax = bx3, ylabel = 'Bz [nT]')
    plt.xlabel('Measurement Sample (5 minute interval)')
    plt.savefig('./plots/'+ name + '_Magnetic_Field.pdf')
    plt.show()

    # Plot Interplanetary magnetic field over time
    fig, (cx1, cx2, cx3) = plt.subplots(3, sharex=True)
    fig.suptitle('Interplanetary Magnetic Field vs Time')
    sat_data_df["imfbx[nT]"].plot(kind = 'line', ax = cx1, ylabel = 'IMFBx [nT]')
    sat_data_df["imfby[nT]"].plot(kind = 'line', ax = cx2, ylabel = 'IMFBy [nT]')
    sat_data_df["imfbz[nT]"].plot(kind = 'line', ax = cx3, ylabel = 'IMFBz [nT]')
    plt.xlabel('Measurement Sample (5 minute interval)')
    plt.savefig('./plots/'+ name + '_IP_Magnetic_Field.pdf')
    plt.show()

    # Plot Solar wind velociy over time
    fig, dx = plt.subplots()
    fig.suptitle('Solar Wind Velocity vs Time')
    sat_data_df["vsw[km/s]"].plot(kind = 'line', ax = dx, ylabel = 'Bx [nT]')
    plt.xlabel('Measurement Sample (5 minute interval)')
    plt.savefig('./plots/'+ name + '_Solar_Wind_Velocity.pdf')
    plt.show()

    # Plot NSW over Time
    fig, ex = plt.subplots()
    fig.suptitle('NSW vs Time')
    sat_data_df["nsw[1/cm^3]"].plot(kind = 'line', ax = ex, ylabel = 'NSW [1/cm^3]')
    plt.savefig('./plots/'+ name + '_NSW.pdf')
    plt.show()

    # Plot Electric Field over time
    fig, fx = plt.subplots()
    fig.suptitle('Ey vs Time')
    sat_data_df["ey[mV/m]"].plot(kind = 'line', ax = fx, ylabel = 'EY [mV/m]')
    plt.xlabel('Measurement Sample (5 minute interval)')
    plt.savefig('./plots/'+ name + '_Electric_Field_y.pdf')
    plt.show()

    # In[4]:    
    
    # Define a pipeline to scale and transform data
    pipe = Pipeline(steps=[("scaler", StandardScaler()), ("unit", MaxAbsScaler())])
    
    # Scale and transform data and save to a new dataframe
    preprocessor_inputs = ColumnTransformer(transformers=[("num_pos", pipe, inputs)])
    preprocessor_field = ColumnTransformer(transformers=[("num_field", pipe, field)])
    Inputs_trans = pd.DataFrame(preprocessor_inputs.fit_transform(sat_data_df[inputs]))
    Inputs_trans.columns = inputs
    Field_trans = pd.DataFrame(preprocessor_field.fit_transform(sat_data_df[field]))
    Field_trans.columns = field
    df_input_field_transformed = pd.concat([Inputs_trans, Field_trans], axis = 1, join = 'inner')
    
    # Initialize variables outside the loops
    ave_pred_eff = []
    model_label = ['All Inputs']
    ave_pred_eff_lin = []
    model_label_lin = ['All Inputs Linear']
    
    # Define loops to train different iterations of the model removing one variable at a time
    for n in range(8):
        prediction_efficiency = []
        prediction_efficiency_lin = []
        
        # Train model multiple (10) times to get an average result (number of iterations can be reduced)
        for i in range(10):
            # Split training and test data
            train = df_input_field_transformed.sample(frac=.8)
            test = df_input_field_transformed.drop(train.index)
            
            # Define input and target training and test data 
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
            train_inputs = torch.from_numpy(np.float32(Input_train)).to(device)
            train_targets = torch.from_numpy(np.float32(Field_train)).to(device)
            test_inputs = torch.from_numpy(np.float32(Input_test)).to(device)
            test_targets = torch.from_numpy(np.float32(Field_test)).to(device)
            
            # Combine the input and target tensors
            train_ds = TensorDataset(train_inputs, train_targets)
            test_ds = TensorDataset(test_inputs, test_targets)
        
            # Define data loaders
            batch_size = 256
            train_dl = DataLoader(train_ds, batch_size, shuffle=False)
            test_dl = DataLoader(test_ds, batch_size, shuffle=False)
        
            
            # Define models, n= 0 corresponds to the 7 input models with all inputs
            if n == 0:
                sat_model = SatSimpleNet_7()
                sat_model.to(device)
                sat_model_lin = Linear_model_7()
                sat_model_lin.to(device)
                
            else:
                sat_model = SatSimpleNet_6()
                sat_model.to(device)
                sat_model_lin = Linear_model_6()
                sat_model_lin.to(device) 
            
            # Define the optimizer and loss functions to be used for backpropogating the model
            opt_n = torch.optim.Adam(sat_model.parameters(), .005)
            loss_fn = F.mse_loss
            
            ## Define the training function
            torch.manual_seed(123)
    
            # Train NN model
            model_train(sat_model, 20)
            
            # Close out Tensorboard
            tb.close()
            
            # Run Linear Model
            model_train(sat_model_lin, 20)
            
            # Track training progress
            xprint(name + ', model = {}, iteration = {}'.format(n, i+1))
            
            # Calculate predicted efficiency for the different models
            expected = test_targets
            predicted = sat_model(test_inputs)
            predicted_lin = sat_model_lin(test_inputs)
            
            average_relative_variance = ave_rel_var(expected, predicted)
            average_relative_variance_lin = ave_rel_var(expected, predicted_lin)
            prediction_efficiency.append(1 - average_relative_variance)
            prediction_efficiency_lin.append(1 - average_relative_variance_lin)
        ave_pred_eff.append(np.round_(np.mean(prediction_efficiency, axis=0), decimals = 3))
        ave_pred_eff_lin.append(np.round_(np.mean(prediction_efficiency_lin, axis=0), decimals = 3))
        
        # Prepare results for plotting
        if n > 0:
            model_label.append('Remove_'+ inputs[n-1])
            model_label_lin.append('Remove '+ inputs[n-1])
        
        dat = expected.cpu().detach().numpy()
        dat = np.concatenate((dat, predicted.cpu().detach().numpy()), axis=1)
        col = ['Measured_Bx','Measured_By','Measured_Bz', 'NN Prediction_Bx',
               'NN Prediction_By', 'NN Prediction_Bz']
        prediction_df = pd.DataFrame(dat, columns = col)
        
        # Plot measured vs predicted magnetic field
        fig, (gx1, gx2, gx3) = plt.subplots(3, sharex=True)
        fig.suptitle(name + 'Magnetic Field Measured vs Predicted' + '\n {}'.format(model_label[n]) )
        prediction_df.plot(y = ['Measured_Bx', 'NN Prediction_Bx'], 
                           kind = 'line', ax = gx1, ylabel = 'Bx', legend=False)
        prediction_df.plot(y = ['Measured_By', 'NN Prediction_By'],
                           kind = 'line', ax = gx2, ylabel = 'By', legend=False)
        prediction_df.plot(y = ['Measured_By', 'NN Prediction_By'], 
                           kind = 'line', ax = gx3, ylabel = 'Bz', legend=False)
        plt.xlabel('Test Measurement Sample')
        fig.legend(['Measured', 'Predicted'], loc='lower right')
        plt.savefig('./plots/' + name + '_Measured_vs_Predicted_model_{}.pdf'.format(n))
        plt.show()
    
    # Prepare results for tabular printing
    model_df = pd.DataFrame(model_label, columns = ['Model NN'])
    model_df[['Bx', 'By', 'Bz']] = ave_pred_eff
    model_lin_df = pd.DataFrame(model_label_lin, columns = ['Model Linear'])
    model_lin_df[['Bx', 'By', 'Bz']] = ave_pred_eff_lin
    
    # Print tables
    xprint('\n')
    xprint('\n')
    xprint(name)
    xprint('\n')
    xprint(model_df.to_markdown())
    xprint('\n')
    xprint(model_lin_df.to_markdown())
    xprint('\n')
    xprint('\n')

