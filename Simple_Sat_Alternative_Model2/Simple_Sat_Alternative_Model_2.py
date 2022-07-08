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

plt.savefig('Magnetic Field Measurement Z Component') 

# Sort data by time
# sat_data_df = sat_data_df.sort_values(by='epoch', ascending=True)

position = ["x[km]", "y[km]", "z[km]"]
field = ["bz[nT]"]

pipe = Pipeline(steps=[("scaler", StandardScaler()), ("unit", MaxAbsScaler())])

preprocessor_pos = ColumnTransformer(transformers=[("num_pos", pipe, position)])
preprocessor_field = ColumnTransformer(transformers=[("num_field", pipe, field)])
Pos_trans = pd.DataFrame(preprocessor_pos.fit_transform(sat_data_df[position]))
Pos_trans.columns = position
Field_trans = pd.DataFrame(preprocessor_field.fit_transform(sat_data_df[field]))
Field_trans.columns = field
df_pos_field_transformed = pd.concat([Pos_trans, Field_trans], axis = 1, join = 'inner')


class SatSimpleNet(nn.Module):
    # Initialize the layers
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 100)
        self.act1 = nn.Tanh() # Activation function
        self.linear2 = nn.Linear(100, 1)
    
    # Perform the computation
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        return x

## Use field data as input
    
train, test = train_test_split(df_pos_field_transformed, test_size=0.20)
Position_train = train[["x[km]", "y[km]", "z[km]"]]
Field_train = train[["bz[nT]"]]
Position_test = test[["x[km]", "y[km]", "z[km]"]]
Field_test = test[["bz[nT]"]]



train_inputs = torch.from_numpy(np.float32(Position_train))
train_targets = torch.from_numpy(np.float32(Field_train))
test_inputs = torch.from_numpy(np.float32(Position_test))
test_targets = torch.from_numpy(np.float32(Field_test))

train_ds = TensorDataset(train_inputs, train_targets)
test_ds = TensorDataset(test_inputs, test_targets)

# Define data loader
batch_size = 128
train_dl = DataLoader(train_ds, batch_size, shuffle=False)
test_dl = DataLoader(test_ds, batch_size, shuffle=False)

    
# Define model (Neural Network)
sat_model = SatSimpleNet()
opt_n = torch.optim.Adam(sat_model.parameters())
loss_fn = F.mse_loss


np.random.seed(123)
torch.manual_seed(123)
training_loss_saved = []
test_loss = []
def train(num_epochs):
    sat_model.train()
    for epoch in range (num_epochs):
        for batch_idx, (data, target) in enumerate(train_dl):
            result = 0
            data, target = Variable(data), Variable(target)
            opt_n.zero_grad()
            output = sat_model(data)
            loss = loss_fn(output, target)
            training_loss_saved.append(loss.detach().numpy())
            xprint('Training iteration loss = '+ str(loss.detach().numpy()) + '\n')
            loss.backward()
            opt_n.step()
            for test_batch_idx, (data, target) in enumerate(test_dl):
                test_data, test_target = Variable(data), Variable(target)
                result = result + loss_fn(sat_model(test_data), test_target)
            result = result/len(test_dl)
            test_loss.append(result)
    return loss.detach().numpy()

def test():
    sat_model.eval()
    output = []
    for batch_idx, (data, target) in enumerate(test_dl):
        data, target = Variable(data), Variable(target)
        output.append(sat_model(data))
    return loss_fn(sat_model(data),target).detach().numpy(), output

# Train model 3 for 3 epochs
sat_nn_regresion_train = train(10)
#error_train = np.sqrt(np.dot((train_targets.detach().numpy()-sat_nn_regresion_train.detach().numpy()).transpose(),(train_targets.detach().numpy()-sat_nn_regresion_train.detach().numpy())))
sat_nn_regresion_test, output = test()
#error_test = np.sqrt(np.dot((test_targets.detach().numpy()-sat_nn_regresion_test.detach().numpy()).transpose(),(test_targets.detach().numpy()-sat_nn_regresion_test.detach().numpy())))

xprint('Single layer neural network training data error = '+ str(sat_nn_regresion_train) + '\n')
xprint('Single layer neural network test data error = '+ str(sat_nn_regresion_test))

# Plot data points
test_loss_saved = ([i.detach().numpy() for i in test_loss])
fig, ax = plt.subplots()
ax.loglog(training_loss_saved)
ax.loglog(test_loss_saved)
plt.legend(['Training Loss', 'Test Loss'], loc = 'upper right')
plt.text(.5, 0, 'Single layer neural network training data error = '+ str(sat_nn_regresion_train) + '\n' + '\n', horizontalalignment='center',
     verticalalignment='bottom', transform=ax.transAxes, fontsize = 7)
plt.text(.5, 0, 'Single layer neural network test data error = '+ str(sat_nn_regresion_test) + '\n', horizontalalignment='center',
     verticalalignment='bottom', transform=ax.transAxes, fontsize = 7)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss (Field as Input)")
plt.show

plt.savefig('Single_Output_loss_Field_Input.pdf') 

output_array = np.concatenate([i.detach().numpy() for i in output])
title = ['output 1']
Odf = Field_test
Odf[title] = output_array
Odf = Odf.sort_index() 

Odf.plot(y = ['bz[nT]','output 1'], kind = 'line')
plt.legend(['Bz_true', 'output 1'])
plt.ylabel("B_z")
plt.title("Measured Bz versus Model Output (Field as Input)")
plt.show

plt.savefig('Single_Output_True_vs_Model_Bz_Position_Input.pdf')


    
