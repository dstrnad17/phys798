# In[1]:


import numpy as np

from os import path


import matplotlib.pyplot as plt
import pandas as pd
import itertools

#scikit-learn related imports
import sklearn
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# pytorch relates imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# imports from captum library
from captum.attr import LayerConductance, LayerActivation, LayerIntegratedGradients
from captum.attr import IntegratedGradients, DeepLift, GradientShap, NoiseTunnel, FeatureAblation


# ## Data loading and pre-processing

# In[2]:

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sat_data_df = pd.read_pickle("./sat_data/sat_data_first_file.pkl")

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


#inputs = ["x[km]", "y[km]", "z[km]", "vsw[km/s]", "ey[mV/m]", "imfbz[nT]", "nsw[1/cm^3]"]
inputs = ["r[km]", "theta[deg]", "phi[deg]", "vsw[km/s]", "ey[mV/m]", "imfbz[nT]", "nsw[1/cm^3]"]
#field = ["bx[nT]", "by[nT]", "bz[nT]"]
field = ["bx[nT]"]
input_long = inputs * 3
field_long = field * 21

pipe = Pipeline(steps=[("scaler", StandardScaler()), ("unit", MaxAbsScaler())])

preprocessor_inputs = ColumnTransformer(transformers=[("num_pos", pipe, inputs)])
preprocessor_field = ColumnTransformer(transformers=[("num_field", pipe, field)])
Inputs_trans = pd.DataFrame(preprocessor_inputs.fit_transform(sat_data_df[inputs]))
Inputs_trans.columns = inputs
Field_trans = pd.DataFrame(preprocessor_field.fit_transform(sat_data_df[field]))
Field_trans.columns = field
df_input_field_transformed = pd.concat([Inputs_trans, Field_trans], axis = 1, join = 'inner')


# In order to retain deterministic results, let's fix the seeds.

# In[3]:


torch.manual_seed(123)
np.random.seed(123)


# Let's use 70% of our data for training and the remaining 30% for testing.

# In[4]:

train, test = train_test_split(df_input_field_transformed, test_size=0.30, random_state=0)
Input_train = train[inputs]
Field_train = train[field]
Input_test = test[inputs]
Field_test = test[field]



# # Data Exploration

# Let's visualize dependent variable vs each independent variable in a separate plot. Apart from that we will also perform a simple regression analysis and plot the fitted line in dashed, red color.

# In[5]:


fig, axs = plt.subplots(nrows = 3, ncols=7, figsize=(30, 20))
for (ax, col, row) in itertools.zip_longest(axs.flat, input_long, field_long):
    x = df_input_field_transformed.loc[:,col]
    y = df_input_field_transformed.loc[:,row]
    pf = np.polyfit(x, y, 1)
    p = np.poly1d(pf)

    ax.plot(x, y, 'o')
    ax.plot(x, p(x),"r--")

    ax.set_title(col + ' vs ' + row )
    ax.set_xlabel(col)
    ax.set_ylabel(row)
    


# Tensorizing inputs and creating batches

# Below we tensorize input features and corresponding labels.

# In[6]:

train_inputs = torch.from_numpy(np.float32(Input_train)).to(device)
train_targets = torch.from_numpy(np.float32(Field_train)).to(device)
test_inputs = torch.from_numpy(np.float32(Input_test)).to(device)
test_targets = y_test = torch.from_numpy(np.float32(Field_test)).to(device)

# train_inputs = torch.from_numpy(np.float32(Input_train))
# train_targets = torch.from_numpy(np.float32(Field_train))
# test_inputs = torch.from_numpy(np.float32(Input_test))
# test_targets = torch.from_numpy(np.float32(Field_test))


train_ds = TensorDataset(train_inputs, train_targets)
test_ds = TensorDataset(test_inputs, test_targets)


# Defining default hyper parameters for the model.


# In[7]:


batch_size = 50
num_epochs = 200
learning_rate = 0.005
size_hidden1 = 100
size_hidden2 = 50
size_hidden3 = 10
size_hidden4 = 1

train_dl = DataLoader(train_ds, batch_size = batch_size, shuffle=False)
test_dl = DataLoader(test_ds, batch_size = batch_size, shuffle=False)

# We define a four layer neural network containing ReLUs between each linear layer. This network is slightly more complex than the standard linear regression model and results in a slightly better accuracy.

# In[8]:


class SatelliteModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(7, size_hidden1)
        self.tan1 = nn.Tanh()
        self.lin2 = nn.Linear(size_hidden1, size_hidden2)
        self.tan2 = nn.Tanh()
        self.lin3 = nn.Linear(size_hidden2, size_hidden3)
        self.tan3 = nn.Tanh()
        self.lin_last = nn.Linear(size_hidden3, size_hidden4)

    def forward(self, X):
        return self.lin_last(self.tan3(self.lin3(self.tan2(self.lin2(self.tan1(self.lin1(X)))))))
    
class SatSimpleNet_7(nn.Module):
    # Initialize the layers
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(7, 50)
        self.tan1 = nn.Tanh() # Activation function
        self.lin_last = nn.Linear(50, 1)
    
    # Perform the computation
    def forward(self, x):
        x = self.lin1(x)
        x = self.tan1(x)
        x = self.lin_last(x)
        return x


def ave_rel_var(expected, predicted):
    error = expected.cpu().detach().numpy() - predicted.cpu().detach().numpy()
    error_var = np.var(error, axis=0)
    exp_var = np.var(expected.cpu().detach().numpy(), axis=0)
    ARV = error_var/exp_var
    return ARV

# In[9]:


#model = SatelliteModel()
model = SatSimpleNet_7()
model.to(device)


# ## Train Satellite Model

# Defining the loss function that will be used for optimization.

# In[10]:


criterion = nn.MSELoss(reduction='sum')


# Defining the training function that contains the training loop and uses RMSprop and given input hyper-parameters to train the model defined in the cell above.

# In[11]:


def train(num_epochs):
    model.train()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for features, labels in train_dl:
            # forward pass
            outputs = model(features)
            # defining loss
            loss = criterion(outputs, features)
            # zero the parameter gradients
            optimizer.zero_grad()
            # computing gradients
            loss.backward()
            # accumulating running loss
            running_loss += loss.item()
            # updated weights based on computed gradients
            optimizer.step()
        if epoch % 20 == 0:    
            print('Epoch [%d]/[%d] running accumulative loss across all batches: %.3f' %
                  (epoch + 1, num_epochs, running_loss))
        running_loss = 0.0
    print('done')


# If the model was previously trained and stored, we load that pre-trained model, otherwise, we train a new model and store it for future uses.
# 


# In[12]:


def train_load_save_model(model_obj, model_path):
    if path.isfile(model_path):
        # load model
        print('Loading pre-trained model from: {}'.format(model_path))
        model_obj.load_state_dict(torch.load(model_path))
    else:    
        # train model
        train(model_obj)
        print('Finished training the model. Saving the model to the path: {}'.format(model_path))
        torch.save(model_obj.state_dict(), model_path)


# # In[13]:


# SAVED_MODEL_PATH = 'models/boston_model.pt'
# train_load_save_model(model, SAVED_MODEL_PATH)


# Let's perform a simple sanity check and compute the performance of the model using Root Squared Mean Error (RSME) metric.

# In[14]:

train(num_epochs)

model.eval()
outputs = model(test_inputs)
err = np.sqrt(mean_squared_error(outputs.cpu().detach().numpy(), test_targets.cpu().detach().numpy()))

expected = torch.from_numpy(np.float32(Field_test)).to(device)
predicted = model(torch.from_numpy(np.float32(Input_test)).to(device))

average_relative_variance = ave_rel_var(expected, predicted)
prediction_efficiency = (1 - average_relative_variance)

print('model err: ', err)
print('prediction efficiency: ', prediction_efficiency)


# Comparing different attribution algorithms

# Let's compute the attributions with respect to the inputs of the model using different attribution algorithms from core `Captum` library and visualize those attributions. We use test dataset defined in the cells above for this purpose.

# We use mainly default settings, such as default baselines, number of steps etc., for all algorithms, however you are welcome to play with the settings. For GradientSHAP specifically we use the entire training dataset as the distribution of baselines.

# Note: Please, be patient! The execution of the cell below takes half a minute.

# In[15]:


ig = IntegratedGradients(model)
ig_nt = NoiseTunnel(ig)
dl = DeepLift(model)
gs = GradientShap(model)
fa = FeatureAblation(model)
print('step 1')
ig_attr_test = ig.attribute(test_inputs, n_steps=50)
print('step 2')
ig_nt_attr_test = ig_nt.attribute(test_inputs)
print('step 3')
dl_attr_test = dl.attribute(test_inputs)
print('step 4')
gs_attr_test = gs.attribute(test_inputs, train_inputs)
print('step 5')
fa_attr_test = fa.attribute(test_inputs)
print('done')


# Now let's visualize attribution scores with respect to inputs (using test dataset) for our simple model in one plot. This will help us to understand how similar or different the attribution scores assigned from different attribution algorithms are. Apart from that we will also compare attribution scores with the learned model weights.
# 
# It is important to note the we aggregate the attributions across the entire test dataset in order to retain a global view of feature importance. This, however, is not ideal since the attributions can cancel out each other when we aggregate then across multiple samples.

# In[16]:


# prepare attributions for visualization

x_axis_data = np.arange(test_inputs.shape[1])
x_axis_data_labels = list(map(lambda idx: inputs[idx], x_axis_data))

ig_attr_test_sum = ig_attr_test.cpu().detach().numpy().sum(0)
ig_attr_test_norm_sum = ig_attr_test_sum / np.linalg.norm(ig_attr_test_sum, ord=1)

ig_nt_attr_test_sum = ig_nt_attr_test.cpu().detach().numpy().sum(0)
ig_nt_attr_test_norm_sum = ig_nt_attr_test_sum / np.linalg.norm(ig_nt_attr_test_sum, ord=1)

dl_attr_test_sum = dl_attr_test.cpu().detach().numpy().sum(0)
dl_attr_test_norm_sum = dl_attr_test_sum / np.linalg.norm(dl_attr_test_sum, ord=1)

gs_attr_test_sum = gs_attr_test.cpu().detach().numpy().sum(0)
gs_attr_test_norm_sum = gs_attr_test_sum / np.linalg.norm(gs_attr_test_sum, ord=1)

fa_attr_test_sum = fa_attr_test.cpu().detach().numpy().sum(0)
fa_attr_test_norm_sum = fa_attr_test_sum / np.linalg.norm(fa_attr_test_sum, ord=1)

lin_weight = model.lin1.weight[0].cpu().detach().numpy()
y_axis_lin_weight = lin_weight / np.linalg.norm(lin_weight, ord=1)

width = 0.14
legends = ['Int Grads', 'Int Grads w/SmoothGrad','DeepLift', 'GradientSHAP', 'Feature Ablation', 'Weights']

plt.figure(figsize=(20, 10))

ax = plt.subplot()
ax.set_title('Comparing input feature importances across multiple algorithms and learned weights')
ax.set_ylabel('Attributions')

FONT_SIZE = 16
plt.rc('font', size=FONT_SIZE)            # fontsize of the text sizes
plt.rc('axes', titlesize=FONT_SIZE)       # fontsize of the axes title
plt.rc('axes', labelsize=FONT_SIZE)       # fontsize of the x and y labels
plt.rc('legend', fontsize=FONT_SIZE - 4)  # fontsize of the legend

ax.bar(x_axis_data, ig_attr_test_norm_sum, width, align='center', alpha=0.8, color='#eb5e7c')
ax.bar(x_axis_data + width, ig_nt_attr_test_norm_sum, width, align='center', alpha=0.7, color='#A90000')
ax.bar(x_axis_data + 2 * width, dl_attr_test_norm_sum, width, align='center', alpha=0.6, color='#34b8e0')
ax.bar(x_axis_data + 3 * width, gs_attr_test_norm_sum, width, align='center',  alpha=0.8, color='#4260f5')
ax.bar(x_axis_data + 4 * width, fa_attr_test_norm_sum, width, align='center', alpha=1.0, color='#49ba81')
ax.bar(x_axis_data + 5 * width, y_axis_lin_weight, width, align='center', alpha=1.0, color='grey')
ax.autoscale_view()
plt.tight_layout()

ax.set_xticks(x_axis_data + 0.5)
ax.set_xticklabels(x_axis_data_labels)

plt.legend(legends, loc=3)
plt.show()

plt.savefig('Cluster1_Feature_Importance.pdf')

# The magnitudes of learned model weights tell us about the correlations between the dependent variable `Price` and each independent variable. Zero weight means no correlation whereas positive weights indicate positive correlations and negatives the opposite. Since the network has more than one layer these weights might not be directly correlated with the price.
# 
# From the plot above we can see that attribution algorithms sometimes disagree on assigning importance scores and that they are not always aligned with weights. However, we can still observe that the top important three features: `LSTAT`, `RM` and `PTRATIO` are also considered to be important based on both most attribution algorithms and the weight scores.
# 
# It is interesting to observe that the feature `B` has high positive attribution score based on some of the attribution algorithms. This can be related, for example, to the choice of the baseline. In this tutorial we use zero-valued baselines for all features, however if we were to choose those values more carefully for each feature the picture will change. Similar arguments apply also when the signs of the weights and attributions mismatches or when one algorithm assigns higher or lower attribution scores compare to the others.
# 
# In terms of least important features, we observe that `CHAS` and `RAD` are voted to be least important both based on most attribution algorithms and learned coefficients.
# 
# Another interesting observation is that both Integrated Gradients and DeepLift return similar attribution scores across all features. This is associated with the fact that although we have non-linearities in our model, their effects aren't significant and DeepLift is close to `(input - baselines) * gradients`. And because the gradients do not change significantly along the straight line from baseline to input, we observe similar situation with Integrated Gradients as well.

# ## Attributing to the layers and comparing with model weights

# Now let's beside attributing to the inputs of the model, also attribute to the layers of the model and understand which neurons appear to be more important.
# 
# In the cell below we will attribute to the inputs of the second linear layer of our model. Similar to the previous case, the attribution is performed on the test dataset.

# In[21]:


# Compute the attributions of the output with respect to the inputs of the fourth linear layer
lc = LayerConductance(model, model.lin_last)

# shape: test_examples x size_hidden
lc_attr_test = lc.attribute(test_inputs, n_steps=100, attribute_to_layer_input=True)

# weights from forth linear layer
# shape: size_hidden4 x size_hidden3
lin_last_weight = model.lin_last.weight


# In the cell below we normalize and visualize the attributions and learned model weights for all 10 neurons in the fourth hidden layer. 
# The weights represent the weight matrix of the fourth linear layer. The attributions are computed with respect to the inputs of the fourth linear layer.

# In[22]:


plt.figure(figsize=(15, 8))

x_axis_data = np.arange(lc_attr_test.shape[1])

y_axis_lc_attr_test = lc_attr_test.mean(0).cpu().detach().numpy()
y_axis_lc_attr_test = y_axis_lc_attr_test / np.linalg.norm(y_axis_lc_attr_test, ord=1)

y_axis_lin_last_weight = lin_last_weight[0].cpu().detach().numpy()
y_axis_lin_last_weight = y_axis_lin_last_weight / np.linalg.norm(y_axis_lin_last_weight, ord=1)

width = 0.25
legends = ['Attributions','Weights']
x_axis_labels = [ 'Neuron {}'.format(i) for i in range(len(y_axis_lin_last_weight))]

ax = plt.subplot()
ax.set_title('Aggregated neuron importances and learned weights in the last linear layer of the model')

ax.bar(x_axis_data + width, y_axis_lc_attr_test, width, align='center', alpha=0.5, color='red')
ax.bar(x_axis_data + 2 * width, y_axis_lin_last_weight, width, align='center', alpha=0.5, color='green')
plt.legend(legends, loc=2, prop={'size': 20})
ax.autoscale_view()
plt.tight_layout()

plt.savefig('Cluster1_Last_Layer_Neuron_Importance.pdf')

#ax.set_xticks(x_axis_data + 0.5)
#ax.set_xticklabels(x_axis_labels)

plt.show()


