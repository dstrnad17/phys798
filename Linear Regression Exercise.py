#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import Numpy & PyTorch
import torch


# In[19]:


# Define the data
a = torch.ones(3,1)
x = torch.randn(100,3)
a0 = torch.randn(100,1)
y = torch.matmul(x,a) + a0
torch.Tensor.size(y)


# In[24]:


# Define model inputs and targets and initalize weights and bias
inputs = x
targets = y
w = torch.randn(1,3, requires_grad=True)
b = torch.randn(1, requires_grad=True)


# In[25]:


# Define the model
def model(x):
    return x @ w.t() + b


# In[33]:


# Define loss function
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()


# In[64]:


# Iterate and modify via gradient decent
for i in range(100):
    preds = model(inputs)
    loss = mse(preds, targets)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()


# In[65]:


print(w)


# In[66]:


print('Training loss: ', loss)


# In[41]:


import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# In[42]:


train_ds = TensorDataset(inputs, targets)


# In[49]:


# Define data loader
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)


# In[50]:


# Define model
model2 = nn.Linear(3, 1)


# In[51]:


# Define optimizer
opt = torch.optim.SGD(model2.parameters(), lr=1e-5)


# In[52]:


# Import nn.functional
import torch.nn.functional as F


# In[54]:


# Define loss function
loss_fn = F.mse_loss
loss = loss_fn(model(inputs), targets)


# In[55]:


# Define a utility function to train the model
def fit(num_epochs, model, loss_fn, opt):
    for epoch in range(num_epochs):
        for xb,yb in train_dl:
            # Generate predictions
            pred = model(xb)
            loss = loss_fn(pred, yb)
            # Perform gradient descent
            loss.backward()
            opt.step()
            opt.zero_grad()
    print('Training loss: ', loss_fn(model(inputs), targets))


# In[67]:


# Train the model for 100 epochs
fit(100, model2, loss_fn, opt)


# In[69]:


class SimpleNet(nn.Module):
    # Initialize the layers
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(3, 3)
        self.act1 = nn.ReLU() # Activation function
        self.linear2 = nn.Linear(3, 1)
    
    # Perform the computation
    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        return x


# In[70]:


model3 = SimpleNet()
opt = torch.optim.SGD(model3.parameters(), 1e-5)
loss_fn = F.mse_loss


# In[71]:


fit(100, model3, loss_fn, opt)


# In[ ]:




