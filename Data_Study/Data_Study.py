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

#load data from Pickle File
sat_data_df = pd.read_pickle("./sat_data/sat_data.pkl")
#Sort data by epoch(time)
sat_data_df = sat_data_df.sort_values(by='epoch', ascending=True)


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


radius_hist = sat_data_df.hist(column = "r[km]", range = (0,.22e6), bins=75)
plt.savefig('Radius_Histogram.pdf')

angle_hist = sat_data_df.hist(column = ["theta[deg]", "phi[deg]"], bins=180)
plt.savefig('Angle_Histograms.pdf')

# fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))
# rad_phi_hist = ax.hist2d(spherical[:,2],spherical[:,0], bins=[20,50], range = [(0,360), (0,.15e6)])


# define binning
rbins = np.linspace(0,.08e6, 40)
abins = np.linspace(0,2*np.pi, 180)

rad_phi_hist, _, _ = np.histogram2d(spherical[:,2], spherical[:,0], bins=(abins, rbins))
A, R = np.meshgrid(abins, rbins)

# plot
fig, ax = plt.subplots(subplot_kw=dict(projection="polar"))

pc = ax.pcolormesh(A, R, rad_phi_hist.T, cmap="magma_r")
fig.colorbar(pc)
plt.title("Measurement Distribution by Radius and Azimuth")

plt.show()

plt.savefig('Radius_Azimuth_Histogram.pdf')




