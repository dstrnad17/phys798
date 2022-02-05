# -*- coding: utf-8 -*-

# imported the requests library
import requests
import numpy as np
import pandas as pd
from pandas.compat import StringIO

data = "https://spdf.gsfc.nasa.gov/pub/data/aaa_special-purpose-datasets/empirical-magnetic-field-modeling-database-with-TS07D-coefficients/database/ascii/cluster1_2001_avg_300_omni.dat"
  
# data file
r = requests.get(data) # create HTTP response object
  
# send a HTTP request to the server and save
# the HTTP response in a response object called r
with open("cluster1_2001_avg_300_omni.dat",'wb') as f:
  
  
    # write the contents of the response (r.content)
    # to a new file in binary mode.
    f.write(r.content)
    
cluster_1_2001 = np.loadtxt( 'cluster1_2001_avg_300_omni.dat', unpack=True )