# -*- coding: utf-8 -*-

# import pandas as pd
import pandas as pd

# import csv module
import csv


from os import listdir
from os.path import isfile, join

mypath = "C:/Users/erikj/OneDrive/Documents/Spring 2022/Dr Weigel/ascii"


files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

main_dataframe = pd.DataFrame(pd.read_csv(files[0]))	
  
#for i in range(1,len(files)):
for i in range(2):
    data = pd.read_table(files[i])
    df = pd.DataFrame(data)
    main_dataframe = pd.concat([main_dataframe,df],axis=1)