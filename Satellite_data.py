# -*- coding: utf-8 -*-

# import pandas as pd
import pandas as pd
import logging, sys, os


# import csv module
import csv


from os import listdir
from os.path import isfile, join


logging.basicConfig(filename='Satellite_data.log', encoding='utf-8', level=logging.DEBUG)

# Create empty file
logfile = os.path.realpath(__file__)[0:-2] + "log"
with open(logfile, "w") as f: pass


def xprint(msg):
    print(msg)
    f = open(logfile, "a")
    f.write(msg + "\n")
    f.close()

mypath = "C:/Users/erikj/OneDrive/Documents/Spring 2022/Dr Weigel/ascii/"


files = [f for f in listdir(mypath) if isfile(join(mypath, f))]

names = pd.DataFrame(pd.read_csv(mypath + files[0], nrows=0, delimiter=','))
main_dataframe = pd.DataFrame(pd.read_csv(mypath + files[0], skiprows=1, header=None, delimiter='\s+'))
main_dataframe.columns = names.columns

  

for i in range(1,3):
    data = pd.read_csv(mypath + files[i], skiprows=1, header=None, delimiter='\s+')
    df = pd.DataFrame(data)
    df.columns = names.columns
    main_dataframe = pd.concat([main_dataframe,df])

xprint(str(main_dataframe.head()))