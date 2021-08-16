import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import math

def map(value,xmin,xmax,ymin,ymax):
    result = (value-xmin)*((ymax-ymin)/(xmax-xmin))+ymin
    return result

def get_val(string):
    for word in string.split():
        try:
            value=float(word)
            break
        except:
            pass
    return value
def get_map(map):
    path=os.getcwd()
    file=open(path+"\\problem\\"+map+".ASC",'r')
    lst=file.read().split('\n')
    column=int(get_val(lst[0]))
    rows=int(get_val(lst[1]))
    x_left=get_val(lst[2])
    y_below=get_val(lst[3])
    cellsize=get_val(lst[4])
    nondata=get_val(lst[5])
    file.close()
    ASCfile = pd.read_csv(path+"\\problem\\"+map+".ASC",skiprows=6,encoding="gbk",engine='python',sep=' ',delimiter=None, index_col=False,header=None,skipinitialspace=True)
    ASCfile = ASCfile[::-1]
    return ASCfile, nondata
