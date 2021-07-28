import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import math

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


#temp, nontemp = get_map("PVOUT")
#GHI, nonghi=  get_map("PVOUT")
#GHI = temp.replace(nonghi,-math.inf)
#tc = temp + sa(5/0.8)
#i = GHI*(9.07+)

#print(sys.getsizeof(ASCfile)/(1024*1024)," MB")
#print(ASCfile[600][800])
#print(column,rows,x_left,y_below,cellsize,nondata, ASCfile.shape)
#X = np.arange(0, column, 1)
#Y = np.arange(0, rows, 1)
#X,Y=np.meshgrid(X,Y)
#fig,ax=plt.subplots(1,1)
#ax.contourf(X,Y, ASCfile,10)
#plt.show()
