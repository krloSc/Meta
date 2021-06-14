import numpy as np
import os
class Problem():

    def __init__(self,problem):
        try:
            path=os.getcwd()
            print(path)
            file=open(path+"\\problem\\"+problem+".prob",'r')
            lst=file.read().split('\n')
            self.problem=lst[0]
            x_min=float(lst[1])
            x_max=float(lst[2])
            y_min=float(lst[3])
            y_max=float(lst[4])
            self.boundaries=x_min,x_max,y_min,y_max
        except:
            print(problem+" not found")
