import numpy as np
import os
class Problem():

    def __init__(self,problem,type):
        self.type=type
        print(self.type)
        if (self.type == "space"):
            try:
                path=os.getcwd()
                file=open(path+"\\problem\\"+problem+".prob",'r')
                lst=file.read().split('\n')
                self.problem=lst[0]
                self.x_min=float(lst[1])
                self.x_max=float(lst[2])
                self.y_min=float(lst[3])
                self.y_max=float(lst[4])
                #self.boundaries=x_min,x_max,y_min,y_max
                file.close()
            except:
                print(problem+" not found")
        elif (self.type == "raster"):
            path=os.getcwd()
            file=open(path+"\\problem\\"+problem+".ASC",'r')
            lst=file.read().split('\n')
            self.column=int(get_val(lst[0]))
            self.rows=int(get_val(lst[1]))
            self.x_left=get_val(lst[2])
            self.y_below=get_val(lst[3])
            self.cellsize=get_val(lst[4])
            self.nondata=get_val(lst[5])
            file.close()
            self.problem = pd.read_csv(path+"\\problem\\"+"PVOUT.ASC",skiprows=6,encoding="gbk",engine='python',sep=' ',delimiter=None, index_col=False,header=None,skipinitialspace=True)
