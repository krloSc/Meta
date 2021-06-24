from Metaheuristics import *
import numpy as np
import matplotlib.pyplot as plt
from math import floor
class Evaluate():

    def __init__(self):
        pass

    def eva (self,metas,problem,epoch=5,visual=False):
        pos=np.ones((epoch,2*len(metas)))
        fitness=np.ones((epoch,len(metas)))
        porc=epoch*len(metas)
        p=0
        for i in range(epoch):
            n=0
            for meta, j in zip(metas,range(len(metas))):
                x,y=meta.run(problem)
                pos[i,n:n+2]=x
                fitness[i,j]=y
                n=n+2
                print(f"{p*100/porc:.2f}%")
                p=p+1
        print(fitness)
        for i in range(len(metas)):
            print(f"{np.std(fitness[:,i]):2f}")

        if visual:
            X = np.arange(problem.x_min, problem.x_max, 0.1)
            Y = np.arange(problem.y_min, problem.y_max, 0.1)
            X,Y=np.meshgrid(X,Y)
            Z=eval(problem.problem)
            fig,ax=plt.subplots(1,1)
            ax.contourf(X, Y, Z,100)
            ax.autoscale(False)
            for i in range(0,2*len(metas),2):
                ax.scatter(pos[:,i],pos[:,i+1],label=metas[floor(i/2)].__class__.__name__,alpha=1,zorder=1)
            ax.legend()
            plt.show()
