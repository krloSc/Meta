from Metaheuristics import *
import numpy as np
import matplotlib.pyplot as plt
from math import floor
class Evaluate():

    def __init__(self):
        pass

    def eva (self,metas,problem,epoch=5,visual=False):
        self.metas=metas
        pos=np.ones((epoch,2*len(self.metas)))
        self.fitness=np.ones((epoch,len(self.metas)))
        porc=epoch*len(self.metas)
        p=0
        for i in range(epoch):
            n=0
            for meta, j in zip(self.metas,range(len(self.metas))):
                x,y=meta.run(problem)
                pos[i,n:n+2]=x
                self.fitness[i,j]=y
                n=n+2
                print(f"{p*100/porc:.2f}%")
                p=p+1
        print("______________________________")
        print("Metaheuristic \t Best Solution")
        print("______________________________")
        self.best_fit=np.ones((len(self.metas),1))
        for i in range(len(self.metas)):
            self.best_fit[i]=np.min(self.fitness[:,i])
            print(self.metas[i].__class__.__name__,":\t ",self.best_fit[i])

        #for i in range(len(self.metas)):
            #print(f"{np.std(self.fitness[:,i]):2f}")

        if visual:
            X = np.arange(problem.x_min, problem.x_max, 0.1)
            Y = np.arange(problem.y_min, problem.y_max, 0.1)
            X,Y=np.meshgrid(X,Y)
            Z=eval(problem.problem)
            fig,ax=plt.subplots(1,1)
            ax.contourf(X, Y, Z,100)
            ax.autoscale(False)
            for i in range(0,2*len(self.metas),2):
                ax.scatter(pos[:,i],pos[:,i+1],label=self.metas[floor(i/2)].__class__.__name__,alpha=1,zorder=1)
            ax.legend()
            plt.show()
    def analysis(self):
        print("______________________________")
        print("\tAnalysis")
        print("______________________________")
        print("Best solution:\t",np.min(self.best_fit))
        print("At:\ postion x,y" )
        print("______________________________")
        print("Metaheuristic \t Parameters \t Best solution \t std \t error")#parametrizar esto, agregar tiempo requerido, num iter, etc.
        for i in range(len(self.metas)):
            print(self.metas[i].__class__.__name__,"\t","xpos, ypos","\t",self.best_fit[i,0],"\t",np.std(self.fitness[:,i]),"error")
