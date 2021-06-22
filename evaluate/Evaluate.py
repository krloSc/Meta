from Metaheuristics import *
import numpy as np

class Evaluate():

    def __init__(self):
        pass

    def eva (self,metas,problem,epoch=5):
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
        print(np.std(fitness[:,0]),np.std(fitness[:,1]))
