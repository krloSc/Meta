import numpy as np
from numpy.random import randint,rand, uniform
class Solution():
    def __init__(self):
            self.sol=np.array([0,0],dtype=float)

    def init_solution(self,x,y):
            sol=uniform(-1,1,(x,y))*10
            return sol

    def generate_from(self,sol,nsolutions,entrophy):
        ms=sol.shape[0]
        dimn=sol.shape[1]
        solutions=np.zeros((ms,nsolutions,dimn),dtype=float)
        for x in range(ms):
            for y in range(nsolutions):
                for i in range(dimn):
                    try:
                        solutions[x,y,i]=(sol[x,i]+sol[x,i]*entrophy[y,i])
                        print("pass")
                    except:
                        solutions[x,y,i]=(sol[x,i]+sol[x,i]*(rand()-0.5)*entrophy)
        return solutions

    def generate_from2(self,sol,nsolutions,entrophy,force_factor=1,anchor=np.array([])):
        ms=sol.shape[0]
        dimn=sol.shape[1]
        solutions=np.zeros((ms,int(nsolutions),dimn),dtype=float)
        force=np.zeros(sol.shape)
        if anchor.shape[0]>1: #buscar emjor solucion
            dist=np.zeros(sol.shape)
            #print(anchor.shape[0])
            for x in range(anchor.shape[0]):
                dist=dist+(sol-anchor[x])**2
                #print("anchor: ",x)
                #print(dist)
            force=force_factor/dist
            #print("force:",force)

        for x in range(ms):
            solutions[x]=sol[x]+sol[x]*entrophy#*force[x]*100
        #print(solutions)
        return solutions

    def update_sol(self,solutions,slopes): #mejorar para dar mas utilidad o eliminar
        solutions=solutions+slopes
        return(solutions)
