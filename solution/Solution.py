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
        solutions=np.zeros((ms,int(nsolutions),dimn),dtype=float)
        for x in range(ms):
            solutions[x]=sol[x]+sol[x]*entrophy
        return solutions

    def update_sol(self,solutions,slopes):
        solutions=solutions+slopes
        return(solutions)
