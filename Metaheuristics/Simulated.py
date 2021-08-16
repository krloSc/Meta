from evaluate.Fitness import *
from solution.Solution import *
import numpy as np
from numpy.random import rand,uniform
import matplotlib.pyplot as plt
import time
fit=Fitness()
sol=Solution()

class Simulated():
    def __init__(self,size,parameters=[]):
        if parameters==[]:
            try:
                path=os.getcwd()
                file=open(path+"\\Metaheuristics\\"+self.__class__.__name__+".param",'r')
                lst=file.read().split('\n')
                parameters=eval(lst[0])

            except:
                print("Parameters not found")
        self.solution=sol.init_solution(size[0],size[1])
        self.parameters=parameters

    def run(self,problem):
        initime=time.time()
        t=self.parameters.get("to",1000)
        ta=self.parameters.get("ta",0.001)
        delta=self.parameters.get("delta",0.99)
        n=1
        while t>ta:
            n_s=5
            factor=uniform(-1,1,(n_s,self.solution.shape[1]))*1/(0.1*n)
            neigbours=sol.generate_from2(self.solution,n_s,factor)
            for i in range(neigbours.shape[0]):
                current_fitness=fit.evaluate(self.solution[i,:],problem)
                best_nbr=neigbours[i,np.argmin(fit.evaluate(neigbours[i,:,:],problem))]
                if (fit.evaluate(best_nbr,problem)<current_fitness):
                    self.solution[i,:]=best_nbr
                else:
                    r=rand()
                    l=current_fitness-fit.evaluate(best_nbr,problem)
                    ann=np.exp(-l/t)
                    if (r<ann):
                        self.solution[i,:]=best_nbr
            t=t*delta
            n+=1
        self.time_taken = (time.time()-initime)
        return self.solution[np.argmin(fit.evaluate(self.solution,problem))] , np.min(fit.evaluate(self.solution,problem))
