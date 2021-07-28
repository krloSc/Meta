from evaluate.Fitness import *
from solution.Solution import *
import numpy as np
from numpy.random import rand,uniform
import matplotlib.pyplot as plt
import time
fit=Fitness()
sol=Solution()

class Pso():

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


    ############### Initial Population #################
    #init_solution=self.solution #eliminar

    ################## Evaluation ######################
    def run(self,problem):
        initime=time.time()
        current_fitness=fit.evaluate(self.solution,problem)
        best_particle=self.solution[np.argmin(current_fitness)]
        velocity=uniform(0,1,self.solution.size)
        velocity=velocity.reshape(-1,self.solution.shape[1])
        best_sol=self.solution
        for i in range(1000):
            r1=uniform(0,1,self.solution.shape[0]).reshape(-1,1)
            r2=uniform(0,1,self.solution.shape[0]).reshape(-1,1)
            velocity=0.5*velocity+r1*0.1*(best_sol-self.solution)+r2*0.5*(best_particle-self.solution)
            self.solution=sol.update_sol(self.solution,velocity) #poblacion actualizada
            #print(self.solution)
            current_fitness=fit.evaluate(self.solution,problem)
            #print(current_fitness)
            if (fit.evaluate(self.solution[np.argmin(current_fitness)],problem)<fit.evaluate(best_particle,problem)):
                best_particle=self.solution[np.argmin(current_fitness)] #
            #print(fit.evaluate(best_sol))
            best_sol[current_fitness<fit.evaluate(best_sol,problem)]=self.solution[current_fitness<fit.evaluate(best_sol,problem)]
        #print("-------------------------------- \n")
        #print(fit.evaluate(best_particle,problem))
        self.time_taken = (time.time()-initime)
        return best_particle, fit.evaluate(best_particle,problem)
