from solution.Solution import *
from problem.Problem import*
from Metaheuristics.meta import Metaheuristic
import numpy as np
from numpy.random import rand,uniform
import matplotlib.pyplot as plt
import time

sol=Solution()

class Simulated(Metaheuristic):

    def run(self,problem):
        initime=time.time()
        self.solution = sol.init_solution(self.size[0],self.size[1], problem.boundaries)
        t=self.parameters.get("to",1000)
        ta=self.parameters.get("ta",0.001)
        delta=self.parameters.get("delta",0.99)
        n=1

        while t>ta:
            n_s=5
            factor=uniform(-1,1,(n_s,self.solution.shape[1]))*100/(0.1*n) #ojoo
            print(factor)
            neigbours=sol.generate_from(self.solution,n_s,factor)

            for i in range(neigbours.shape[0]):
                current_solution = self.solution[i,:].reshape(1,2)
                current_fitness=problem.eval_fitness_function(current_solution)
                neigbours_fitness = problem.eval_fitness_function(neigbours[i,:,:])
                best_nbr=neigbours[i,self.best_index(neigbours_fitness)]
                best_nbr_fitness = problem.eval_fitness_function(best_nbr.reshape(1,2))
                if self.comparator(best_nbr_fitness,current_fitness):
                    self.solution[i,:]=best_nbr
                else:
                    r=rand()
                    l=current_fitness-best_nbr_fitness
                    ann=np.exp(-l/t)
                    if (r<ann):
                        self.solution[i,:]=best_nbr
                fitness_list = problem.eval_fitness_function(self.solution)
                self.lines.append(self.best_value(fitness_list))
            t=t*delta
            n+=1
        self.time_taken = (time.time()-initime)
        fitness_list = problem.eval_fitness_function(self.solution)
        best_solution = self.solution[self.best_index(fitness_list)]
        best_solution_fitness = self.best_value(fitness_list)
        return  best_solution , best_solution_fitness
