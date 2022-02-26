from solution.Solution import *
from problem.Problem import*
from Metaheuristics.meta import Metaheuristic
import numpy as np
from numpy.random import rand,uniform
import matplotlib.pyplot as plt
import time

class Simulated(Metaheuristic):

    def run(self) -> tuple:
        """ Run the Simulated Annealing algorithm and return the best solution and its fitness"""

        initime=time.time()
        self.solution = self.sol.init_solution(self.size[0],self.size[1], self.problem.boundaries)
        t=self.parameters.get("to",1000)
        ta=self.parameters.get("ta",0.001)
        delta=self.parameters.get("delta",0.99)
        upper_limit = self.problem.boundaries["y_max"]
        slope = (upper_limit-5)/(t-ta)
        bias = 5-ta*slope

        while t>ta:
            n_s=5
            factor=uniform(-1,1,(n_s,self.solution.shape[1]))*(t*slope+bias) #ojoo
            neigbours=self.sol.generate_from(self.solution,n_s,factor)

            for i in range(neigbours.shape[0]):
                current_solution = self.solution[i]
                current_fitness,_=self.problem.eval_fitness_function(current_solution)
                neigbours_fitness,_ = self.problem.eval_fitness_function(neigbours[i])
                best_nbr=neigbours[i,self.best_index(neigbours_fitness)]
                best_nbr_fitness,_ = self.problem.eval_fitness_function(best_nbr.reshape(1,2))
                if self.comparator(best_nbr_fitness,current_fitness):
                    self.solution[i]=best_nbr
                else:
                    r=rand()
                    l=abs(current_fitness-best_nbr_fitness)
                    ann=np.exp(-l/t)
                    if (r<ann):
                        self.solution[i]=best_nbr
            fitness_list,_ = self.problem.eval_fitness_function(self.solution)
            self.lines.append(self.best_value(fitness_list))
            t=t*delta
        self.time_taken.append(time.time()-initime)
        fitness_list,power = self.problem.eval_fitness_function(self.solution)
        best_index = self.best_index(fitness_list)
        best_solution = self.solution[best_index]
        return  best_solution , fitness_list[best_index], power[best_index]
