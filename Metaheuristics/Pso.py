from solution.Solution import *
from problem.Problem import*
import numpy as np
from numpy.random import rand,uniform
import matplotlib.pyplot as plt
import time

sol=Solution()

class Pso():

    def __init__(self,size, optimization: OptimizationType, parameters={}):
        if parameters==[]:
            try:
                path=os.getcwd()
                file=open(path+"\\Metaheuristics\\"+self.__class__.__name__+".param",'r')
                lst=file.read().split('\n')
                parameters=eval(lst[0])

            except:
                print("Parameters not found")
        self.size = size
        self.parameters=parameters

        if optimization == OptimizationType.MINIMIZATION:
            self.comparator = np.less
            self.better_index = np.argmin
            self.best_value = min

        else:
            self.comparator = np.greater
            self.better_index = np.argmax
            self.best_value = max

    def update_velocity(self, prev_velocity, best_sol, best_particle, r1, r2):

        velocity = (
                    0.5*prev_velocity
                    + r1*0.1*(best_sol-self.solution)
                    + r2*0.5*(best_particle-self.solution))
        return velocity

    def run(self,problem):

        init_time=time.time()
        self.solution=sol.init_solution(
                                        self.size[0],
                                        self.size[1],
                                        problem.boundaries
                                        )

        current_fitness=problem.eval_fitness_function(self.solution)
        best_particle=self.solution[self.better_index(current_fitness)]
        velocity = uniform(0,1,self.solution.size)
        velocity = velocity.reshape(-1,self.solution.shape[1])
        best_sol = self.solution

        for i in range(500):

            r1 = uniform(0,1,self.solution.shape[0]).reshape(-1,1)
            r2 = uniform(0,1,self.solution.shape[0]).reshape(-1,1)
            velocity = self.update_velocity(
                                            velocity,
                                            best_sol,
                                            best_particle,
                                            r1,
                                            r2
                                            )
            self.solution=sol.update_sol(self.solution,velocity)
            current_fitness = problem.eval_fitness_function(self.solution)
            current_best_fitness = self.best_value(current_fitness)
            best_particle_fitness = problem.eval_fitness_function(best_particle)

            if self.comparator(current_best_fitness, best_particle_fitness):
                best_particle=self.solution[self.better_index(current_fitness)]

            previous_fitness = problem.eval_fitness_function(best_sol)
            index_mask = self.comparator(current_fitness, previous_fitness)
            best_sol[index_mask] = self.solution[index_mask]

        self.time_taken = (time.time()-init_time)
        return best_particle, problem.eval_fitness_function(best_particle)
