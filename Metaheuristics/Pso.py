from solution.Solution import *
from problem.Problem import*
from Metaheuristics.meta import Metaheuristic
import numpy as np
from numpy.random import rand,uniform
import matplotlib.pyplot as plt
import time

sol=Solution()

class Pso(Metaheuristic):

    def update_velocity(self, prev_velocity, best_sol, best_particle, r1, r2):

        velocity = (
                    0.1*prev_velocity
                    + r1*1.5*(best_sol-self.solution)
                    + r2*3*(best_particle-self.solution))
        #print(velocity)
        return velocity

    def run(self,problem):

        init_time=time.time()
        self.solution=sol.init_solution(
                                        self.size[0],
                                        self.size[1],
                                        problem.boundaries
                                        )

        current_fitness=problem.eval_fitness_function(self.solution)
        best_particle=self.solution[self.best_index(current_fitness)]
        velocity = uniform(0,1,self.solution.size)
        velocity = velocity.reshape(-1,self.solution.shape[1])
        best_sol = self.solution

        for i in range(100):

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
            self.lines.append(best_particle_fitness)
            if self.comparator(current_best_fitness, best_particle_fitness):
                best_particle=self.solution[self.best_index(current_fitness)]

            previous_fitness = problem.eval_fitness_function(best_sol)
            index_mask = self.comparator(current_fitness, previous_fitness)
            best_sol[index_mask] = self.solution[index_mask]

        self.time_taken = (time.time()-init_time)
        return best_particle, problem.eval_fitness_function(best_particle)
