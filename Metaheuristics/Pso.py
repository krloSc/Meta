from solution.Solution import *
from problem.Problem import*
from Metaheuristics.meta import Metaheuristic
import numpy as np
from numpy.random import rand,uniform
import matplotlib.pyplot as plt
import time

class Pso(Metaheuristic):

    def update_velocity(self,
            prev_velocity: np.ndarray,
            best_sol: np.ndarray,
            best_particle: np.ndarray,
            r1: float,
            r2: float) -> np.ndarray:
        """Calculate the velocity of the particles"""
        velocity = (
                    self.inertia*prev_velocity
                    + r1*self.r1_factor*(best_sol-self.solution)
                    + r2*self.r2_factor*(best_particle-self.solution))
        return velocity

    def run(self) -> tuple:
        """ Run the PSO algorithm and return the best solution and its fitness"""

        self.iterations = self.parameters.get("iterations",100)
        self.inertia = self.parameters.get("inertia",0.1)
        self.r1_factor = self.parameters.get("r_one_factor", 1.5) # must be < total genes
        self.r2_factor = self.parameters.get("r_two_factor", 3)

        init_time=time.time()
        self.solution=self.sol.init_solution(
                                        self.size[0],
                                        self.size[1],
                                        self.problem.boundaries
                                        )

        current_fitness,_=self.problem.eval_fitness_function(self.solution)
        best_particle=self.solution[self.best_index(current_fitness)]
        velocity = uniform(0,1,self.solution.size)
        velocity = velocity.reshape(-1,self.solution.shape[1])
        best_sol = self.solution

        for i in range(self.iterations):

            r1 = uniform(0,1,self.solution.shape[0]).reshape(-1,1)
            r2 = uniform(0,1,self.solution.shape[0]).reshape(-1,1)
            velocity = self.update_velocity(
                                            velocity,
                                            best_sol,
                                            best_particle,
                                            r1,
                                            r2
                                            )
            self.solution=self.sol.update_sol(self.solution,velocity)
            current_fitness,_ = self.problem.eval_fitness_function(self.solution)
            current_best_fitness = self.best_value(current_fitness)
            best_particle_fitness,_ = self.problem.eval_fitness_function(best_particle)
            self.lines.append(float(best_particle_fitness))
            if self.comparator(current_best_fitness, best_particle_fitness):
                best_particle=self.solution[self.best_index(current_fitness)]

            previous_fitness,_ = self.problem.eval_fitness_function(best_sol)
            index_mask = self.comparator(current_fitness, previous_fitness)
            best_sol[index_mask] = self.solution[index_mask]

        self.time_taken.append(time.time()-init_time)
        return best_particle, *self.problem.eval_fitness_function(best_particle)
