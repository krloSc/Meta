from solution.Solution import *
from problem.Problem import*
from Metaheuristics.meta import Metaheuristic
import numpy as np
from numpy.random import rand, uniform, randint, choice
import matplotlib.pyplot as plt
import time

sol=Solution()

class HillClimbing(Metaheuristic):

    def create_mask(self, population, dimension):

        mask =  np.zeros((population, dimension))
        for i in range(population):
            mask[i] = choice(range(dimension), dimension, replace = False)

        return mask

    def improve(self, solution: np.ndarray) -> np.ndarray:

        mask = self.create_mask(*solution.shape)
        improved_solution = solution + mask*self.step*uniform(-1,1)
        sol.check_boundaries(improved_solution)
        return improved_solution

    def explore(self, solution: np.ndarray, problem) -> np.ndarray:

        mask = self.create_mask(*solution.shape)
        random_solution = sol.init_solution(self.size[0],self.size[1], problem.boundaries)
        new_solution = solution*(-mask+1)+random_solution*mask
        sol.check_boundaries(new_solution)
        return new_solution

    def run(self,problem):
        initime=time.time()
        self.step = self.parameters.get("step", 10)
        iterations = self.parameters.get("iterations", 200)
        beta = self.parameters.get("beta",0.2)
        solution = sol.init_solution(self.size[0],self.size[1], problem.boundaries)
        fitness = problem.eval_fitness_function(solution)
        for i in range(iterations):

            if rand() <= beta:
                solution_prime = self.explore(solution, problem)
            else:
                solution_prime = self.improve(solution)

            current_fitness = problem.eval_fitness_function(solution_prime)
            better_index = self.comparator(current_fitness, fitness)
            if np.any(better_index):
                solution[better_index] = solution_prime[better_index]
            fitness = problem.eval_fitness_function(solution)
            self.lines.append(self.best_value(fitness))
        best_fitness = self.best_index(fitness)
        self.time_taken = (time.time()-initime)
        return solution[best_fitness], fitness[best_fitness]
