from solution.Solution import *
from problem.Problem import*
from Metaheuristics.meta import Metaheuristic
import numpy as np
from numpy.random import rand, uniform, randint, choice
import matplotlib.pyplot as plt
import time


class HillClimbing(Metaheuristic):

    def create_mask(self, population, dimension):
        """Create a mask for randomly choose the dimension where the spark will
            move"""

        mask =  np.zeros((population, dimension))
        for i in range(population):
            mask[i] = choice(range(dimension), dimension, replace = False)

        return mask

    def improve(self, solution: np.ndarray) -> np.ndarray:
        """Perform exploitation of the solution"""

        mask = self.create_mask(*solution.shape)
        improved_solution = solution + mask*self.step*uniform(-1,1)
        self.sol.check_boundaries(improved_solution)
        return improved_solution

    def explore(self, solution: np.ndarray) -> np.ndarray:
        """Perform exploration to get out of local minima/maxima"""

        mask = self.create_mask(*solution.shape)
        random_solution = self.sol.init_solution(self.size[0],self.size[1], self.problem.boundaries)
        new_solution = solution*(-mask+1)+random_solution*mask
        self.sol.check_boundaries(new_solution)
        return new_solution

    def run(self) -> tuple:
        """ Run the Hill-Climbing algorithm and return the best solution and its fitness"""

        initime=time.time()
        self.step = self.parameters.get("step", 10)
        iterations = self.parameters.get("iterations", 200)
        beta = self.parameters.get("beta",0.2)
        solution = self.sol.init_solution(self.size[0],self.size[1], self.problem.boundaries)
        fitness,_ = self.problem.eval_fitness_function(solution)
        for i in range(iterations):

            if rand() <= beta:
                solution_prime = self.explore(solution)
            else:
                solution_prime = self.improve(solution)

            current_fitness,_ = self.problem.eval_fitness_function(solution_prime)
            better_index = self.comparator(current_fitness, fitness)

            if np.any(better_index):
                solution[better_index] = solution_prime[better_index]

            fitness,power = self.problem.eval_fitness_function(solution)
            self.lines.append(self.best_value(fitness))

        best_fitness = self.best_index(fitness)
        self.time_taken.append(time.time()-initime)
        return solution[best_fitness], fitness[best_fitness], power[best_fitness]
