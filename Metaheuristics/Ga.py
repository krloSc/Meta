from solution.Solution import *
from problem.Problem import*
import numpy as np
from numpy.random import rand,uniform
import matplotlib.pyplot as plt
import time
from util import param

sol=Solution()

class Ga():

    def __init__(self,size, optimization: OptimizationType, parameters = {}):
        """initilizer function for Ga Metaheuristic"""

        if type(parameters) == str:
            parameters = param.get_parameters(parameters)

        self.lines = []
        self.size = size
        self.parameters=parameters

        if optimization == OptimizationType.MINIMIZATION:
            self.comparator = np.less
            self.best_index = np.argmin
            self.best_value = min
            self.worst = max
            self.order = -1
        else:
            self.comparator = np.greater
            self.best_index = np.argmax
            self.best_value = max
            self.worst = min
            self.order = 1
        return

    def individual_fitness(self, solution: np.ndarray, rows) -> np.ndarray:
        """Obtain the fitness of a entire cromosome perfoming a sum of every gene's fitness in it"""

        fitness = self.problem.eval_fitness_function(solution.reshape(-1,2))
        fitness = np.sum(fitness.reshape(rows,-1), axis = 1)
        return fitness

    def parents_selection(self, individuals: np.ndarray) -> np.ndarray:
        """Parents selection through Roulette Wheel selection"""

        wheel = []
        for i in range(len(individuals)):
            wheel.append([individuals[i]]*(len(individuals)-i))
        wheel = np.concatenate(wheel)
        size_wheel = sum(range(len(individuals)+1))
        parent_a = wheel[np.random.randint(0,size_wheel)]
        while True:
            parent_b = wheel[np.random.randint(0,size_wheel)]
            if parent_a != parent_b:
                break
        return parent_a, parent_b

    def generate_individuals(self, columns: np.ndarray, elite: np.ndarray):
        """Generate a set of cromosomes including elites from previous generations"""

        cromosome_len = self.parameters.get("cromosome_len", 4) #cromosome size
        solution = sol.init_solution(self.rows, columns, self.problem.boundaries)
        for i in range(cromosome_len-1): #number of solution per cromosome
            solution = np.append(solution,
                            sol.init_solution(self.rows, columns, self.problem.boundaries),
                            axis = 1)
        solution = solution.reshape(self.rows,-1,2)
        if elite.size != 0:
            for i in range(elite.shape[0]):
                self.solution_update(elite[i].reshape(1,-1,2), solution)
        return solution

    def solution_update(self, offspring: np.ndarray, solution: np.ndarray) -> None:
        """Update the solution if the new cromosme is better than the worst"""

        current_fitness = self.individual_fitness(solution, solution.shape[0])
        offspring_fitness = self.individual_fitness(offspring, offspring.shape[0])
        if self.comparator(offspring_fitness, self.worst(current_fitness)):
            index = np.argsort(current_fitness)[0]
            current_fitness[index] = offspring_fitness
            solution[index] = offspring
        #update line:
        best_index = np.argsort(current_fitness)[-1]
        self.lines.append(current_fitness[best_index])
        return

    def recombination(
            self, parent_a: np.ndarray,
            parent_b: np.ndarray,
            solution: np.ndarray) -> None:
        """perform a recombination process to produce a new offspring"""

        max_index =  parent_a.shape[0]
        index = np.random.randint(1,max_index)
        child = np.concatenate((parent_a[:index], parent_b[index:]), axis = 0)
        child = child.reshape(1,-1,2)
        self.solution_update(child, solution)
        return

    def mutation(
            self,
            parent: np.ndarray,
            mut_genes: int,
            randomness: float,
            solution: np.ndarray) -> np.ndarray:

        """ perform cromosome mutation """
        max_genes = np.random.randint(1,mut_genes+1)
        max_index = parent.shape[0]
        genes = np.random.choice(range(max_index), max_genes, replace=False)
        mutated = parent.copy()
        for i in genes:
            mutated[i] = sol.generate_single(parent[i], randomness)
        mutated = mutated.reshape(1,-1,2)
        self.solution_update(mutated, solution)
        return


    def run(self, problem: Problem) -> tuple:
        """ Run the Ga algorithm and return the best solution and its fitness"""

        initime=time.time()
        self.problem = problem
        cross_rate = self.parameters.get("cross_rate",0.3)
        mutation_rate = self.parameters.get("mutation_rate",0.7)
        max_mut_genes = self.parameters.get("mut_genes", 3) # must be < total genes
        randomness = self.parameters.get("randomness", 500)
        decreasing = self.parameters.get("decreasing", 0.8)
        rnd_thold = self.parameters.get("rnd_thold", 0.8)
        generations = self.parameters.get("generations", 10)
        self.rows = self.size[0]
        columns = self.size[1]
        elite = np.array([])

        for i in range(generations):
            solution = self.generate_individuals(
                    columns,
                    elite)

            fitness = self.individual_fitness(solution, solution.shape[0])
            random_amount = randomness

            while random_amount > rnd_thold:
                index_a, index_b= self.parents_selection(np.argsort(fitness[::self.order])) #take a look on this
                parent_a = solution[index_a]
                parent_b = solution[index_b]
                self.recombination(parent_a, parent_b, solution)
                self.mutation(parent_a, max_mut_genes, randomness, solution) #probably the best cromosome
                random_amount *= 0.90

            elite = solution[np.argsort(fitness[:-4:-1])] #elite size

        best = solution[self.best_index(fitness)].reshape(-1,2)
        fit = problem.eval_fitness_function(best)
        best_gene = best[self.best_index(fit)]
        best_gene_fit = problem.eval_fitness_function(best_gene)
        self.time_taken = (time.time()-initime)

        return best_gene, best_gene_fit
