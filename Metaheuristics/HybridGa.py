from solution.Solution import *
from problem.Problem import*
import numpy as np
from numpy.random import rand,uniform, choice, randint
import matplotlib.pyplot as plt
import time
from util import param
from Metaheuristics.meta import Metaheuristic


class HybridGa(Metaheuristic):


    def individual_fitness(self, solution: np.ndarray) -> np.ndarray:
        """Obtain the fitness of a entire cromosome perfoming a sum of every gene's fitness in it"""

        fitness,_ = self.problem.eval_fitness_function(solution.reshape(-1,2))
        fitness = np.amax(fitness.reshape(self.rows, -1),axis=1)
        genes = solution.shape[1]
        return fitness

    def parents_selection(self, individuals: np.ndarray) -> tuple:
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

    def generate_individuals(self, elite: np.ndarray) -> np.ndarray:
        """Generate a set of cromosomes including elites from previous generations"""

        solution = self.sol.init_solution(self.rows, self.columns, self.problem.boundaries)
        for i in range(self.cromosome_len-1): #number of solution per cromosome
            solution = np.append(solution,
                            self.sol.init_solution(self.rows, self.columns, self.problem.boundaries),
                            axis = 1)
        solution = solution.reshape(self.rows,-1,2)
        if elite.size != 0:
            for i in range(elite.shape[0]):
                solution[i] = elite[i].reshape(1,-1,2)
        return solution

    def recombination(
            self,
            parent_a: np.ndarray,
            parent_b: np.ndarray) -> tuple:
        """perform a recombination process to produce a new offspring"""

        max_index =  parent_a.shape[0]
        index = np.random.randint(1,max_index)
        child_a = np.concatenate((parent_a[:index], parent_b[index:]), axis = 0)
        child_b = np.concatenate((parent_b[:index], parent_a[index:]), axis = 0)
        child_a = child_a.reshape(1,-1,2)
        child_b = child_b.reshape(1,-1,2)
        return child_a, child_b


    def mutation(
            self,
            parent_a: np.ndarray,
            parent_b: np.ndarray,
            mut_genes: int,
            randomness: float) -> np.ndarray:
        """ perform cromosome mutation """

        max_genes = np.random.randint(1,mut_genes+1)
        max_index = parent_a.shape[0]
        genes_a = np.random.choice(range(max_index), max_genes, replace=False)
        genes_b = np.random.choice(range(max_index), max_genes, replace=False)
        child_a = parent_a.copy()
        child_b = parent_b.copy()
        for i,j in zip(genes_a, genes_b):
            child_a[i] = self.sol.generate_single(parent_a[i], randomness)
            child_b[j] = self.sol.generate_single(parent_b[i], randomness)
        child_a = child_a.reshape(1,-1,2)
        child_b = child_b.reshape(1,-1,2)

        return child_a, child_b

    def select_elites(
            self,
            solution: np.ndarray,
            fitness: np.ndarray) -> np.ndarray:
        """ Return an array of elites for the next generation"""

        elites = solution[np.argsort(fitness)]
        if self.order == 1: #MAXIMIZATION
            elites = elites[::-1]

        return elites[:self.elite_size]

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
        return improved_solution.reshape(1,-1,2)

    def explore(self, solution: np.ndarray) -> np.ndarray:
        """Perform exploration to get out of local minima/maxima"""
        mask = self.create_mask(*solution.shape)
        random_solution = self.sol.init_solution(self.cromosome_len, self.size[1], self.problem.boundaries)
        new_solution = solution*(-mask+1)+random_solution*mask
        self.sol.check_boundaries(new_solution)
        return new_solution.reshape(1,-1,2)

    def run(self) -> tuple:
        """ Run the Hybrid GA/Hill-Climbing algorithm and return the best solution and its fitness"""

        initime=time.time()
        self.cromosome_len = self.parameters.get("cromosome_len", 4) #cromosome size
        cross_rate = self.parameters.get("cross_rate",0.3)
        mutation_rate = self.parameters.get("mutation_rate",0.7)
        max_mut_genes = self.parameters.get("mut_genes", 3) # must be < total genes
        randomness = self.parameters.get("randomness", 500)
        decreasing_rate = self.parameters.get("decreasing", 0.95)
        rnd_thold = self.parameters.get("rnd_thold", 1)
        generations = self.parameters.get("generations", 10)
        self.elite_size = self.parameters.get("elite_size", 3)
        self.step = self.parameters.get("step", 10)
        iterations = self.parameters.get("iterations", 200)
        beta = self.parameters.get("beta",0.2)
        improve = self.parameters.get("improve",0.2)
        self.rows = self.size[0]
        self.columns = self.size[1]
        elite = np.array([])
        random_amount = randomness
        for i in range(generations):

            solution = self.generate_individuals(
                    elite)

            new_generation = np.array([])
            fitness = self.individual_fitness(solution)

            while new_generation.size < self.rows*self.cromosome_len*2:

                index_a, index_b= self.parents_selection(np.argsort(fitness*-self.order)) #take a look on this
                parent_a = solution[index_a]
                parent_b = solution[index_b]

                if rand() <= cross_rate:
                    child_a, child_b = self.recombination(
                                                            parent_a,
                                                            parent_b)
                else:
                    child_a, child_b = parent_a.copy(), parent_b.copy()

                if rand() <= mutation_rate:
                    child_a, child_b = self.mutation(
                                                        child_a.reshape(-1,2),
                                                        child_b.reshape(-1,2),
                                                        max_mut_genes,
                                                        random_amount)

                if rand() <= beta:
                    child_a = self.explore(child_a.reshape(-1,2))
                    child_b = self.explore(child_b.reshape(-1,2))
                if rand() <= improve:
                    child_a = self.improve(child_a.reshape(-1,2))
                    child_b = self.improve(child_b.reshape(-1,2))

                children = np.append(child_a,child_b)
                new_generation = np.append(new_generation,children)

            new_generation = new_generation.reshape(-1,self.cromosome_len,2)
            new_fitness = self.individual_fitness(new_generation)
            better = self.comparator(new_fitness,fitness)
            if (np.any(better)):
                solution[better] = new_generation[better]
            fitness =  self.individual_fitness(solution)
            elite = self.select_elites(solution, fitness)
            self.lines.append(self.best_value(fitness))
            random_amount *= decreasing_rate

        best_cromosome = solution[self.best_index(fitness)].reshape(-1,2)
        cromosme_fitness,_ = self.problem.eval_fitness_function(best_cromosome)
        best_gene = best_cromosome[self.best_index(cromosme_fitness)]
        self.time_taken.append(time.time()-initime)
        return best_gene, *self.problem.eval_fitness_function(best_gene)