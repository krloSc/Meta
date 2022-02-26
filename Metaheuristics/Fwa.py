from problem.Problem import*
from solution.Solution import *
from Metaheuristics.meta import Metaheuristic
import numpy as np
from numpy.random import rand,uniform, choice, randint, normal
import matplotlib.pyplot as plt
import time

class Fwa(Metaheuristic):

    def nfire(self,solution):
        """Return the n-1 sparks for the next explosion using probability"""

        dist=np.zeros((solution.shape[0],1))
        for i in range(solution.shape[0]):
            current=solution[i]
            list=np.delete(solution,i,0)
            dist[i]=np.linalg.norm(list-current)

        sum=np.sum(dist,axis=0)
        prob=dist/sum
        index=np.argsort(prob,axis=None)
        new_solution = solution[index]
        prob[self.problem.eval_fitness_function(new_solution)[0]<0] = 0
        index=np.argsort(prob,axis=None)
        new_solution = new_solution[index[-self.order*(self.n_fireworks-1):]]
        return new_solution

    def create_mask(self, population, dimension):
        """Create a mask for randomly choose the dimension where the spark will
            move"""

        mask =  np.zeros((population, dimension))
        for i in range(population):
            mask[i] = choice(range(dimension), dimension, replace = False)

        return mask

    def gaussian_improve(self, solution: np.ndarray) -> np.ndarray:
        """Improve the solution using a gaussian distribution"""
        mask = self.create_mask(solution.shape[0],2)[:,0].astype(int)
        factor = normal(1,1,solution.shape)
        solution[mask] = solution[mask]*factor
        self.sol.check_boundaries(solution)
        return

    def run(self) -> tuple:
        """ Run the firework algorithm and return the best solution and its fitness"""

        self.solution=self.sol.init_solution(self.size[0],self.size[1], self.problem.boundaries)
        e=self.parameters.get("e",0.001)
        m=self.parameters.get("m",100)
        a_hat=self.parameters.get("a_hat",500)
        n_explosion=self.parameters.get("n_explosion",30)
        xmin=5
        xmax=50
        initime=time.time()
        self.n_fireworks = self.solution.shape[0]

        for explosion in range(n_explosion):

            fitness_list,_ = self.problem.eval_fitness_function(self.solution)
            best=self.best_value(fitness_list)
            worst=self.worst(fitness_list)
            s=np.rint(
                        m*(worst-fitness_list+e)
                        /(np.sum(worst-fitness_list)+e)
                        )
            s = np.clip(s,xmin,xmax)
            a=a_hat*(fitness_list-best+e)/(np.sum(fitness_list-best)+e)
            a = np.clip(a,10,1000)
            solutions = np.array([])
            for i in range(self.n_fireworks):

                dimension_mask = self.create_mask(int(s[i]),2)
                amplitude=a[i]*dimension_mask*uniform(-1,1,dimension_mask.shape)              # (s,2)
                sparks=self.sol.generate_from(self.solution[i].reshape(1,-1),s[i],amplitude)
                solutions=np.append(solutions,sparks)

            solutions=np.append(solutions,self.solution)
            solutions = solutions.reshape(-1,2)
            bindex=self.best_index(self.problem.eval_fitness_function(solutions)[0])
            best=self.best_value(self.problem.eval_fitness_function(solutions)[0])
            worst=self.worst(self.problem.eval_fitness_function(solutions)[0])
            best_spark=solutions[bindex].reshape(1,-1)
            self.lines.append(float(self.problem.eval_fitness_function(best_spark)[0]))
            solutions=np.delete(solutions,bindex,0)
            prev_time = time.time()
            n_minus=self.nfire(solutions)
            #self.gaussian_improve(n_minus)
            self.solution=np.concatenate((best_spark,n_minus))
        self.time_taken.append(time.time()-initime)
        return best_spark, *self.problem.eval_fitness_function(best_spark)
