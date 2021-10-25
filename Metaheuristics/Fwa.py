from problem.Problem import*
from solution.Solution import *
from Metaheuristics.meta import Metaheuristic
import numpy as np
from numpy.random import rand,uniform, choice, randint
import matplotlib.pyplot as plt
import time
sol=Solution()

class Fwa(Metaheuristic):

    def nfire(self,solution):
        dist=np.zeros((solution.shape[0],1))
        for i in range(solution.shape[0]):
            current=solution[i]
            list=np.delete(solution,i,0)
            dist[i]=np.linalg.norm(list-current)

        sum=np.sum(dist,axis=0)
        prob=dist/sum
        index=np.argsort(prob,axis=None)

        return solution[index[:self.order*(self.n_fireworks-1)]]
        #return solution[index[-self.order*(self.n_fireworks-1):]]

    def create_mask(self, population, dimension):

        mask =  np.zeros((population, dimension))
        for i in range(population):
            mask[i] = choice(range(dimension), dimension, replace = False)

        return mask

    def run(self,problem):
        self.solution=sol.init_solution(self.size[0],self.size[1], problem.boundaries)
        e=self.parameters.get("e",0.001)
        m=self.parameters.get("m",100)
        a_hat=self.parameters.get("a_hat",500)
        n_explosion=self.parameters.get("n_explosion",30)
        xmin=8
        xmax=20
        #sparks = []
        initime=time.time()
        self.n_fireworks = self.solution.shape[0]

        #input(problem.eval_fitness_function(self.solution))

        for explosion in range(n_explosion):

            fitness_list = problem.eval_fitness_function(self.solution)
            best=self.best_value(fitness_list)
            worst=self.worst(fitness_list)
            s=np.rint(
                        m*(worst-fitness_list+e)
                        /(np.sum(worst-fitness_list)+e)
                        )
            s = np.clip(s,xmin,xmax)
            #print(s)
            a=a_hat*(fitness_list-best+e)/(np.sum(fitness_list-best)+e)
            a = np.clip(a,2,1000)

            for i in range(self.n_fireworks):

                dimension_mask = self.create_mask(int(s[i]),2)
                amplitude=a[i]*dimension_mask*uniform(-1,1,dimension_mask.shape)
                #print(amplitude)                     # (s,2)
                sparks=sol.generate_from(self.solution[i].reshape(1,-1),s[i],amplitude)
                sparks = sparks.reshape(-1,2)

                #input(sparks)

                try:
                    solutions=np.concatenate((solutions,sparks))
                except:
                    solutions=sparks

            solutions=np.concatenate((solutions,self.solution))
            bindex=self.best_index(problem.eval_fitness_function(solutions))
            best=self.best_value(problem.eval_fitness_function(solutions))
            worst=self.worst(problem.eval_fitness_function(solutions))
            best_spark=solutions[bindex].reshape(1,-1)
            self.lines.append(problem.eval_fitness_function(best_spark))
            solutions=np.delete(solutions,bindex,0)
            n_minus=self.nfire(solutions)
            self.solution=np.concatenate((best_spark,n_minus))
            #input(problem.eval_fitness_function(self.solution))
        self.time_taken = (time.time()-initime)
        return best_spark, problem.eval_fitness_function(best_spark)
