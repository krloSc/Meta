from problem.Problem import*
from solution.Solution import *
import numpy as np
from numpy.random import rand,uniform,randint
import matplotlib.pyplot as plt
import time
sol=Solution()

class Fwa():

    def __init__(self,size, optimization: OptimizationType, parameters={}):
        if parameters=={}:
            try:
                path=os.getcwd()
                file=open(path+"\\Metaheuristics\\"+__class__.__name__+".param",'r')
                lst=file.read().split('\n')
                self.parameters=eval(lst[0])

            except:
                print("Parameters not found")
                self.parameters=parameters

        self.size = size
        if optimization == OptimizationType.MINIMIZATION:
            self.comparator = np.less
            self.better_index = np.argmin
            self.best_value = min
            self.worst = max
        else:
            self.comparator = np.greater
            self.better_index = np.argmax
            self.best_value = max
            self.worst = min
        return

    def nfire(solution):
        dist=np.zeros((solution.shape[0],1))
        for i in range(solution.shape[0]):
            current=solution[i]
            list=np.delete(solution,i,0)
            dist[i]=np.linalg.norm(list-current)
        index=np.argsort(dist,axis=None)
        sum=np.sum(dist,axis=0)
        prob=dist/sum
        index=np.argsort(prob,axis=None)
        return solution[index[-19:]]

    def run(self,problem):
        self.solution=sol.init_solution(self.size[0],self.size[1], problem.boundaries)
        e=self.parameters.get("e")
        s_hat=self.parameters.get("s_hat")
        a_hat=self.parameters.get("a_hat")
        fitness_list = problem.eval_fitness_function(self.solution)
        best=self.best_value(fitness_list)
        worst=self.worst(fitness_list)
        xmin=1
        xmax=10
        initime=time.time()
        for i in range(20):
            for i in range(self.solution.shape[0]): #numero de fireworks
                s=np.rint(
                            s_hat*(worst-fitness_list+e)
                            /(np.sum(worst-fitness_list)+e)
                            )

                s = np.clip(s,xmin,xmax)
                a=a_hat*(problem.eval_fitness_function(self.solution)-best+e)/(np.sum(problem.eval_fitness_function(self.solution)-best)+e)
                r_d=randint(0,2,(int(s[i]),2))
                r_d[(np.max(r_d,axis=1)<1)]=np.array([1,1]) #al menos alguno de los dos debe actualizarse
                upd=a[i]*r_d*uniform(-1,1,r_d.shape)                       # (s,2)
                variable=sol.generate_from(self.solution[i,:].reshape(1,-1),s[i],upd)
                try:
                    solutions=np.concatenate((solutions,variable[0]))
                except:
                    solutions=variable[0]
            solutions=np.concatenate((solutions,self.solution))
            bindex=self.better_index(problem.eval_fitness_function(solutions))
            best=self.best_value(problem.eval_fitness_function(solutions))
            worst=self.worst(problem.eval_fitness_function(solutions))
            best_spark=solutions[bindex].reshape(1,-1)
            solutions=np.delete(solutions,bindex,0)
            n_minus=Fwa.nfire(solutions)
            self.solution=np.concatenate((best_spark,n_minus))
        #print(best_spark,problem.eval_fitness_function(best_spark,problem)))
        self.time_taken = (time.time()-initime)
        return best_spark, problem.eval_fitness_function(best_spark)
