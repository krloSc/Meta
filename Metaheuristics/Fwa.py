from evaluate.Fitness import *
from solution.Solution import *
import numpy as np
from numpy.random import rand,uniform,randint
import matplotlib.pyplot as plt
import time
fit=Fitness()
sol=Solution()

class Fwa():

    def __init__(self,size,parameters=[]):
        if parameters==[]:
            print(__class__.__name__)
            try:
                path=os.getcwd()
                file=open(path+"\\Metaheuristics\\"+__class__.__name__+".param",'r')
                lst=file.read().split('\n')
                parameters=eval(lst[0])

            except:
                print("Parameters not found")
        self.solution=sol.init_solution(size[0],size[1])
        self.parameters=parameters

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
        e=self.parameters.get("e")
        s_hat=self.parameters.get("s_hat")
        a_hat=self.parameters.get("a_hat")
        best=min(fit.evaluate(self.solution,problem))
        worst=np.max(fit.evaluate(self.solution,problem))
        xmin=1
        xmax=10
        initime=time.time()
        for i in range(20):
            for i in range(self.solution.shape[0]): #numero de fireworks
                s=np.rint(s_hat*(worst-fit.evaluate(self.solution,problem)+e)/(np.sum(worst-fit.evaluate(self.solution,problem))+e))
                s[s<xmin]=xmin
                s[s>xmax]=xmax
                a=a_hat*(fit.evaluate(self.solution,problem)-best+e)/(np.sum(fit.evaluate(self.solution,problem)-best)+e)
                r_d=randint(0,2,(int(s[i]),2))
                r_d[(np.max(r_d,axis=1)<1)]=np.array([1,1]) #al menos alguno de los dos debe actualizarse
                upd=a[i]*r_d*uniform(-1,1,r_d.shape)                       # (s,2)
                variable=sol.generate_from(self.solution[i,:].reshape(1,-1),s[i],upd)
                try:
                    solutions=np.concatenate((solutions,variable[0]))
                except:
                    solutions=variable[0]
            solutions=np.concatenate((solutions,self.solution))
            bindex=np.argmin(fit.evaluate(solutions,problem))
            best=min(fit.evaluate(solutions,problem))
            worst=max(fit.evaluate(solutions,problem))
            best_spark=solutions[bindex].reshape(1,-1)
            solutions=np.delete(solutions,bindex,0)
            n_minus=Fwa.nfire(solutions)
            self.solution=np.concatenate((best_spark,n_minus))
        #print(best_spark,fit.evaluate(best_spark,problem)))
        self.time_taken = (time.time()-initime)
        return best_spark, fit.evaluate(best_spark,problem)
