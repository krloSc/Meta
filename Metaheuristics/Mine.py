from evaluate.Fitness import *
from solution.Solution import *
from problem.Problem import*
import numpy as np
from numpy.random import rand,uniform,randint
import matplotlib.pyplot as plt
import time
fit=Fitness()
sol=Solution()

#Inicializacion
class Mine():

    def __init__(self,size,parameters=[]):
        if parameters==[]:
            try:
                path=os.getcwd()
                file=open(path+"\\Metaheuristics\\"+self.__class__.__name__+".param",'r')
                lst=file.read().split('\n')
                parameters=eval(lst[0])

            except:
                print("Parameters not found")
        self.solution=sol.init_solution(size[0],size[1])
        self.parameters=parameters

    def run(self,problem):
        initime=time.time()
        new_sol=np.copy(self.solution)
        subset=sol.generate_from(self.solution,10,uniform(-0.2,0.2,(10,2))) #mejorar estancamiento
        #print(solution)

        for i in range(subset.shape[0]):
            current_sol=subset[i,np.argmin(fit.evaluate(subset[i],problem))]
            if (fit.evaluate(current_sol,problem)<fit.evaluate(self.solution[i],problem)):
                new_sol[i]=current_sol
        delta=new_sol-self.solution
        #print(delta)
        sub_dim=5
        deltaf=fit.evaluate(new_sol,problem)-fit.evaluate(self.solution,problem)
        zmask=np.ones(self.solution.shape)
        current_sol=np.ones(self.solution.shape)
        n=np.ones((sub_dim,1)) #controla moviemiento gradient
        prev_sol=np.copy(self.solution)
        anchor=np.array([])
        color='r'
        for i in range(100):
            #solution[solution>10]=uniform(-10,10)
            #solution[solution<-10]=uniform(-10,10)
            dif=np.any(prev_sol!=self.solution,axis=1)
            prev_sol[dif]=self.solution[dif]
            prev_deltaf=np.copy(deltaf)
            #print(dif)
            r_d=randint(0,2,self.solution.shape)
            r_d[(np.max(r_d,axis=1)<1)]=np.array([1,1])
                #print(r_d)
            #input()
            entrophy=uniform(-1,1,(sub_dim,self.solution.shape[1]))/(0.01*n)#*r_d
            subset=sol.generate_from(self.solution,sub_dim,entrophy)
            for i in range(self.solution.shape[0]):
                current_sub=subset[i]
                current_sol[i]=current_sub[np.argmin(fit.evaluate(subset[i],problem))]
            self.solution[fit.evaluate(current_sol,problem)<fit.evaluate(self.solution,problem)]=current_sol[fit.evaluate(current_sol,problem)<fit.evaluate(self.solution,problem)]
            dif=np.any(prev_sol!=self.solution,axis=1)
            delta[dif]=(self.solution-prev_sol)[dif]
            deltaf[dif]=(fit.evaluate(self.solution,problem)-fit.evaluate(prev_sol,problem))[dif]

            try:
                anchor=np.concatenate((anchor,self.solution[np.abs(np.sum(delta,axis=1))<0.0000001])) #limitar anchors
            except Exception as e:
                anchor=self.solution[np.abs(np.sum(delta,axis=1))<0.0000001]
                #print(e)
            #print("anchor ",anchor)
            n=n+2
        self.time_taken = (time.time()-initime)
        return self.solution[np.argmin(fit.evaluate(self.solution,problem))] , np.min(fit.evaluate(self.solution,problem))
