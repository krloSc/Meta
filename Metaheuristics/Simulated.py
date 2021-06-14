from evaluate.Fitness import *
from solution.Solution import *
import numpy as np
from numpy.random import rand,uniform
import matplotlib.pyplot as plt
fit=Fitness()
sol=Solution()

class Simulated_A():
    def __init__(self,size,parameters):
        self.solution=sol.init_solution(size[0],size[1])
        self.parameters=parameters
        print(self.solution)

    def simulated(self,problem):
        t=self.parameters.get("to")
        ta=self.parameters.get("ta")
        delta=self.parameters.get("delta")
        n=1
        while t>ta:
            n_s=5
            factor=uniform(-1,1,(n_s,self.solution.shape[1]))*1/(0.1*n)                                                           # neighbour (3,5,2)
            neigbours=sol.generate_from2(self.solution,n_s,factor)
            #print(neigbours)
            for i in range(neigbours.shape[0]):
                current_fitness=fit.evaluate(self.solution[i,:],problem)
                best_nbr=neigbours[i,np.argmin(fit.evaluate(neigbours[i,:,:],problem))]
                if (fit.evaluate(best_nbr,problem)<current_fitness):
                    self.solution[i,:]=best_nbr
                else:
                    r=rand()
                    l=current_fitness-fit.evaluate(best_nbr,problem)
                    ann=np.exp(-l/t)
                    if (r<ann):
                        self.solution[i,:]=best_nbr
            t=t*delta
            n+=1
        print(fit.evaluate(self.solution,problem))
        return self.solution




                                                     #completar cambios de esta gente
    """

    parameters={"to":1000,"ta":0.001,"delta":0.99}
    solution=simulated(solution,5,parameters)
    #print(solution)
    X = np.arange(-10, 10, 0.1)
    Y = np.arange(-10, 10, 0.1)
    X,Y=np.meshgrid(X,Y)
    Z=X**2 + Y**2 + (25 * (np.sin(X)**2 + np.sin(Y)**2))
    #Z=np.cos(np.sqrt(X**2+Y**2))*np.sin(X/2+4)
    fig,ax=plt.subplots(1,1)
    ax.contourf(X, Y, Z,100)
    ax.autoscale(False)
    ax.scatter(solution[:,0],solution[:,1],color='r',alpha=1,zorder=1)
    plt.show()"""
