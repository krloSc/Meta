from evaluate.Fitness import *
from solution.Solution import *
import numpy as np
from numpy.random import rand,uniform,randint
import matplotlib.pyplot as plt
fit=Fitness()
sol=Solution()

#Inicializacion
solution=sol.init_solution(5,2)


def mine(solution):
    new_sol=np.copy(solution)
    subset=sol.generate_from2(solution,10,uniform(-0.2,0.2,(10,2))) #mejorar estancamiento
    #print(subset)

    for i in range(subset.shape[0]):
        current_sol=subset[i,np.argmin(fit.evaluate(subset[i]))]
        if (fit.evaluate(current_sol)<fit.evaluate(solution[i])):
            new_sol[i]=current_sol

    delta=new_sol-solution
    deltaf=fit.evaluate(new_sol)-fit.evaluate(solution)
    #print(solution)
    #print(fit.evaluate(solution))
    #print(deltaf)
    #input()
    for i in range(10):
        solution[solution>10]=uniform(-10,10)
        solution[solution<-10]=uniform(-10,10)
        r_d=randint(0,2,solution.shape)
        r_d[(np.max(r_d,axis=1)<1)]=np.array([1,1])
        mask=np.zeros((solution.shape[0],1))
        mask[np.sum(r_d,axis=1)==1]=1 #mascara de movimiento unidimensional
        current_solution=solution+(1/delta)*0.05*deltaf.reshape([-1,1])*r_d
        current_delta=current_solution-solution
        delta[current_delta!=0]=current_delta[current_delta!=0]
        print(r_d*delta*rand()*0.1*deltaf.reshape([-1,1])*r_d)
        #input()
        current_deltaf=fit.evaluate(current_solution)-fit.evaluate(solution)
        zmask=r_d*current_deltaf.reshape([-1,1])
        zmask[zmask>0]=-1
        zmask[zmask!=-1]=1
        delta=delta*zmask
        solution[fit.evaluate(current_solution)<fit.evaluate(solution)]=current_solution[fit.evaluate(current_solution)<fit.evaluate(solution)]
        deltaf=current_deltaf
        #print(deltaf)
        #print(solution," Fitness ",fit.evaluate(solution))
        ax.scatter(solution[:,0],solution[:,1],color='r',alpha=0.5,zorder=1)
    return solution

X = np.arange(-10, 10, 0.1)
Y = np.arange(-10, 10, 0.1)
X,Y=np.meshgrid(X,Y)
Z=X**2 + Y**2 + (25 * (np.sin(X)**2 + np.sin(Y)**2))
#Z=np.cos(np.sqrt(X**2+Y**2))*np.sin(X/2+4)
fig,ax=plt.subplots(1,1)
ax.contourf(X, Y, Z,100)
ax.autoscale(False)
solution=mine(solution)

ax.scatter(solution[:,0],solution[:,1],color='b',alpha=1,zorder=1)

#print(solution)
plt.show()
