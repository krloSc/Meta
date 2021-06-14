from evaluate.Fitness import *
from solution.Solution import *
from problem.Problem import*
import numpy as np
from numpy.random import rand,uniform,randint
import matplotlib.pyplot as plt
fit=Fitness()
sol=Solution()

#Inicializacion
solution=sol.init_solution(5,2)


def mine(solution,problem):
    new_sol=np.copy(solution)
    subset=sol.generate_from2(solution,10,uniform(-0.2,0.2,(10,2))) #mejorar estancamiento
    #print(solution)

    for i in range(subset.shape[0]):
        current_sol=subset[i,np.argmin(fit.evaluate(subset[i],problem))]
        if (fit.evaluate(current_sol,problem)<fit.evaluate(solution[i],problem)):
            new_sol[i]=current_sol
    delta=new_sol-solution
    #print(delta)
    sub_dim=5
    deltaf=fit.evaluate(new_sol,problem)-fit.evaluate(solution,problem)
    zmask=np.ones(solution.shape)
    current_sol=np.ones(solution.shape)
    n=np.ones((sub_dim,1)) #controla moviemiento gradient
    prev_sol=np.copy(solution)
    anchor=np.array([])
    color='r'
    for i in range(100):
        #solution[solution>10]=uniform(-10,10)
        #solution[solution<-10]=uniform(-10,10)
        dif=np.any(prev_sol!=solution,axis=1)
        prev_sol[dif]=solution[dif]
        prev_deltaf=np.copy(deltaf)
        #print(dif)
        r_d=randint(0,2,solution.shape)
        r_d[(np.max(r_d,axis=1)<1)]=np.array([1,1])
        #print(r_d)
        #input()
        entrophy=uniform(-1,1,(sub_dim,solution.shape[1]))/(0.01*n)#*r_d
        subset=sol.generate_from2(solution,sub_dim,entrophy,100,anchor=anchor)
        for i in range(solution.shape[0]):
            current_sub=subset[i]
            current_sol[i]=current_sub[np.argmin(fit.evaluate(subset[i],problem))]
        solution[fit.evaluate(current_sol,problem)<fit.evaluate(solution,problem)]=current_sol[fit.evaluate(current_sol,problem)<fit.evaluate(solution,problem)]
        dif=np.any(prev_sol!=solution,axis=1)
        delta[dif]=(solution-prev_sol)[dif]
        deltaf[dif]=(fit.evaluate(solution,problem)-fit.evaluate(prev_sol,problem))[dif]

        try:
            anchor=np.concatenate((anchor,solution[np.abs(np.sum(delta,axis=1))<0.0000001])) #limitar anchors
        except Exception as e:
            anchor=solution[np.abs(np.sum(delta,axis=1))<0.0000001]
            print(e)
        print("anchor ",anchor)
        ax.scatter(solution[:,0],solution[:,1],color=color,alpha=0.9)
        n=n+2
    return solution,anchor

problem=Problem("ground")

X = np.arange(problem.boundaries[0], problem.boundaries[1], 0.1)
Y = np.arange(problem.boundaries[2], problem.boundaries[3], 0.1)
X,Y=np.meshgrid(X,Y)
Z=eval(problem.problem)
fig,ax=plt.subplots(1,1)
ax.contourf(X, Y, Z,100)
ax.autoscale(False)
ax.scatter(solution[:,0],solution[:,1],color='g',alpha=1)
print(solution)



solution,anchor=mine(solution,problem) #function call
ax.scatter(solution[:,0],solution[:,1],color='b',alpha=1)
ax.scatter(anchor[:,0],anchor[:,1],color='w',alpha=1)
print(np.min(fit.evaluate(solution,problem)),np.min(fit.evaluate(anchor,problem)))
plt.show()
