from evaluate.Fitness import *
from solution.Solution import *
import numpy as np
from numpy.random import rand,uniform,randint
import matplotlib.pyplot as plt
fit=Fitness()
sol=Solution()

#Inicializacion
solution=sol.init_solution(30,2)


def mine(solution):
    new_sol=np.copy(solution)
    subset=sol.generate_from2(solution,10,uniform(-0.2,0.2,(10,2))) #mejorar estancamiento
    #print(solution)

    for i in range(subset.shape[0]):
        current_sol=subset[i,np.argmin(fit.evaluate(subset[i]))]
        if (fit.evaluate(current_sol)<fit.evaluate(solution[i])):
            new_sol[i]=current_sol
    delta=new_sol-solution
    #print(delta)
    sub_dim=5
    deltaf=fit.evaluate(new_sol)-fit.evaluate(solution)
    zmask=np.ones(solution.shape)
    current_sol=np.ones(solution.shape)
    n=np.ones((sub_dim,1)) #controla moviemiento gradient
    prev_sol=np.copy(solution)
    anchor=np.array([])
    for i in range(100):
        #solution[solution>10]=uniform(-10,10)
        #solution[solution<-10]=uniform(-10,10)
        dif=np.any(prev_sol!=solution,axis=1)
        prev_sol[dif]=solution[dif]
        prev_deltaf=np.copy(deltaf)
        #print(dif)
        r_d=randint(0,2,solution.shape)
        r_d[(np.max(r_d,axis=1)<1)]=np.array([1,1])
        subset=sol.generate_from2(solution,sub_dim,uniform(-1,1,(sub_dim,solution.shape[1]))/(0.01*n),100,anchor=anchor)
        for i in range(solution.shape[0]):
            current_sub=subset[i]
            current_sol[i]=current_sub[np.argmin(fit.evaluate(subset[i]))]
        solution[fit.evaluate(current_sol)<fit.evaluate(solution)]=current_sol[fit.evaluate(current_sol)<fit.evaluate(solution)]
        dif=np.any(prev_sol!=solution,axis=1)
        delta[dif]=(solution-prev_sol)[dif]
        deltaf[dif]=(fit.evaluate(solution)-fit.evaluate(prev_sol))[dif]

        try:
            anchor=np.concatenate((anchor,solution[np.abs(np.sum(delta,axis=1))<0.00001])) #limitar anchors
        except Exception as e:
            anchor=solution[np.abs(np.sum(delta,axis=1))<0.00001]
            print(e)
        print("anchor ",anchor)
        ax.scatter(solution[:,0],solution[:,1],color='r',alpha=0.5)
        n=n+2
    return solution,anchor

X = np.arange(-10, 10, 0.1)
Y = np.arange(-10, 10, 0.1)
X,Y=np.meshgrid(X,Y)
Z=X**2 + Y**2 + (25 * (np.sin(X)**2 + np.sin(Y)**2))
#Z=np.cos(np.sqrt(X**2+Y**2))*np.sin(X/2+4)
fig,ax=plt.subplots(1,1)
ax.contourf(X, Y, Z,100)
ax.autoscale(False)
ax.scatter(solution[:,0],solution[:,1],color='g',alpha=1)
print(solution)
solution,anchor=mine(solution)

ax.scatter(solution[:,0],solution[:,1],color='b',alpha=1)
ax.scatter(anchor[:,0],anchor[:,1],color='w',alpha=1)
print(np.min(fit.evaluate(solution)))
plt.show()
