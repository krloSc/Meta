from evaluate.Fitness import *
from solution.Solution import *
import numpy as np
from numpy.random import rand,uniform
import matplotlib.pyplot as plt
fit=Fitness()
sol=Solution()

print("PSO Metaheuristic")

############### Initial Population #################
swarm_size=500
dimension=2
particles=sol.init_solution(swarm_size,dimension)
init_particles=particles #eliminar
current_fitness=fit.evaluate(particles)
best_particle=particles[np.argmin(current_fitness)]
velocity=uniform(0,1,swarm_size*dimension)
velocity=velocity.reshape(-1,dimension)
best_particles=particles
################## Evaluation ######################
def pso(particles,criteria):
    global velocity, best_particle, best_particles
    for i in range(criteria):
        r1=uniform(0,1,swarm_size).reshape(-1,1)
        r2=uniform(0,1,swarm_size).reshape(-1,1)
        velocity=0.5*velocity+r1*0.1*(best_particles-particles)+r2*0.5*(best_particle-particles)
        particles=sol.update_sol(particles,velocity) #poblacion actualizada
        #print(particles)
        current_fitness=fit.evaluate(particles)
        #print(current_fitness)
        if (fit.evaluate(particles[np.argmin(current_fitness)])<fit.evaluate(best_particle)):
            best_particle=particles[np.argmin(current_fitness)] #
        #print(fit.evaluate(best_particles))
        best_particles[current_fitness<fit.evaluate(best_particles)]=particles[current_fitness<fit.evaluate(best_particles)]
    print("-------------------------------- \n")
    print(fit.evaluate(best_particle))




X = np.arange(-10, 10, 0.1)
Y = np.arange(-10, 10, 0.1)
X,Y=np.meshgrid(X,Y)
#Z=X**2 + Y**2 + (25 * (np.sin(X)**2 + np.sin(Y)**2))
Z=0.5+((np.sin(np.sqrt(X**2+Y**2))**2-0.5)/(1+0.001*(X**2+Y**2))**2)
#Z=np.cos(np.sqrt(X**2+Y**2))*np.sin(X/2+4)
fig,ax=plt.subplots(1,1)
ax.contourf(X, Y, Z,100)
ax.autoscale(False)
pso(particles,100)
ax.scatter(best_particles[:,0],best_particles[:,1],color='r',alpha=1,zorder=1)
ax.scatter(init_particles[:,0],init_particles[:,1],color='r',alpha=0.3,zorder=1)
plt.show()
