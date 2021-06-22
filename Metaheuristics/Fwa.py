from evaluate.Fitness import *
from solution.Solution import *
import numpy as np
from numpy.random import rand,uniform,randint
import matplotlib.pyplot as plt
fit=Fitness()
sol=Solution()

class Fwa():

    def __init__(self,size,parameters=[]):
        if parameters==[]:
            try:
                path=os.getcwd()
                file=open(path+"\\Metaheuristics\\"+"Simulated"+".param",'r')
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

    def fwa(solution):
        e=0.001
        s_hat=100
        a_hat=10
        best=min(fit.evaluate(solution))
        worst=np.max(fit.evaluate(solution))
        xmin=1
        xmax=10
        for i in range(20):
            for i in range(solution.shape[0]): #numero de fireworks
                s=np.rint(s_hat*(worst-fit.evaluate(solution)+e)/(np.sum(worst-fit.evaluate(solution))+e))
                s[s<xmin]=xmin
                s[s>xmax]=xmax
                a=a_hat*(fit.evaluate(solution)-best+e)/(np.sum(fit.evaluate(solution)-best)+e)
                r_d=randint(0,2,(int(s[i]),2))
                r_d[(np.max(r_d,axis=1)<1)]=np.array([1,1]) #al menos alguno de los dos debe actualizarse
                upd=a[i]*r_d*uniform(-1,1,r_d.shape)                       # (s,2)
                variable=sol.generate_from2(solution[i,:].reshape(1,-1),s[i],upd)
                try:
                    solutions=np.concatenate((solutions,variable[0]))
                except:
                    solutions=variable[0]
            solutions=np.concatenate((solutions,solution))
            bindex=np.argmin(fit.evaluate(solutions))
            best=min(fit.evaluate(solutions))
            worst=max(fit.evaluate(solutions))
            best_spark=solutions[bindex].reshape(1,-1)
            solutions=np.delete(solutions,bindex,0)
            n_minus=nfire(solutions)
            solution=np.concatenate((best_spark,n_minus))
        print(best_spark,fit.evaluate(best_spark))
        return solution

    solution=fwa(solution)
    X = np.arange(-25, 25, 0.1)
    Y = np.arange(-25, 25, 0.1)
    X,Y=np.meshgrid(X,Y)
    #Z=X**2 + Y**2 + (25 * (np.sin(X)**2 + np.sin(Y)**2))
    Z=0.5+((np.sin(np.sqrt(X**2+Y**2))**2-0.5)/(1+0.001*(X**2+Y**2))**2)
    #Z=np.cos(np.sqrt(X**2+Y**2))*np.sin(X/2+4)
    fig,ax=plt.subplots(1,1)
    ax.contourf(X, Y, Z,100)
    ax.autoscale(False)
    ax.scatter(solution[:,0],solution[:,1],color='r',alpha=1,zorder=1)

    #print(solution)
    plt.show()
