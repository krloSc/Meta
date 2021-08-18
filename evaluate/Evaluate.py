from Metaheuristics import *
import numpy as np
import matplotlib.pyplot as plt
from math import floor
class Evaluate():

    def __init__(self):
        pass

    def eva (self,metas,problem,epoch=5,visual=False):
        """Obtain an array of results and print the best solutions"""

        self.metas=metas
        pos=np.ones((epoch,2*len(self.metas)))
        self.fitness=np.ones((epoch,len(self.metas)))
        self.results=np.ones((len(metas),epoch,3))
        porc=epoch*len(self.metas)
        p=0
        for i in range(epoch):
            n=0
            for meta, j in zip(self.metas,range(len(self.metas))):
                resul,fit=meta.run(problem)
                pos[i,n:n+2]=resul
                self.results[j,i,0:2]=resul
                self.fitness[i,j]=fit
                self.results[j,i,2]=fit
                n=n+2
                print(f"{p*100/porc:.2f}%")
                p=p+1

        print("______________________________")
        print("Metaheuristic \t Best Solution")
        print("______________________________")
        self.best_fit=np.ones((len(self.metas),1))
        for i in range(len(self.metas)):
            self.best_fit[i]=np.min(self.fitness[:,i])
            print(f'{self.metas[i].__class__.__name__:<15}',
            ":\t ",
            f'{self.best_fit[i][0]:^15.13f}')
        if visual:
            self.visual(problem, pos)

    def visual(self, problem, pos) -> None:
            X = np.arange(problem.x_min, problem.x_max, 0.1)
            Y = np.arange(problem.y_min, problem.y_max, 0.1)
            X,Y=np.meshgrid(X,Y)
            Z=eval(problem.problem)
            fig,ax=plt.subplots(1,1)
            ax.contourf(X, Y, Z,100)
            ax.autoscale(False)
            for i in range(0,2*len(self.metas),2):
                ax.scatter(pos[:,i],pos[:,i+1],label=self.metas[floor(i/2)].__class__.__name__,alpha=1,zorder=1)
            ax.legend()
            plt.show()

    def analysis(self):
        fit_index=np.argmin(self.results[:,:,2],axis=1)
        global_fit=np.argmin(self.results[range(len(self.metas)),fit_index,:][:,2])
        print("______________________________")
        print("\tAnalysis")
        print("______________________________")
        print("Best solution:\t",self.results[global_fit,fit_index[global_fit],2])
        print("At:\ postion x,y",self.results[global_fit,fit_index[global_fit],0:2] )
        print("______________________________")
        print(f"{'Metaheuristic':^15}\t{'Best solution':^15}\t{'Std':^15}\t{'Error(mean)':^15}\t{'Time taken':^15}")
        for i in range(len(self.metas)):
            index=np.argmin(self.results[i,:,2]) #Menor entre cada epoch
            name=self.metas[i].__class__.__name__
            x_pos=self.results[i,index,0]
            y_pos=self.results[i,index,1]
            best_sol=self.results[i,index,2]
            std=np.std(self.results[i,:,2])
            error=np.abs(np.mean(self.results[i,:,2])-global_fit)
            time=self.metas[i].time_taken
            #print(name,"\t",x_pos,y_pos,"\t",best_sol,std,error,time,"sec")
            print(f'{name:^15.13}\t'
                    f'{best_sol:^15.4e}\t'
                    f'{std:^15.4e}\t'
                    f'{error:^15.4f}\t'
                    f'{time:>8.4f} sec'
                    )
