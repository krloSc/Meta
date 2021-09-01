from problem.Problem import*
from Metaheuristics import *
import numpy as np
import matplotlib.pyplot as plt
from math import floor
class Evaluate():

    def __init__(self):
        pass

    def eva (self,metas,problem,epoch=5):
        """Obtain an array of results and print the best solutions"""

        if problem.optimization_type == OptimizationType.MINIMIZATION:
            self.best_value = min
            self.best_index = np.argmin
        else:
            self.best_value = max
            self.best_index = np.argmax

        self.problem = problem
        self.metas=metas
        position=np.ones((epoch,2*len(self.metas)))
        self.fitness=np.ones((epoch,len(self.metas)), dtype = float)
        self.results=np.ones((len(metas),epoch,3), dtype = float)
        porc=epoch*len(self.metas)
        p=0

        for i in range(epoch):
            n=0
            for meta, j in zip(self.metas,range(len(self.metas))): #ennumerate
                resul,fit=meta.run(problem)
                position[i,n:n+2]=resul
                self.results[j,i,0:2]=resul
                self.fitness[i,j]=fit
                self.results[j,i,2]=fit
                n=n+2
                print(f"{p*100/porc:.2f}%")
                p=p+1

        self.position = position
        print("______________________________")
        print("Metaheuristic \t Best Solution")
        print("______________________________")
        self.best_fit = np.ones((len(self.metas),1))

        for i in range(len(self.metas)):
            self.best_fit[i] = self.best_value(self.fitness[:,i])
            print(f'{self.metas[i].__class__.__name__:<15}',
            ":\t ",
            f'{self.best_fit[i][0]:^15.13f}')

        return

    def visual(self) -> None:
        """Visual output for Space-like problems"""

        x_min = self.problem.boundaries["y_min"]
        x_max = self.problem.boundaries["y_max"]
        y_min = self.problem.boundaries["x_min"]
        y_max = self.problem.boundaries["x_max"]
        X = np.arange(x_min, x_max, 0.1)
        Y = np.arange(y_min, y_max, 0.1)
        X,Y=np.meshgrid(X,Y)
        Z=eval(self.problem.problem)
        fig,ax=plt.subplots(1,1)
        ax.contourf(X,Y,Z,10)
        ax.autoscale(False)
        for i in range(0,2*len(self.metas),2):
            ax.scatter(
                        self.position[:,i+1],
                        self.position[:,i],
                        label=self.metas[floor(i/2)].__class__.__name__,
                        alpha=1,
                        zorder=1
                        )
        ax.legend()
        plt.show()
        return

    def visual_raster(self) -> None:
        """Visual output for Raster-like problems"""

        Z=self.problem.problem.values
        fig,ax=plt.subplots(1,1)
        ax.contourf(Z,10)
        ax.autoscale(False)
        for i in range(0,2*len(self.metas),2):
            ax.scatter(
                        self.position[:,i+1],
                        self.position[:,i],
                        label=self.metas[floor(i/2)].__class__.__name__,
                        alpha=1,
                        zorder=1
                        )
        ax.legend()
        plt.show()
        return

    def analysis(self):
        """Print statistical analysis of Metaheuristics' performance"""

        fit_index=self.best_index(self.results[:,:,2],axis=1)
        global_fit=self.best_index(self.results[range(len(self.metas)),fit_index,:][:,2])
        print("______________________________")
        print("\tAnalysis")
        print("______________________________")
        print("Best solution:\t",self.results[global_fit,fit_index[global_fit],2])
        print("At:\ positiontion x,y",self.results[global_fit,fit_index[global_fit],::-1] ) ## fix x,y pos
        print("______________________________")
        print(f"{'Metaheuristic':^15}\t"
                f"{'Best solution':^15}\t"
                f"{'Std':^15}\t"
                f"{'Error(mean)':^15}\t"
                f"{'Time taken':^15}")

        for i in range(len(self.metas)):
            index=self.best_index(self.results[i,:,2]) #Menor entre cada epoch
            name=self.metas[i].__class__.__name__
            x_position=self.results[i,index,0]
            y_position=self.results[i,index,1]
            best_sol=self.results[i,index,2]
            std=np.std(self.results[i,:,2])
            error=np.abs(np.mean(self.results[i,:,2])-global_fit)
            time=self.metas[i].time_taken
            print(f'{name:^15.13}\t'
                    f'{best_sol:^15.4e}\t'
                    f'{std:^15.4e}\t'
                    f'{error:^15.4f}\t'
                    f'{time:>8.4f} sec'
                    )
        return
