from problem.Problem import*
from Metaheuristics import *
import math
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from datetime import datetime
import os
from util import map

class Evaluate():

    def __init__(self):
        pass

    def eva (self,metas,problem,epoch=5):
        """Obtain an array of results and print the best solutions"""

        self.x_min = problem.boundaries["y_min"]
        self.x_max = problem.boundaries["y_max"]
        self.y_min = problem.boundaries["x_min"]
        self.y_max = problem.boundaries["x_max"]

        if problem.optimization_type == OptimizationType.MINIMIZATION:
            self.best_value = min
            self.best_index = np.argmin
        else:
            self.best_value = max
            self.best_index = np.argmax

        self.problem = problem
        self.metas=metas
        self.results=np.ones((len(metas),epoch,3), dtype = float)
        self.graphs = []

        for i in range(epoch):
            for j, meta in enumerate(metas):
                print(f"Running {meta.__class__.__name__:} {i+1}/{epoch}")
                resul,fit=meta.run(problem)
                self.results[j,i,0:2]=resul
                self.results[j,i,2]=fit
                self.graphs.append(meta.lines)
                meta.lines = []

        print("______________________________")
        print("Metaheuristic \t Best Solution")
        print("______________________________")
        self.best_fit = np.ones((len(self.metas),1))

        for i in range(len(self.metas)):
            self.best_fit[i] = self.best_value(self.results[i,:,2])
            print(f'{self.metas[i].__class__.__name__:<15}',
            ":\t ",
            f'{self.best_fit[i][0]:^15.13f}')
        return

    def visual(self) -> None:
        """Visual output for Space-like problems"""

        X = np.arange(self.x_min, self.x_max, 0.1)
        Y = np.arange(self.y_min, self.y_max, 0.1)
        X,Y=np.meshgrid(X,Y)
        Z=eval(self.problem.problem)
        fig,ax=plt.subplots(1,1)
        ax.contourf(X,Y,Z,10)
        ax.autoscale(False)
        for i in range(len(self.metas)):
            ax.scatter(
                        self.results[i,:,1],
                        self.results[i,:,0],
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
        ax.contourf(Z,50, cmap = "YlOrBr")
        ax.autoscale(False)

        for i in range(len(self.metas)):
            ax.scatter(
                        self.results[i,:,1],
                        self.results[i,:,0],
                        label=self.metas[floor(i/2)].__class__.__name__,
                        alpha=1,
                        zorder=1
                        )
#marker=r'$\clubsuit$',
#s=40,
        ax.scatter(
                    self.problem.sub_stations_index[:,0],
                    self.problem.sub_stations_index[:,1],
                    alpha=0.5,
                    zorder=1,
                    color = 'w'
                    )

        ax.legend()
        plt.show()
        return

    def plot_graphs(self):
        """Plot the fitness for each Metaheuristic"""

        if math.sqrt(len(self.metas)) == int(math.sqrt(len(self.metas))):

            grid = int(math.sqrt(len(self.metas)))
            fig, axs = plt.subplots(grid,grid)

        elif len(self.metas)%2 == 0:

            fig, axs = plt.subplots(2,int(len(self.metas)/2))

        else:

            fig, axs = plt.subplots(1, int(len(self.metas)))

        for meta, ax, i in zip(self.metas, axs.flatten(), range(len(self.metas))):
            ax.plot(self.graphs[i])
            ax.set_title(meta.__class__.__name__, fontsize=8)

        plt.show()

        pass

    def analysis(self, detailed = False):
        """Print statistical analysis of Metaheuristics' performance"""

        date = datetime.today()
        path = os.getcwd()
        file_name = (str(date.year)
                +"_"+str(date.month)
                +"_"+str(date.day)
                +"_"+str(date.hour)
                +str(date.minute)
                +str(date.second)
                +"_"+"log")

        file = open(path+"/results/"+file_name+".txt","w+")
        fit_index=self.best_index(self.results[:,:,2],axis=1)
        global_fit=self.best_index(self.results[range(len(self.metas)),fit_index,:][:,2])
        file.write("______________________________")
        file.write("\tAnalysis")
        file.write("______________________________\n")
        global_fitness = self.results[global_fit,fit_index[global_fit],2]
        file.write(f"Best solution:\t{global_fitness}\n")
        x_position = self.results[global_fit,fit_index[global_fit]][1]
        y_position = self.results[global_fit,fit_index[global_fit]][0]

        if isinstance(self.problem, RasterProblem):
            y_position, x_position = self.problem.get_coordinates(np.array([[x_position,y_position]]))
            x_position = x_position[0]
            y_position = y_position[0]


        file.write(f"At: {y_position, x_position}\n") #lat & long
        file.write("____________________________________________________________________\n")
        file.write(f"{'Metaheuristic':^15}\t"
                f"{'Best solution':^15}\t"
                f"{'Std':^15}\t"
                f"{'Error(mean) respect to best':^24}\t"
                f"{'Time taken':^15}\n")

        for i in range(len(self.metas)):
            index=self.best_index(self.results[i,:,2])
            name=self.metas[i].__class__.__name__
            x_position=self.results[i,index,0]
            y_position=self.results[i,index,1]
            best_sol=self.results[i,index,2]
            std=np.std(self.results[i,:,2])
            error=np.abs(np.mean(self.results[i,:,2])-global_fitness)
            time=self.metas[i].time_taken
            file.write(f'{name:^15.13}\t'
                    f'{best_sol:^15.4}\t'
                    f'{std:^15.4e}\t'
                    f'{error:^24.4f}\t'
                    f'{time:>8.4f} sec\n'
                    )
        if detailed:
            file.write("\n")
            file.write(f'{"":_<35} Detailed analysis {"":_>35}\n')
            for i in range(len(self.metas)):
                parameters = self.metas[i].parameters
                fitness = self.results[i,:,2]
                index=self.best_index(fitness)
                x_pos=self.results[i,index,0]
                y_pos=self.results[i,index,1]
                if isinstance(self.problem, RasterProblem):
                    x_pos, y_pos = self.problem.get_coordinates(np.array([[x_pos,y_pos]]))
                    x_pos = x_pos[0]
                    y_pos = y_pos[0]
                best_sol=self.results[i,index,2]
                number_occurrence = np.sum(fitness==best_sol)
                std = np.std(fitness)
                mean = np.mean(fitness)
                imprecision = np.mean(np.abs(fitness-mean))

                file.write("\n")
                file.write(f'{"Name:":<25} {self.metas[i].__class__.__name__ : <15}\n')
                file.write(f'{"Best solution:":<25} {best_sol:<15.7}\n')
                file.write(f'{"Location:":<25} {x_pos:<15}\t{y_pos:<15}\n')
                file.write(f'{"Number of Ocurrence:":<25} {number_occurrence:<15}\n')
                file.write(f'{"Mean:":<25} {mean:<15}\n')
                file.write(f'{"Standard deviation:":<25} {std:<15}\n')
                file.write(f'{"imprecision:":<25} {imprecision:<15}\n')
                file.write("\n")

                file.write(f'{"Parameters used":^30}\n')
                for key, value in zip(parameters.keys(), parameters.values()):
                    file.write(f'{key:<25} {value:<15}\n')

                file.write(f'{"":_>85}\n')


        file.seek(0,0) # set the pointer to the begining
        text_result = file.read().split("\n")
        for lines in text_result:
            print(lines)
        file.close()
        return
