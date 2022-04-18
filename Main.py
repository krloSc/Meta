from Metaheuristics import *
from problem.Problem import*
from evaluate import Evaluate
from util import param
import numpy as np
import matplotlib.pyplot as plt

#Creamos una instancia del problema
problem = RasterProblem("Sucre",OptimizationType.MAXIMIZATION)
problem.get_values_from_file()

#Instanciamos las metaheurísticas, indicamos el tamaño, el problema y los
#parametros, en este caso el nombre del archivo que los contiene

simulated = Simulated.Simulated((2,2),problem, parameters= "simulated")
fireworks = Fwa.Fwa((5,2),problem,parameters="firework")
pso = Pso.Pso((60,2),problem,parameters="PSO")
ga3 = ga_v3.Ga_v3((6,2),problem,parameters="Ga_v2")
hill = Hill.HillClimbing((60,2),problem,parameters="hill")
hybrid = HybridGa.HybridGa((6,2),problem,parameters="hybrid2")

metas = [
            pso,
            fireworks,
            simulated,
            ga3,
            hill,
            hybrid

        ]

#Creamos una instancia de Evaluate
evaluate = Evaluate.Evaluate(metas,problem,epoch=5)

evaluate.eva() #Ejecuta las metaheurísticas
evaluate.analysis(detailed=True)    #Realiza el analisis estadístico
evaluate.plot_graphs()
evaluate.visual_raster() #Grafica el mapa y los resultados
input("Press Enter key to exit...")
