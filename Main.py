from Metaheuristics import *
from problem.Problem import*
from evaluate import Evaluate
from util import param
import numpy as np
import matplotlib.pyplot as plt

problem = RasterProblem("PVOUT",OptimizationType.MAXIMIZATION)
problem.get_values_from_file()
evaluate = Evaluate.Evaluate()
parameters = {"to":1500,"ta":0.0001,"delta":0.95}
simulated_one = Simulated.Simulated((1,2),problem.optimization_type, parameters)
fireworks = Fwa.Fwa((20,2),problem.optimization_type)
pso = Pso.Pso((5,2),problem.optimization_type)
ga = Ga.Ga((5,2),problem.optimization_type,parameters="ga_parameters_v1")
#problem.boundaries = {
#                    "x_min" : 600,
#                    "x_max" : 1100,
#                    "y_min" : 400,
#                    "y_max" : 800,
#                }
metas = [   pso,
            simulated_one,
            ga,
            fireworks
        ]
evaluate.eva(metas,problem,2)

plt.plot(pso.lines)
plt.ylabel('fitness')
plt.show()
evaluate.analysis()
evaluate.visual_raster()
input("Press Enter key to exit...")
