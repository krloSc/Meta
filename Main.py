from Metaheuristics import *
from problem.Problem import*
from evaluate import Evaluate
from util import param
import numpy as np
import matplotlib.pyplot as plt

problem = RasterProblem("Falcon",OptimizationType.MAXIMIZATION)
problem.get_values_from_file()
evaluate = Evaluate.Evaluate()
parameters = {"to":1500,"ta":0.0001,"delta":0.95}
simulated_one = Simulated.Simulated((1,2),problem, parameters)
fireworks = Fwa.Fwa((20,2),problem)
pso = Pso.Pso((5,2),problem)
ga = Ga2.Ga((5,2),problem,parameters="ga_parameters_v1")
metas = [   simulated_one
        ]
evaluate.eva(metas,problem,2)

plt.plot(pso.lines)
plt.ylabel('fitness')
plt.show()
evaluate.analysis()
evaluate.visual_raster()
input("Press Enter key to exit...")
