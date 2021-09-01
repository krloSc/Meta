from Metaheuristics import *
from problem.Problem import*
from evaluate import Evaluate
from util import map
import numpy as np

problem = RasterProblem("PVOUT",OptimizationType.MAXIMIZATION)
problem.get_values_from_file()
evaluate = Evaluate.Evaluate()
parameters = {"to":1500,"ta":0.0001,"delta":0.95}
simulated_one = Simulated.Simulated((1,2),problem.optimization_type, parameters)
fireworks = Fwa.Fwa((20,2),problem.optimization_type)
pso = Pso.Pso((50,2),problem.optimization_type)

metas = [
            pso,
            fireworks,
            simulated_one
        ]

evaluate.eva(metas,problem,5)
evaluate.analysis()
evaluate.visual_raster()
print(problem.problem)
#evaluate.visual()
input("Press Enter key to exit...")
