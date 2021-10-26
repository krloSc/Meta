from Metaheuristics import *
from problem.Problem import*
from evaluate import Evaluate
from util import param
import numpy as np
import matplotlib.pyplot as plt

problem = RasterProblem("Falcon",OptimizationType.MAXIMIZATION)
problem.get_values_from_file()
evaluate = Evaluate.Evaluate()
parameters = {"to":1000,"ta":0.01,"delta":0.99}
simulated_one = Simulated.Simulated((1,2),problem, parameters)
fireworks = Fwa.Fwa((10,2),problem,parameters="ga_parameters_v1")
pso = Pso.Pso((100,2),problem,parameters="ga_parameters_v1")
ga = Ga.Ga((5,2),problem,parameters="ga_parameters_v1")
hill = Hill.HillClimbing((50,2),problem,parameters="ga_parameters_v1")
hybrid = HybridGa.HybridGa((5,2),problem,parameters="ga_parameters_v1")
metas = [
            pso,
            simulated_one,
            ga,
            pso
        ]
evaluate.eva(metas,problem,5)
evaluate.plot_graphs() #alpha
evaluate.analysis(detailed=True)
evaluate.visual_raster()
input("Press Enter key to exit...")
