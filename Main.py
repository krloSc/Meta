from Metaheuristics import *
from problem.Problem import*
from evaluate import Evaluate
from util import param
import numpy as np
import matplotlib.pyplot as plt

problem = RasterProblem("Venezuela",OptimizationType.MAXIMIZATION)
problem.get_values_from_file()
evaluate = Evaluate.Evaluate()
simulated = Simulated.Simulated((2,2),problem, parameters= "simulated")
fireworks = Fwa.Fwa((5,2),problem,parameters="firework")
pso = Pso.Pso((100,2),problem,parameters="PSO")
ga = Ga.Ga((5,2),problem,parameters="ga_parameters_v1")
ga2 = GA_MOD.Ga((10,2),problem,parameters="Ga_v2")
ga3 = ga_v3.Ga_v3((6,2),problem,parameters="Ga_v2")
hill = Hill.HillClimbing((60,2),problem,parameters="hill")
hybrid = HybridGa.HybridGa((6,2),problem,parameters="hybrid2")
metas = [
            pso,
            ga3,
            hill,
            hybrid,
            fireworks,
            simulated
        ]
evaluate.eva(metas,problem,20)
evaluate.analysis(detailed=True)
evaluate.plot_graphs() #alpha
evaluate.visual_raster()
input("Press Enter key to exit...")
