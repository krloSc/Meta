from Metaheuristics import *
from problem.Problem import*
from evaluate import Evaluate
from util import param
import numpy as np
import matplotlib.pyplot as plt

problem = SpaceProblem("himmeblau",OptimizationType.MAXIMIZATION)
problem.get_values_from_file()
simulated = Simulated.Simulated((2,2),problem, parameters= "simulated")
fireworks = Fwa.Fwa((5,2),problem,parameters="firework")
pso = Pso.Pso((60,2),problem,parameters="PSO")
ga = Ga.Ga((5,2),problem,parameters="ga_parameters_v1")
ga2 = GA_MOD.Ga((10,2),problem,parameters="Ga_v2")
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
evaluate = Evaluate.Evaluate(metas,problem,epoch=1)
evaluate.eva()
evaluate.analysis(detailed=True)
evaluate.plot_graphs()
evaluate.visual()
input("Press Enter key to exit...")
