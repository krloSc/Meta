from Metaheuristics import *
from problem.Problem import*
from evaluate.Evaluate import *
from util import map

problem = SpaceProblem("functions",OptimizationType.MINIMIZATION)
problem.get_values_from_file()
evaluate = Evaluate()
parameters = {"to":1500,"ta":0.0001,"delta":0.95}
simulated_one = Simulated.Simulated((4,2),parameters)
simulated_two = Simulated.Simulated((1,2))
fireworks = Fwa.Fwa((20,2))
pso = Pso.Pso((500,2))
custom_meta = Mine.Mine((5,2))
metas = [simulated_one,
        simulated_two,
        fireworks,
        pso,
        custom_meta]
evaluate.eva(metas,problem,5,visual=True)
evaluate.analysis()
input("Press Enter key to exit...")
