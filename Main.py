from Metaheuristics import *
from problem.Problem import*
from evaluate.Evaluate import *
from util import map
problem=SpaceProblem("functions",OptimizationType.MINIMIZATION)
problem.get_values_from_file()
evaluate=Evaluate()
parameters={"to":1500,"ta":0.0001,"delta":0.95}
x=Simulated.Simulated((4,2),parameters)
y=Simulated.Simulated((1,2))
z=Fwa.Fwa((20,2))
aa=Pso.Pso((500,2))
bb=Mine.Mine((5,2))
metas = [x,y,z,aa,bb]
evaluate.eva(metas,problem,5,visual=True)
evaluate.analysis()
