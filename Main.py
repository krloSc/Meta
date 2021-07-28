from Metaheuristics import *
from problem.Problem import*
from evaluate.Evaluate import *
problem=Problem("functions","space")
evaluate=Evaluate()
parameters=[{"to":1000,"ta":0.001,"delta":0.99}]
x=Simulated.Simulated((4,2))#,parameters)
y=Simulated.Simulated((1,2))
z=Fwa.Fwa((20,2))
aa=Pso.Pso((500,2))
bb=Mine.Mine((5,2))
metas = [x,y,z,aa,bb]
evaluate.eva(metas,problem,5,visual=True)
evaluate.analysis()
