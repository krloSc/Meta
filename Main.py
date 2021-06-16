from Metaheuristics import *
from problem.Problem import*

problem=Problem("ground")
#parameters={"to":1000,"ta":0.001,"delta":0.99}
x=Simulated.Simulated_A((4,3))#,parameters)
solution=x.simulated(problem)
print(solution)
