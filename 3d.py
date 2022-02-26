from Metaheuristics import *
from problem.Problem import*
from evaluate import Evaluate
from util import param
import numpy as np
import matplotlib.pyplot as plt

problem = SpaceProblem("himmeblau",OptimizationType.MAXIMIZATION)
problem.get_values_from_file()
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = eval(problem.problem)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('Inverted Himmelblau');
plt.show()
input("Press Enter key to exit...")
