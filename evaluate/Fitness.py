import numpy as np
import pandas as pd
import os

class Fitness():
    def __init__(self) -> None:
        pass

    def evaluate(self, solutions: np.ndarray, problem) -> float:
        """Evaluate the problem's fitness function"""

        try:
            #for population
            X=solutions[:,0]
            Y=solutions[:,1]
        except:
            #for single solution
            X=solutions[0]
            Y=solutions[1]

        if (problem.__class__.__name__ == "SpaceProblem"):
            Z=eval(problem.problem)
        else:
            Z=pd.problem.problem[X][Y]

        return Z
