import numpy as np
import pandas as pd
import os

class Fitness():
    def __init__(self):
        pass

    def evaluate(self,solutions,problem):
        try:
            X=solutions[:,0]
            Y=solutions[:,1]
        except:
            X=solutions[0]
            Y=solutions[1]

        if (problem.__class__.__name__ == "SpaceProblem"):
            Z=eval(problem.problem)
        else:
            Z=pd.problem.problem[X][Y]

        return Z
