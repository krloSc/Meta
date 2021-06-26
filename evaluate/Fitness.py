import numpy as np
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
        Z=eval(problem.problem)
        return Z
