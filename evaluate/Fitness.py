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
        #Z=np.cos(np.sqrt(X**2+Y**2))*np.sin(X/2+4)
        #Z=-X*np.sin(np.sqrt(np.abs(X)))-Y*np.sin(np.sqrt(np.abs(Y)))
        #Z=0.5+(np.sin(np.sqrt(X**2+Y**2))**2-0.5)/(1+0.001*(X**2+Y**2))**2
        #Z=X**2 + Y**2 + (25 * (np.sin(X)**2 + np.sin(Y)**2))
        Z=eval(problem.problem)
        return Z
