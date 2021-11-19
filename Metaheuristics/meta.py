from abc import ABC, abstractmethod
from solution.Solution import *
from problem.Problem import*
import numpy as np
from numpy.random import rand,uniform
import matplotlib.pyplot as plt
import time
from util import param

@dataclass
class Metaheuristic(ABC):
    """initilizer function for Ga Metaheuristic"""
    size: tuple
    problem: Problem
    parameters: dict = None
    
    def __post_init__(self):
        if type(self.parameters) == str:
            self.parameters = param.get_parameters(self.parameters)

        elif self.parameters == None:
            print("Parameters not speficied for "+self.__class__.__name__+", using default values")

        self.lines = []

        if  self.problem.optimization_type == OptimizationType.MINIMIZATION:
            self.comparator = np.less
            self.best_index = np.argmin
            self.best_value = min
            self.worst = max
            self.order = -1
        else:
            self.comparator = np.greater
            self.best_index = np.argmax
            self.best_value = max
            self.worst = min
            self.order = 1

    @abstractmethod
    def run():
        pass
