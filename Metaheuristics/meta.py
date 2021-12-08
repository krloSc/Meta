from abc import ABC, abstractmethod
from solution.Solution import *
from problem.Problem import*
import numpy as np
from numpy.random import rand,uniform
import matplotlib.pyplot as plt
import time
import re
import os
#from util import param

@dataclass
class Metaheuristic(ABC):
    """initilizer function for Ga Metaheuristic"""
    size: tuple
    problem: Problem
    parameters: dict = None

    def __post_init__(self):
        if type(self.parameters) == str:
            self.parameters = self.get_parameters(self.parameters)

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

        self.sol = Solution()
        self.time_taken = []

    def get_parameters(self, file_name: str):
        """Read the parameters file and retrieve the values"""
        try:
            path=os.getcwd()
            file=open(path+"/Metaheuristics/parameters/"+file_name+".param",'r')
            lst=file.read().split('\n')
            parameters = dict()
            for lines in lst:
                parameter =  re.search("\w*", lines).group()
                value =re.search('([0-9]|-).*',lines).group()
                if value.find(".") != -1: #if value has a decimal point
                    value = float(value)
                else:
                    value = int(value)
                parameters[parameter] = value

        except Exception as e:
            print(file_name+" - Parameters file not found - Using default values")
            print(e)
        return parameters

    @abstractmethod
    def run():
        pass
