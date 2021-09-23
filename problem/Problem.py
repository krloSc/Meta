from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
import numpy as np
import pandas as pd
import os
import re

class OptimizationType(Enum):
    """Assign ids to optimization problem types"""

    MAXIMIZATION = auto()
    MINIMIZATION = auto()

@dataclass
class Problem(ABC):
    """Problem definition"""

    name: str
    optimization_type: OptimizationType = OptimizationType.MAXIMIZATION

    @abstractmethod
    def get_values_from_file(self) -> None:
        pass

    @abstractmethod
    def eval_fitness_function(self, solutions: np.ndarray) -> np.ndarray:
        pass


class SpaceProblem(Problem):
    """Space problem definition"""

    def get_values_from_file(self) -> None:
        """"Get raster data values from problem files"""

        path=os.getcwd()
        file=open(path+"\\problem\\"+self.name+".prob",'r')
        lst=file.read().split('\n')
        self.problem=lst[0]
        x_min=float(lst[1])
        x_max=float(lst[2])
        y_min=float(lst[3])
        y_max=float(lst[4])
        self.boundaries = {
                            "x_min" : x_min,
                            "x_max" : x_max,
                            "y_min" : y_min,
                            "y_max" : y_max,
                        }
        file.close()
        return

    def eval_fitness_function(self, solutions: np.ndarray) -> np.ndarray:
        """Evaluate the problem's fitness function"""
        X=solutions.reshape(-1,2)[:,0]
        Y=solutions.reshape(-1,2)[:,1]
        Z=eval(self.problem)
        return Z


class RasterProblem(Problem):
    """Space problem definition"""

    sub_stations = np.random.randint(600,1200,(50,2))

    def get_digit(self, doc_line: str) -> int:
        """ return the first digit found in a string"""

        try:
            digit = [value for value in re.findall(r'-?\d+', doc_line)]
            return float(digit[0])
        except:
            # in case nondata value is not a number
            return np.NaN

    def get_values_from_file(self) -> None:
        """Get raster data values from ASC files"""

        path=os.getcwd()
        file=open(path+"\\problem\\"+self.name+".ASC",'r')
        lst=file.read().split('\n')
        self.columns=self.get_digit(lst[0])
        self.rows=self.get_digit(lst[1])
        self.x_left=self.get_digit(lst[2])
        self.y_below=self.get_digit(lst[3])
        self.cellsize=self.get_digit(lst[4])
        self.nondata=self.get_digit(lst[5])
        file.close()
        self.problem = pd.read_csv(path+"\\problem\\"+"PVOUT.ASC",
                                    skiprows=6,
                                    encoding="UTF-8",
                                    engine='python',
                                    sep=' ',
                                    delimiter=None,
                                    index_col=False,
                                    header=None,
                                    skipinitialspace=True
                                    )

        self.problem = self.problem[::-1]
        self.problem = self.problem.replace(self.nondata,0)
        self.boundaries = {
                            "x_min" : 0,
                            "x_max" : self.rows-1,
                            "y_min" : 0,
                            "y_max" : self.columns-1,
                        }
        return

    def eval_fitness_function(self, solutions: np.ndarray) -> np.ndarray:
        """Evaluate the problem's fitness function"""

        solutions = solutions.reshape(-1,2)
        X=np.rint(solutions.reshape(-1,2)[:,0])
        Y=np.rint(solutions.reshape(-1,2)[:,1])
        try:
            Z=np.diag(self.problem.iloc[X,Y])
        except:
            Z=self.problem.iloc[X,Y]

        delta_x = np.zeros((solutions.shape[0],self.sub_stations.shape[0]))
        delta_y = np.zeros((solutions.shape[0],self.sub_stations.shape[0]))
        for i in range(solutions.shape[0]):
            delta_x[i] = solutions[i,0] - self.sub_stations[:,0]
            delta_y[i] = solutions[i,1] - self.sub_stations[:,1]
        del_x = delta_x**2
        del_y = delta_y**2
        distance = np.sqrt(del_x + del_y)
        Z = Z #- np.min(distance, axis = 1)/10
        return Z


def main() -> None:

    problem_one = SpaceProblem("functions", OptimizationType.MINIMIZATION)
    problem_one.get_values_from_file()
    print(problem_one.problem)
    problem_two = RasterProblem("PVOUT", OptimizationType.MAXIMIZATION)
    problem_two.get_values_from_file()
    print(problem_two)
    return

if __name__ == "__main__":
    main()
