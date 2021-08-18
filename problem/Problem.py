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


class SpaceProblem(Problem):
    """Space problem definition"""

    def get_values_from_file(self) -> None:
        """"Get raster data values from problem files"""

        path=os.getcwd()
        file=open(path+"\\problem\\"+self.name+".prob",'r')
        lst=file.read().split('\n')
        self.problem=lst[0]
        self.x_min=float(lst[1])
        self.x_max=float(lst[2])
        self.y_min=float(lst[3])
        self.y_max=float(lst[4])
        file.close()
        return


class RasterProblem(Problem):
    """Space problem definition"""

    def get_digit(self,doc_line: str) -> int:
        """ return the first digit found in a string"""

        try:
            digit = [value for value in re.findall(r'-?\d+', doc_line)]
            return digit[0]
        except:
            # in case nondata value is not a number
            digit = doc_line.split()
            return digit[1]

    def get_values_from_file(self):
        """Get raster data values from ASC files"""
        path=os.getcwd()
        file=open(path+"\\problem\\"+self.name+".ASC",'r')
        lst=file.read().split('\n')
        self.column=self.get_digit(lst[0])
        self.rows=self.get_digit(lst[1])
        self.x_left=self.get_digit(lst[2])
        self.y_below=self.get_digit(lst[3])
        self.cellsize=self.get_digit(lst[4])
        self.nondata=self.get_digit(lst[5])
        file.close()
        self.problem = pd.read_csv(path+"\\problem\\"+"PVOUT.ASC",
                                    skiprows=6,
                                    encoding="gbk",
                                    engine='python',
                                    sep=' ',
                                    delimiter=None,
                                    index_col=False,
                                    header=None,
                                    skipinitialspace=True)

        self.problem = self.problem[::-1]
        return

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
