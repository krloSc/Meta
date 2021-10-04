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
        file=open(path+"/problem/"+self.name+".prob",'r')
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
            digit = doc_line.split()
            return digit[1]

    def get_values_from_file(self) -> None:
        """Get raster data values from .asc files"""

        path=os.getcwd()
        file=open(path+"/problem/"+self.name+".asc",'r')
        lst=file.read().split('\n')
        self.columns=self.get_digit(lst[0])
        self.rows=self.get_digit(lst[1])
        self.x_left=self.get_digit(lst[2])
        self.y_below=self.get_digit(lst[3])
        self.cellsize=self.get_digit(lst[4])
        self.nondata=self.get_digit(lst[5])
        file.close()
        self.problem = pd.read_csv(path+"/problem/"+"PVOUT.asc",
                                    skiprows=6,
                                    encoding="gbk",
                                    engine='python',
                                    sep=' ',
                                    delimiter=None,
                                    index_col=False,
                                    header=None,
                                    skipinitialspace=True
                                    )

        self.problem = self.problem[::-1]
        self.problem = self.problem.replace(np.NaN,0) #//cambiar
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

        peak_power = 3000 #kwp
        implementation_cost = 1250 # $/kwp
        powerline_cost = 177000 # aprox per km (69kv)
        life_span = 25
        distance = np.sqrt(del_x + del_y) #Unit? Coger la minima
        nominal_discount_rate = 0.08
        inflation_rate = 0.05
        loan_duration = 10
        loan_interest = 0.09
        selling_price = 0.15
        energy_sold = 4500000 #justificar valor produccion anual
        actual_discount_rate = (nominal_discount_rate-inflation_rante)/(1+inflation_rate)
        ki = 1/(1+actual_discount_rate)
        kg = (1+inflation_rate)/(1+actual_discount_rate)
        maintenance = 14*peak_power
        maintenance_increase_rate =0.01
        kmo = (1+maintenance_increase_rate)/(1+actual_discount_rate)

        investment = implementation_cost*peak_power + distance*powerline_cost

        annual_payment = 0.8*investment*((loan_interest*(1+loan_interest)**loan_duration)/(((1+loan_interest)**loan_duration)-1))

        present_value = 0.2*investment + annual_payment*((ki*(1-ki)**life_span)/(1-ki**life_span))

        present_income = selling_price*energy_sold*(kg*(1-kg**life_span)/(1-kg))

        present_cashout = maintenance*((1-kmo**life_span)/(1-kmo))

        net_value = present_income - present_cashout - present_value


        Z = net_value
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
