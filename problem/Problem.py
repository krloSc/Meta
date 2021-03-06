from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from util.Util import *
import numpy as np
import pandas as pd
import os
import re
map = Map()

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
        return Z , Z


class RasterProblem(Problem):
    """Space problem definition"""

    def get_indexes(self, sub_stations: np.ndarray) -> np.ndarray:
        """Convert the coordinates into index integer"""
        xmin = np.array([self.x_left,self.y_below])
        upper_lon = self.x_left + self.columns*self.cellsize
        upper_lat = self.y_below + self.rows*self.cellsize
        xmax = np.array([upper_lon,upper_lat])
        ymin = np.array([0,0])
        ymax = np.array([self.columns, self.rows])

        return map.map(sub_stations,xmin,xmax,ymin,ymax)

    def get_digit(self, doc_line: str) -> int:
        """ return the first digit found in a string"""

        try:
            value = re.search('([0-9]|-).*',doc_line).group()
        except:
            return re.search('\w*$',doc_line).group()

        if value.find(".") != -1: #if value has a decimal point
            value = float(value)
        else:
            value = int(value)

        return value

    def get_values_from_file(self) -> None:
        """Get raster data values from .asc files"""

        path=os.getcwd()
        file=open(path+"/problem/maps/"+self.name+".asc",'r')
        lst=file.read().split('\n')
        self.columns=self.get_digit(lst[0])
        self.rows=self.get_digit(lst[1])
        self.x_left=self.get_digit(lst[2])
        self.y_below=self.get_digit(lst[3])
        self.cellsize=self.get_digit(lst[4])
        self.nondata=self.get_digit(lst[5])
        file.close()

        self.problem = pd.read_csv(path+"/problem/maps/"+self.name+".asc",
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

        file = open(path+"/problem/sub_stations.txt","r")
        self.sub_stations = np.array(file.read().split())
        self.sub_stations = self.sub_stations.astype(np.float)
        self.sub_stations = self.sub_stations.reshape(-1,2)
        self.sub_stations_index = self.get_indexes(self.sub_stations)
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

        subs_lat, subs_lon = self.sub_stations[:,1],self.sub_stations[:,0]
        current_lat, current_lon = self.get_coordinates(solutions)
        distance = []
        for lat, lon in zip(current_lat, current_lon):
            distance.append(np.min(map.get_distance(lat, lon, subs_lat, subs_lon)))

        distance = np.array(distance)
        peak_power = 10000 #kwp
        implementation_cost = 1360 # $/kwp
        powerline_cost = 177000 # aprox per km (69kv)
        life_span = 25
        nominal_discount_rate = 0.08
        inflation_rate = 0.05
        loan_duration = 10
        loan_interest = 0.09
        selling_price = 0.10
        energy_sold = peak_power * Z * 365 #justificar valor produccion anual
        actual_discount_rate = 0.08# (nominal_discount_rate-inflation_rate)/(1+inflation_rate)
        ki = 1/(1+actual_discount_rate)
        kg = (1+inflation_rate)/(1+actual_discount_rate)
        maintenance = 17 * peak_power
        maintenance_increase_rate =0.01
        kmo = (1+maintenance_increase_rate)/(1+actual_discount_rate)
        investment = implementation_cost*peak_power + distance*powerline_cost

        annual_payment = 0.8*investment*((loan_interest*(1+loan_interest)**loan_duration)/(((1+loan_interest)**loan_duration)-1))

        present_value = 0.2*investment + annual_payment*((ki*(1-ki)**life_span)/(1-ki**life_span))

        present_income = selling_price * energy_sold * ((kg*(1-(kg**life_span)))/(1-kg))

        present_cashout = maintenance*((1-kmo**life_span)/(1-kmo))

        net_value = present_income - present_cashout - present_value

        net_value = net_value/1000000 #In MM USD
        return net_value.round(4), energy_sold/1000000

    def get_coordinates(self, solution: np.ndarray) -> np.ndarray:
        """Convert a index value into a coordinate"""
        longitude = self.x_left+solution[:,1]*self.cellsize
        latitude = self.y_below+solution[:,0]*self.cellsize
        return latitude, longitude


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
