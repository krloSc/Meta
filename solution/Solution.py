import numpy as np
from numpy.random import randint,rand, uniform
class Solution():
    def __init__(self) -> np.ndarray:
        """Initialize an empty array of solutions"""

        self.sol=np.array([0,0],dtype=float)

    def init_solution(self, x: int, y: int) -> np.ndarray:
        """Generate a random array of solutions"""

        sol = uniform(-1,1,(x,y))*10
        return sol

    def generate_from(
                        self,
                        sol: np.ndarray,
                        new_solutions: int,
                        entrophy: float
                        ) -> np.ndarray:
        """Generate a random array of solutions arround every given solution"""

        number_sol=sol.shape[0]
        dimn=sol.shape[1]
        solutions=np.zeros((number_sol,int(new_solutions),dimn),dtype=float)
        for x in range(number_sol):
            solutions[x]=sol[x]+sol[x]*entrophy
        return solutions

    def update_sol(self, solutions: np.ndarray, slopes: np.ndarray) -> np.ndarray:
        """Update a solution according to a rate of change"""

        solutions=solutions+slopes
        return(solutions)
