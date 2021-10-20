import numpy as np
from numpy.random import randint,rand, uniform

class Solution():
    def __init__(self) -> np.ndarray:
        """Initialize an empty array of solutions"""

        self.sol=np.array([[0,0]], dtype=float) #//modificar urgent

    def init_solution(self, rows: int, columns: int, boundaries: dict) -> np.ndarray:
        """Generate a random array of solutions"""

        self.x_min = boundaries["x_min"]
        self.x_max = boundaries["x_max"]
        self.y_min = boundaries["y_min"]
        self.y_max = boundaries["y_max"]
        sol_x = uniform(self.x_min,self.x_max, size = (rows,1))
        sol_y = uniform(self.y_min,self.y_max, size = (rows,1))
        sol = np.append(sol_x, sol_y, axis = 1)
        return sol

    def generate_from(
                        self,
                        sol: np.ndarray,
                        new_solutions: int,
                        entrophy: float
                        ) -> np.ndarray:
        """Generate a random array of solutions arround every given solution"""

        number_sol=sol.shape[0]
        dimn = sol.shape[1]
        solutions = np.zeros((number_sol,int(new_solutions),dimn),dtype = float)
        for x in range(number_sol):
            solutions[x] = sol[x] + entrophy
        solutions[:,:,0] = np.clip(solutions[:,:,0],self.x_min,self.x_max-1)
        solutions[:,:,1] = np.clip(solutions[:,:,1],self.y_min,self.y_max-1)
        return solutions

    def generate_single(self, origin, randomness = 1):

        solution = origin + np.random.rand(2)*randomness
        solution[0] = np.clip(solution[0],self.x_min,self.x_max-1)
        solution[1] = np.clip(solution[1],self.y_min,self.y_max-1)
        return solution

    def update_sol(self, solutions: np.ndarray, slopes: np.ndarray) -> np.ndarray:
        """Update a solution according to a rate of change"""

        solutions = solutions+slopes
        solutions[:,0] = np.clip(solutions[:,0],self.x_min,self.x_max-1)
        solutions[:,1] = np.clip(solutions[:,1],self.y_min,self.y_max-1)
        return solutions

    def check_boundaries(self,solutions):
        solutions[:,0] = np.clip(solutions[:,0],self.x_min,self.x_max-1)
        solutions[:,1] = np.clip(solutions[:,1],self.y_min,self.y_max-1)
