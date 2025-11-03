from abc import ABC, abstractmethod
from collections.abc import Callable
from pso.swarm import Swarm
from pso.cost import CostBase
import numpy as np

class PSOBase(ABC):
    """
    Abstract base class for Particle Swarm Optimization (PSO).
    """
    def __init__(self, lim: tuple, n_particles: int = 10, max_iter: int = 100, inertia: float = 0.9, c_cognitive: float = 0.5, c_social: float = 0.3, stop_at_variance: float | None = None, lbx: np.ndarray | None = None, ubx: np.ndarray | None = None):
        """
        Initialize the PSO base class with given parameters.
        
        :param lim: Tuple representing the limits for the search space.
        :param n_particles: Number of particles in the swarm.
        :param max_iter: Maximum number of iterations.
        :param inertia: Inertia coefficient.
        :param c_cognitive: Cognitive coefficient.
        :param c_social: Social coefficient.
        """
        self._lim: tuple = lim
        self._max_iter: int = max_iter
        self._inertia: float = inertia
        self._c_cognitive: float = c_cognitive
        self._c_social: float = c_social
        self._n_particles: int = n_particles
        
    @abstractmethod
    def optimize(self) -> None:
        """
        Abstract method to perform optimization.
        """
        pass

    @abstractmethod
    def get_optimal_position(self) -> np.ndarray:
        """
        Abstract method to get the optimal position found by the swarm.
        """
        pass
    
    @abstractmethod
    def get_optimal_cost(self) -> float:
        """
        Abstract method to get the optimal cost found by the swarm.
        """
        pass
    
class PSO(PSOBase):
    """
    Concrete implementation of the PSO algorithm.
    """
    def __init__(self, cost: CostBase, lim: tuple, n_particles: int = 10, max_iter: int = 100, inertia: float = 0.9, c_cognitive: float = 0.5, c_social: float = 0.3, stop_at_variance: float | None = None, lbx: np.ndarray | None = None, ubx: np.ndarray | None = None):
        """
        Initialize the PSO algorithm with given parameters.
        
        :param cost: Cost function to be minimized.
        :param lim: Tuple representing the limits for the search space.
        :param n_particles: Number of particles in the swarm.
        :param max_iter: Maximum number of iterations.
        :param inertia: Inertia coefficient.
        :param c_cognitive: Cognitive coefficient.
        :param c_social: Social coefficient.
        """
        super().__init__(lim, n_particles, max_iter, inertia, c_cognitive, c_social)
        self._cost_fn: CostBase = cost
        self._swarm: Swarm = Swarm(cost, lim, n_particles, max_iter=max_iter, inertia=inertia, c_cognitive=c_cognitive, c_social=c_social, stop_at_variance=stop_at_variance, lbx=lbx, ubx=ubx)

    def optimize(self) -> None:
        """
        Perform optimization using the swarm.
        """
        self._swarm.optimize()

    def get_optimal_position(self) -> np.ndarray:
        """
        Get the optimal position found by the swarm.
        
        :return: Numpy array representing the optimal position.
        """
        return self._swarm.get_optimized_position()
    
    def get_optimal_cost(self) -> float:
        """
        Get the optimal cost found by the swarm.
        
        :return: Float representing the optimal cost.
        """
        return self._swarm.get_optimized_cost()