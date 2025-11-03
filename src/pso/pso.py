from abc import ABC, abstractmethod
from pso.swarm import Swarm
from pso.cost import CostBase, CostCollection
import numpy as np
from typing import Tuple

class PSOBase(ABC):
    """
    Abstract base class for Particle Swarm Optimization (PSO).
    """
    def __init__(self, lim: Tuple, n_particles: int = 10, max_iter: int = 100, inertia: float = 0.9, c_cognitive: float = 0.5, c_social: float = 0.3, stop_at_variance: float | None = None, lbx: np.ndarray | None = None, ubx: np.ndarray | None = None):
        """
        Initialize the PSO base class with given parameters.
        
        :param lim: Tuple representing the limits for the search space.
        :param n_particles: Number of particles in the swarm.
        :param max_iter: Maximum number of iterations.
        :param inertia: Inertia coefficient.
        :param c_cognitive: Cognitive coefficient.
        :param c_social: Social coefficient.
        """
        self._lim: Tuple = lim
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
    def __init__(self, cost: CostBase | CostCollection, lbx: Tuple, ubx: Tuple, n_particles: int = 10, max_iter: int = 100, inertia: float = 0.9, c_cognitive: float = 0.5, c_social: float = 0.3, stop_at_variance: float | None = None):
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
        assert isinstance(cost, CostBase) or isinstance(cost, CostCollection), f"cost must be an instance of either CostBase or CostCollection. Got type(cost)={type(cost)}."
        super().__init__((lbx, ubx), n_particles, max_iter, inertia, c_cognitive, c_social)
        self._cost_fn: CostCollection = CostCollection(cost) if isinstance(cost, CostBase) else cost
        self._swarm: Swarm = Swarm(self._cost_fn, (lbx, ubx), n_particles, max_iter=max_iter, inertia=inertia, c_cognitive=c_cognitive, c_social=c_social, stop_at_variance=stop_at_variance)

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
    
    @property
    def swarm(self) -> Swarm:
        return self._swarm
    
    @property
    def cost_fn(self) -> CostCollection:
        return self._cost_fn