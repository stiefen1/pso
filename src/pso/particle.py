from utils import *
import numpy as np

class Particle:
    def __init__(self, lim: tuple):
        """
        Initialize a Particle instance.

        Parameters:
        lim (tuple): Tuple of tuples containing the lower and upper bounds of the search space.
                     Example: lim = ((d1_min, d2_min, d3_min), (d1_max, d2_max, d3_max)) for a 3D search space.
        """
        is_lim_valid(lim)
        self._lim = lim
        self._n_dimensions = len(lim[0])
        self._position = self.get_initial_position_within_lim()
        self._velocity = self.get_initial_velocity()
        self._pbest_position = self._position.copy()
        self._pbest_cost = float('inf')
        self._cost = float('inf')

    def get_initial_position_within_lim(self) -> np.ndarray:
        """Generate an initial position within the given limits."""
        return get_sample_within_lim(self._lim)

    def get_initial_velocity(self, min_vel: float = 0.0, max_vel: float = 0.0) -> np.ndarray:
        """Generate an initial velocity within the given range."""
        return (max_vel - min_vel) * np.random.rand(self._n_dimensions) + min_vel
    
    def update_position(self) -> None:
        """Update the particle's position based on its velocity."""
        self._position += self._velocity

    def update_velocity(self, inertia: float, c_cognitive: float, c_social: float, gbest_position: np.ndarray) -> None:
        """
        Update the particle's velocity based on inertia, cognitive, and social components.

        Parameters:
        inertia (float): Inertia weight.
        c_cognitive (float): Cognitive coefficient.
        c_social (float): Social coefficient.
        gbest_position (np.ndarray): Global best position.
        """
        cognitive_component = c_cognitive * np.random.rand(self._n_dimensions) * (self._pbest_position - self._position)
        social_component = c_social * np.random.rand(self._n_dimensions) * (gbest_position - self._position)
        self._velocity = inertia * self._velocity + cognitive_component + social_component

    def update_pbest(self) -> None:
        """Update the particle's personal best position and cost."""
        if self._cost < self._pbest_cost:
            self._pbest_cost = self._cost
            self._pbest_position = self._position.copy()

    def update(self, inertia: float, c_cognitive: float, c_social: float, gbest_position: np.ndarray) -> None:
        """
        Update the particle's velocity, position, and personal best.

        Parameters:
        cost_fn (Callable): Cost function to evaluate the particle's position.
        inertia (float): Inertia weight.
        c_cognitive (float): Cognitive coefficient.
        c_social (float): Social coefficient.
        gbest_position (np.ndarray): Global best position.
        """
        self.update_velocity(inertia, c_cognitive, c_social, gbest_position)
        self.update_position()

    @property
    def position(self) -> np.ndarray:
        """Get a copy of the particle's position."""
        return self._position.copy()
    
    @property
    def velocity(self) -> np.ndarray:
        """Get a copy of the particle's velocity."""
        return self._velocity.copy()
    
    @property
    def pbest_position(self) -> np.ndarray:
        """Get a copy of the particle's personal best position."""
        return self._pbest_position.copy()
    
    @pbest_position.setter
    def pbest_position(self, value: np.ndarray) -> None:
        """Set the particle's personal best position."""
        self._pbest_position = value

    @property
    def pbest_cost(self) -> float:
        """Get the particle's personal best cost."""
        return self._pbest_cost
    
    @pbest_cost.setter
    def pbest_cost(self, value: float) -> None:
        """Set the particle's personal best cost."""
        self._pbest_cost = value

    @property
    def cost(self) -> float:
        """Get the particle's current cost."""
        return self._cost
    
    @cost.setter
    def cost(self, value: float) -> None:
        """Set the particle's current cost."""
        self._cost = value