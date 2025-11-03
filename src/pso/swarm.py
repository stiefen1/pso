from utils import *
from pso.cost import CostBase, CostCollection
from pso.particle import Particle
import numpy as np
from typing import Tuple, List, Literal
from matplotlib.axes import Axes

class Swarm:
    def __init__(self,
                 cost: CostCollection,
                 lim: Tuple,
                 n_particles: int = 10,
                 initial_gbest_position: np.ndarray | None = None,
                 max_iter: int = 100,
                 inertia: float = 0.9,
                 c_cognitive: float = 0.5,
                 c_social: float = 0.3,
                 stop_at_variance: float | None = None
                ):
        """
        Initialize a Swarm of n_particles particles within lim that should be optimized according to a cost.

        - cost: CostBase
            Cost function associated to the swarm. It can return a float or a list of floats. If a list is returned,
            the total cost will be computed as sum(cost_list). This is useful for visualization purposes, as this
            class provides a method to plot each contribution of the cost in the parameter space.
        - lim: Tuple[Tuple, Tuple]
            Tuple of tuples, each containing the lower and upper bounds of the search space.
            Example: lim = ((d1_min, d2_min, d3_min), (d1_max, d2_max, d3_max)) for a 3D search space.
        - n_particles: int
            Number of particles within the swarm.
        - initial_gbest_position: np.ndarray
            Initial guess for the value to optimize.
        """
        assert isinstance(cost, CostCollection), "cost must be a CostCollection object"
        is_lim_valid(lim)
        self._cost_fn: CostCollection = cost
        self._lim: Tuple = lim
        self._dim: int = len(lim[0])
        self._n_particles: int = n_particles
        self._particles: list[Particle] = [Particle(lim) for _ in range(n_particles)]
        self._gbest_position: np.ndarray = get_sample_within_lim(lim) if initial_gbest_position is None else initial_gbest_position.copy()
        self._gbest_cost: float = sum(self.cost_fn(*self._gbest_position))
        self._max_iter: int = max_iter
        self._inertia: float = inertia
        self._c_cognitive: float = c_cognitive
        self._c_social: float = c_social
        self._stop_at_variance_vector: float | np.ndarray = stop_at_variance*np.ones_like(self._gbest_position) if stop_at_variance is not None else 0.0
        self._iter = 0
        # self._lbx = np.array(lim[0])
        # self._ubx = np.array(lim[1])

    def optimize(self) -> None:
        """
        Runs the main optimization loop of the swarm.

        - max_iter: int
            Max number of iterations of the optimization loop.
        - inertia: float
            Inertia parameter used to update the velocity of each particle.
        - c_cognitive: float
            Cognitive weight used to update the velocity of each particle.
        - c_social: float
            Social weight used to update the velocity of each particle.
        """
        self._iter = 0
        for _ in range(self._max_iter):
            self.optimization_step(self._inertia, self._c_cognitive, self._c_social)

    def optimization_step(self, inertia: float, c_cognitive: float, c_social: float, **kwargs) -> None:
        """
        Runs one single optimization step of the swarm.

        - inertia: float
            Inertia parameter used to update the velocity of each particle.
        - c_cognitive: float
            Cognitive weight used to update the velocity of each particle.
        - c_social: float
            Social weight used to update the velocity of each particle.
        """
        if self._stop_at_variance_vector is not None and np.any(np.greater_equal(self.get_variance(), self._stop_at_variance_vector)):
            self._iter += 1
            for particle in self._particles:
                self.single_particle_optimization_step(particle, inertia, c_cognitive, c_social)

    def single_particle_optimization_step(self, particle: Particle, inertia: float, c_cognitive: float, c_social: float) -> None:
        """
        Runs one single optimization step of one particle and updates the global best value.

        - particle: Particle
            Particle to optimize.
        - inertia: float
            Inertia parameter used to update the velocity of each particle.
        - c_cognitive: float
            Cognitive weight used to update the velocity of each particle.
        - c_social: float
            Social weight used to update the velocity of each particle.
        """
        particle.update(inertia, c_cognitive, c_social, self._gbest_position) # Update the particle's velocity and position
        particle._position = np.clip(particle._position, self._lim[0], self._lim[1])
        particle.cost = sum(self.cost_fn(*particle.position)) # Compute the cost of the particle
        particle.update_pbest()
        self.update_gbest(particle)

    def update_gbest(self, particle: Particle) -> None:
        """
        Check if the provided cost is better than the current global best, and if yes, update the latter.

        - particle: Particle
            Particle to check and potentially update the global best.
        """
        if particle.cost < self._gbest_cost:
            self._gbest_cost = particle.cost
            self._gbest_position = particle.position

    def get_optimized_position(self) -> np.ndarray:
        """
        Get the optimized global best position.
        """
        return self.gbest_position

    def get_optimized_cost(self) -> float:
        """
        Get the optimized global best cost.
        """
        return self.gbest_cost

    def get_optimized_particle_positions(self) -> list:
        """
        Get the positions of all particles.
        """
        return [particle.position for particle in self.particles]

    def get_optimized_particle_pbest_positions(self) -> list:
        """
        Get the personal best positions of all particles.
        """
        return [particle.pbest_position for particle in self.particles]

    def get_optimized_particle_pbest_costs(self) -> list:
        """
        Get the personal best costs of all particles.
        """
        return [particle.pbest_cost for particle in self.particles]

    def get_optimized_particle_velocities(self) -> list:
        """
        Get the velocities of all particles.
        """
        return [particle.velocity for particle in self.particles]

    def plot_vec_cost(self, grid_dim: Tuple = (20, 20), levels: int = 30, log_scale: bool = False, particles : bool = True, center : bool = False, **kwargs):
        """
        See plot_cost_and_particles.
        """
        return self.plot_cost_and_particles(grid_dim=grid_dim, total=False, levels=levels, log_scale=log_scale, particles=particles, center=center, **kwargs)

    def plot_total_cost(self, grid_dim: Tuple = (20, 20), levels: int = 30, log_scale: bool = False, particles: bool = True, center : bool = False, **kwargs):
        """
        See plot_cost_and_particles.
        """
        return self.plot_cost_and_particles(grid_dim=grid_dim, total=True, levels=levels, log_scale=log_scale, particles=particles, center=center, **kwargs)

    def plot_cost_and_particles(self, grid_dim: Tuple = (20, 20), total: bool = True, levels: int = 30, log_scale: bool = False, particles: bool = True, verbose: bool | list[bool] = True, center : bool = False, lim : Tuple | None = None, best: bool = True, axs: List[Axes] | None = None, display: Literal['vertical', 'horizontal'] = 'horizontal', **kwargs):
        """
        Plot the total cost function for the selected dimensions.

        - dim: Tuple[int, int]
            2D Tuple to select which dimensions of the parameter space to plot.
        - mesh_res: Tuple[float, float]
            Resolution of the mesh in the two selected dimensions.
        - levels: int
            Number of level curves.
        - log_scale: bool
            Whether to use a logarithmic scale for the cost values.
        """
        lim = lim or self._lim
        center_of_plot = np.mean(lim, 0) if center else np.array([0., 0.])
        axs = self.cost_fn.plot(lim, grid_dim=grid_dim, total=total, levels=levels, log_scale=log_scale, verbose=verbose, offset=center_of_plot, axs=axs, display=display, **kwargs)

        if particles:
            self.plot_particles(axs, lim, center_of_plot=center_of_plot)
        if best:
            self.plot_best(axs, lim, center_of_plot=center_of_plot, total=total)
        return axs
    
    def plot_particles(self, axs: List [Axes], lim: Tuple[ Tuple, Tuple ], center_of_plot: np.ndarray = np.array([0., 0.]), c: str = 'red', s: int = 10) -> List[Axes]:
        x0, y0 = lim[0]
        xf, yf = lim[1]
        for k in range(len(axs)):
            for particle in self._particles:
                particle_position = np.array(particle.position) - center_of_plot
                axs[k].scatter(*particle_position, c=c, s=s)

            axs[k].set_xlim(x0, xf)
            axs[k].set_ylim(y0, yf)
        return axs
    
    def plot_best(self, axs: List[Axes], lim, center_of_plot=np.array([0., 0.]), total: bool = True) -> List[Axes]:
        x0, y0 = lim[0]
        xf, yf = lim[1]
        gbest_position = np.array(self._gbest_position) - center_of_plot
        cost_coll = self.cost_fn(*self._gbest_position)
        for k in range(len(axs)):
            axs[k].scatter(*gbest_position, c='red', marker='x')
            cost = sum(cost_coll) if total else cost_coll[k] 
            axs[k].text(gbest_position[0], gbest_position[1], f"{cost:.3f}", c='red')
            axs[k].set_xlim(x0, xf)
            axs[k].set_ylim(y0, yf)
        return axs

    @property
    def gbest_position(self) -> np.ndarray:
        """
        Get the global best position.
        """
        return self._gbest_position.copy()

    @gbest_position.setter
    def gbest_position(self, value: np.ndarray) -> None:
        """
        Set the global best position.
        """
        self._gbest_position = value

    @property
    def gbest_cost(self) -> float:
        """
        Get the global best cost.
        """
        return self._gbest_cost

    @gbest_cost.setter
    def gbest_cost(self, value: float) -> None:
        """
        Set the global best cost.
        """
        self._gbest_cost = value

    @property
    def particles(self) -> list[Particle]:
        """
        Get the list of particles.
        """
        return self._particles

    @property
    def cost_fn(self) -> CostCollection:
        """
        Get the cost function.
        """
        return self._cost_fn

    @cost_fn.setter
    def cost_fn(self, value: CostCollection):
        """
        Set the cost function.
        """
        assert isinstance(value, CostCollection)
        self._cost_fn = value

    @property
    def dim(self) -> int:
        """
        Get the dimension of the search space.
        """
        return self._dim
    
    @property
    def n_iter(self) -> int:
        """
        Get the current number of optimization iterations
        """
        return self._iter
    
    def get_variance(self) -> np.ndarray:
        sum_of_square = np.zeros_like(self.particles[0].position)
        for p in self.particles:
            sum_of_square += np.square(p.position-self.gbest_position)
        return sum_of_square / self._n_particles
    
    def get_standard_deviation(self) -> np.ndarray:
        return np.sqrt(self.get_variance())