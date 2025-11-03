from pso import PSO, CostBase
import numpy as np, matplotlib.pyplot as plt

# Toy example cost function with two local minima
class QuadOpt(CostBase):
    def eval(self, x:float, y:float, *args, **kwargs) -> float:
        return np.min([(x+3)**2 + (y+3)**2, (x-3)**2 + (y-3)**2])
    
class NLOpt(CostBase):
    def eval(self, x:float, y:float, *args, **kwargs) -> float:
        return -np.min([x, y]) * np.sin(x)*np.cos(y)

# Create PSO optimizer
optimizer = PSO(
    cost=QuadOpt() + 3 * NLOpt(),
    lbx=(-10, -10),
    ubx=(10, 10),# Search space limits
    n_particles=50,
    max_iter=300,
    inertia=0.6,# 1.0,
    c_cognitive=0.1,#0.55,
    c_social=0.3,# 0.325,
    stop_at_variance=1e-6
)

# Particles before optimization
optimizer.swarm.plot_cost_and_particles(grid_dim=(300, 300))
plt.show()

# Run optimization
optimizer.optimize()

# Get results
best_position = optimizer.get_optimal_position()
best_cost = optimizer.get_optimal_cost()

print(f"Results after {optimizer._swarm._iter} iterations: ")
print(f"Best position: {best_position}")
print(f"Best cost: {best_cost}")

optimizer.swarm.plot_cost_and_particles(grid_dim=(300, 300), total=False, display='vertical')
plt.show()