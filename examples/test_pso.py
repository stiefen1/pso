from pso import PSO, CostBase
import numpy as np

# Define a simple optimization problem (minimize x^2 + y^2)
class QuadOpt(CostBase):
    pass
def cost_function(position):
    return np.sum(position**2)

# Create PSO optimizer
optimizer = PSO(
    cost=cost_function,
    lim=(-10, 10),  # Search space limits
    n_particles=30,
    max_iter=100,
    inertia=0.9,
    c_cognitive=0.5,
    c_social=0.3
)

# Run optimization
optimizer.optimize()

# Get results
best_position = optimizer.get_optimal_position()
best_cost = optimizer.get_optimal_cost()

print(f"Best position: {best_position}")
print(f"Best cost: {best_cost}")