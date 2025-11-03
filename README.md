# PSO - Particle Swarm Optimization

A Python library implementing Particle Swarm Optimization (PSO) algorithm for optimization problems.

## Features

- Abstract base class for PSO implementations
- Concrete PSO implementation with customizable parameters
- Swarm-based optimization with particle management
- Support for custom cost functions
- Configurable optimization parameters (inertia, cognitive and social coefficients)
- Early stopping based on variance threshold

## Installation

### From source

```bash
git clone https://github.com/stiefen1/pso.git
cd pso
pip install .
```

### Development installation

```bash
git clone https://github.com/stiefen1/pso.git
cd pso
pip install -e .[dev]
```

## Quick Start

```python
from pso import PSO
import numpy as np

# Define a simple optimization problem (minimize x^2 + y^2)
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
```

## Parameters

- `cost`: Cost function to minimize
- `lim`: Tuple defining search space limits
- `n_particles`: Number of particles in the swarm (default: 10)
- `max_iter`: Maximum number of iterations (default: 100)
- `inertia`: Inertia coefficient (default: 0.9)
- `c_cognitive`: Cognitive coefficient (default: 0.5)
- `c_social`: Social coefficient (default: 0.3)
- `stop_at_variance`: Optional variance threshold for early stopping
- `lbx`: Lower bound array for each dimension
- `ubx`: Upper bound array for each dimension

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0

## License

This project is licensed under the MIT License - see the LICENSE file for details.
