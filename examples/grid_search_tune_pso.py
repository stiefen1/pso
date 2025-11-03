import numpy as np
from pso import CostBase, PSO

def cost_function(x, y):
    return np.min([(x+3)**2 + (y+3)**2, (x-3)**2 + (y-3)**2]) - np.min([x, y]) * np.sin(x)*np.cos(y)

class QuadOpt(CostBase):
    def eval(self, x:float, y:float, *args, **kwargs) -> float:
        return cost_function(x, y)

COGNITIVE = np.linspace(0.1, 1, 5)
SOCIAL = np.linspace(0.1, 1, 5)
INERTIA = np.linspace(0.1, 1, 5)
n_opt_per_params = 10

results = np.ndarray((0, 5))

for c in COGNITIVE:
    for s in SOCIAL:
        for i in INERTIA:
            cost_k = 0
            iter_k = 0
            for k in range(n_opt_per_params):
                # Create PSO optimizer
                optimizer = PSO(
                    cost=QuadOpt(),
                    lbx=(-10, -10),
                    ubx=(10, 10),  # Search space limits
                    n_particles=50,
                    max_iter=100,
                    inertia=i,
                    c_cognitive=c,
                    c_social=s,
                    stop_at_variance=1e-6
                )
                optimizer.optimize()
                cost_k += optimizer.get_optimal_cost()
                iter_k += optimizer.swarm.n_iter

            print(f"Cumulated cost for c={c:.1f}, s={s:.1f}, i={i:.1f}, iter={iter_k} : {cost_k:.3f}")
            results = np.append(results, np.array([[cost_k/n_opt_per_params, c, s, i, iter_k]]), axis=0)
print(f"Best set of parameters:\n {results[np.argsort(results[:, 0]), :]}")

