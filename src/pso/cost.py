from abc import ABC, abstractmethod
import numpy as np, matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Tuple, List
from copy import deepcopy


"""
The general idea is to instantiate a Cost object with necessary information (those that allow to map (x, y) -> z)
and then call it with the position (x, y) to get the cost at that position. The cost object can be a single cost
or a collection of costs. The collection of costs can be added together to get a new collection of costs. The cost
object can be added to another cost object to get a collection of costs. 

cost = CostObstacleA(obstacles)
cost += CostEnergy(initial_path)
cost += Costpath(initial_path)
pso = PSOWaypointsOpt(path, obs, lim, **params)

"""
class CostBase(ABC):
    def __init__(self, weight:float=1.):
        self._weight = weight
        self._colorbars = []

    @abstractmethod
    def eval(self, x:float, y:float, *args, **kwargs) -> float:
        pass

    def __call__(self, x:float, y:float, *args, **kwargs):
        return [self._weight * self.eval(x, y, *args, **kwargs)]

    def __add__(self, new_cost: "CostBase") -> "CostCollection":
        return CostCollection([deepcopy(self), deepcopy(new_cost)])
    
    def plot(self, lim:Tuple[Tuple, Tuple], grid_dim: Tuple[int, int] = (50, 50), total:bool=False, levels: int = 30, log_scale: bool = False, colorbar_res:int=5, cost_to_use : Tuple | None = None, display : str = 'horizontal', offset : np.ndarray = np.array([0., 0.]), axs=None, **kwargs) -> List[Axes]:
        x0, y0 = lim[0]
        xf, yf = lim[1]

        # Get number of cost to plot
        n = 1 if total else self.output_size if cost_to_use is None else len(cost_to_use)

        x = np.linspace(x0, xf, int(grid_dim[0]))
        y = np.linspace(y0, yf, int(grid_dim[1]))
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((n, *X.shape))

        # Evaluate cost for each point in the grid
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                cost = self(xi, yj, total=total, cost_to_use=cost_to_use)
                for k, cost_k in enumerate(cost):
                    Z[k, j, i] = cost_k

        X = X - offset[0]
        Y = Y - offset[1]

        # Plot cost
        if axs is not None and len(axs) == n:
            pass # Means default axs is valid and we'll use it
        else:
            if display == 'horizontal':
                _, axs = plt.subplots(1, n)
            elif display == 'vertical':
                _, axs = plt.subplots(n, 1)
            else:
                raise ValueError(f"display can be either 'horizontal' or 'vertical' not {display}")
            axs = axs if n > 1 else [axs]
        
        try:
            self.clear_colorbars()
        except:
            pass
        for k in range(n):
            if log_scale:
                Z[k] = np.log(Z[k])
            min_z, max_z = np.min(Z[k]), np.max(Z[k])
            cont = axs[k].contourf(X, Y, Z[k], levels=levels, **kwargs)
            self._colorbars.append(plt.colorbar(cont, ticks=np.linspace(min_z, max_z, colorbar_res), ax=axs[k]))
        if total:
            axs[0].set_title("Total Cost")
        plt.tight_layout()
        return axs
    
    def plot_verbose(self, ax, offset: np.ndarray = np.array([0, 0]), **kwargs):
        return ax
    
    def clear_colorbars(self) -> None:
        for colorbar in self._colorbars:
            colorbar.remove()
        self._colorbars = []

    @property
    def weight(self) -> float:
        return self._weight
    
    @property
    def output_size(self) -> int:
        return 1
    
    @weight.setter
    def weight(self, value:float) -> None:
        assert value >= 0., f"Weight must be positive, got {value}"
        self._weight = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(weight={self._weight})"
    
    def __mul__(self, value:float):
        self_copy = deepcopy(self)
        self_copy.weight *= value
        return self_copy
    
    def __rmul__(self, value:float):
        return self.__mul__(value)
    
class CostCollection(CostBase):
    def __init__(self, cost: CostBase | list[CostBase], weight: float=1.) -> None:
        super().__init__(weight)
        if isinstance(cost, CostBase):
            self._collection = [cost]
        elif isinstance(cost, list):
            self._collection = cost
        else:
            raise TypeError(f"Expected CostBase or list of CostBase, got {type(cost)}")

    def plot(self, *args, verbose: bool|list[bool]=False, cost_to_use: Tuple | None = None, offset:np.ndarray=np.array([0., 0.]), total: bool=False, **kwargs) -> List[Axes]:
        axs = super().plot(*args, cost_to_use=cost_to_use, offset=offset, total=total, **kwargs)
        # n_cost = len(self.collection) if cost_to_use is None else len(cost_to_use)
        if cost_to_use is None:
            cost_to_use = tuple(i for i in range(len(self.collection))) 

        collection_to_use:list[CostBase] = []
        for idx in cost_to_use:
            collection_to_use.append(self.collection[idx])     
        n_cost = len(collection_to_use)
        n_axs = len(axs)
        if verbose or isinstance(verbose, list): # True if verbose is True
            if isinstance(verbose, list):
                assert len(verbose) == n_cost
            else:
                verbose = n_cost * [verbose]
            for i, cost in enumerate(collection_to_use):
                idx = i if n_axs==n_cost else 0
                if verbose[i]:
                    cost.plot_verbose(ax=axs[idx], offset=offset, total=total)
        return axs

    def eval(self, x:float, y:float, *args, **kwargs) -> List[ float ]:
        return [self._weight * cost(x, y, *args, **kwargs)[0] for cost in self._collection]

    def __call__(self, x: float, y: float, total: bool = False, cost_to_use: Tuple | None = None, *args, **kwargs):
        if cost_to_use is None:
            cost_to_use = tuple(i for i in range(self.output_size))
        val = self.eval(x, y, *args, **kwargs)
        val_to_use = []
        for cost_to_use_i in cost_to_use:
            val_to_use.append(val[cost_to_use_i])
        return val_to_use if not(total) else [sum(val_to_use)]
    
    def __getitem__(self, key:int):
        item = deepcopy(self._collection[key])
        item *= self._weight
        return item
    
    def __len__(self) -> int:
        return len(self._collection)
    
    def __add__(self, new_cost: CostBase) -> "CostCollection":
        self_copy = deepcopy(self)
        new_cost_copy = deepcopy(new_cost)
        if isinstance(new_cost, CostCollection):
            for cost in self_copy:
                cost.weight *= self_copy._weight
            for cost in new_cost_copy:
                cost.weight *= new_cost_copy.weight
            return CostCollection(self_copy.collection + new_cost_copy.collection)

        elif isinstance(new_cost, CostBase):
            new_collection = []
            for cost in self_copy:
                cost.weight *= self_copy._weight
                new_collection.append(cost)
            new_collection.append(new_cost_copy)
            return CostCollection(new_collection)

    @property
    def collection(self) -> list["CostCollection"]:
        return self._collection
    
    @property
    def output_size(self) -> int:
        return len(self._collection)
    
    def __iter__(self):
        return iter(self._collection)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(costs={self._collection}, weight={self._weight})"

def main() -> None:
    pass

if __name__=="__main__":
    main()