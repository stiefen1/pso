from abc import ABC, abstractmethod
import numpy as np, matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import Tuple, List, Literal
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
        return 0.0

    def __call__(self, x:float, y:float, *args, **kwargs) -> float:
        return self._weight * self.eval(x, y, *args, **kwargs)

    def __add__(self, new_cost: "CostBase") -> "CostCollection":
        return CostCollection([deepcopy(self), deepcopy(new_cost)])
    
    def plot(
            self,
            lim:Tuple[Tuple, Tuple],
            grid_dim: Tuple[int, int] = (50, 50),
            total:bool=False,
            levels: int = 30,
            log_scale: bool = False,
            colorbar_res:int=5,
            cost_to_use : Tuple | None = None,
            offset : np.ndarray = np.array([0., 0.]),
            ax: Axes | None = None,
            **kwargs
            ) -> Axes:
        
        x0, y0 = lim[0]
        xf, yf = lim[1]

        x = np.linspace(x0, xf, int(grid_dim[0]))
        y = np.linspace(y0, yf, int(grid_dim[1]))
        X, Y = np.meshgrid(x, y)
        Z = np.zeros(X.shape)

        # Evaluate cost for each point in the grid
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                cost = self(xi, yj, total=total, cost_to_use=cost_to_use)
                Z[j, i] = cost

        X = X - offset[0]
        Y = Y - offset[1]

        # Plot cost
        if ax is None:
            _, ax = plt.subplots()

        if log_scale:
            Z = np.log(Z)
        min_z, max_z = np.min(Z), np.max(Z)
        cont = ax.contourf(X, Y, Z, levels=levels, **kwargs)
        self._colorbars.append(plt.colorbar(cont, ticks=np.linspace(min_z, max_z, colorbar_res), ax=ax))
        plt.tight_layout()
        return ax
    
    def plot_verbose(self, ax: Axes, *args, **kwargs) -> Axes:
        return ax
    
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
    
class CostCollection:
    def __init__(self, cost: CostBase | list[CostBase]) -> None:
        if isinstance(cost, CostBase):
            self._collection = [cost]
        elif isinstance(cost, list):
            self._collection = cost
        else:
            raise TypeError(f"Expected CostBase or list of CostBase, got {type(cost)}")

    def plot(
            self,
            lim:Tuple[Tuple, Tuple],
            grid_dim: Tuple[int, int] = (50, 50),
            total:bool=False,
            verbose: bool | List[bool] = False,
            levels: int = 30,
            log_scale: bool = False,
            colorbar_res:int=5,
            display : Literal['horizontal', 'vertical'] = 'horizontal',
            offset : np.ndarray = np.array([0., 0.]),
            axs: List[Axes] | Axes | None = None,
            **kwargs
        ) -> List[Axes]:

        if total:
            return [self.plot_total_cost(lim, grid_dim, levels, log_scale, colorbar_res, offset, axs if isinstance(axs, Axes) else None)]

        # Get number of cost to plot
        n = 1 if total else self.output_size

        # Plot cost
        if isinstance(axs, list) and len(axs) == n:
            pass # Means default axs is valid and we'll use it
        else:
            if display == 'horizontal':
                _, ax = plt.subplots(1, n)
            elif display == 'vertical':
                _, ax = plt.subplots(n, 1)
            else:
                raise ValueError(f"display can be either 'horizontal' or 'vertical' not {display}")
            axs = ax.tolist() if n > 1 else [ax]

        assert isinstance(axs, list) and isinstance(axs[0], Axes), f"axs must be of type List[ Axes ] but is {type(axs)}."

        for k, cost in enumerate(self.collection):
            cost.plot(lim, grid_dim=grid_dim, total=True, levels=levels, log_scale=log_scale, colorbar_res=colorbar_res, offset=offset, ax=axs[k])   

        return axs
    
    def plot_total_cost(
            self,
            lim:Tuple[Tuple, Tuple],
            grid_dim: Tuple[int, int] = (50, 50),
            levels: int = 30,
            log_scale: bool = False,
            colorbar_res:int=5,
            offset : np.ndarray = np.array([0., 0.]),
            ax: Axes | None = None,
            **kwargs
        ) -> Axes:
            x0, y0 = lim[0]
            xf, yf = lim[1]

            x = np.linspace(x0, xf, int(grid_dim[0]))
            y = np.linspace(y0, yf, int(grid_dim[1]))
            X, Y = np.meshgrid(x, y)
            Z = np.zeros(X.shape)

            # Evaluate cost for each point in the grid
            for i, xi in enumerate(x):
                for j, yj in enumerate(y):
                    cost = sum(self(xi, yj))
                    Z[j, i] = cost

            X = X - offset[0]
            Y = Y - offset[1]

            # Plot cost
            if ax is None:
                _, ax = plt.subplots()
            if log_scale:
                Z = np.log(Z)

            min_z, max_z = np.min(Z), np.max(Z)
            cont = ax.contourf(X, Y, Z, levels=levels, **kwargs)
            plt.colorbar(cont, ticks=np.linspace(min_z, max_z, colorbar_res), ax=ax)
            plt.tight_layout()
            return ax

    
    def eval(self, x:float, y:float, *args, **kwargs) -> List[ float ]:
        return [cost(x, y, *args, **kwargs) for cost in self._collection]

    def __call__(self, x: float, y: float, *args, **kwargs) -> List[float]:
        return self.eval(x, y, *args, **kwargs)
    
    def __getitem__(self, key:int):
        item = deepcopy(self._collection[key])
        return item
    
    def __len__(self) -> int:
        return len(self._collection)
    
    def __add__(self, new_cost: CostBase | CostCollection) -> "CostCollection":
        self_copy = deepcopy(self)
        new_cost_copy = deepcopy(new_cost)
        if isinstance(new_cost_copy, CostCollection):
            return CostCollection(self_copy.collection + new_cost_copy.collection)
        elif isinstance(new_cost, CostBase):
            new_collection = []
            for cost in self_copy:
                new_collection.append(cost)
            new_collection.append(new_cost_copy)
            return CostCollection(new_collection)
        else:
            raise TypeError(f"cannot add CostCollection object with object of type {type(new_cost)}.")

    @property
    def collection(self) -> list[CostBase]:
        return self._collection
    
    @property
    def output_size(self) -> int:
        return len(self._collection)
    
    def __iter__(self):
        return iter(self._collection)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(costs={self._collection})"

def main() -> None:
    pass

if __name__=="__main__":
    main()