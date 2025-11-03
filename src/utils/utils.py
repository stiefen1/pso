import numpy as np

def is_lim_valid(lim: tuple) -> bool:
    """
    Validate the limit tuple.

    Parameters:
    lim (tuple): A tuple containing two lists or arrays (lower_bounds, upper_bounds).

    Returns:
    bool: True if the limit is valid, raises an AssertionError otherwise.
    """
    assert isinstance(lim, tuple), "lim must be a tuple"
    assert len(lim) == 2, "lim must contain 2 elements: (lower_bounds, upper_bounds)"
    assert len(lim[0]) == len(lim[1]), "lower and upper bounds must have the same number of dimensions"
    return True

def get_sample_within_lim(lim: tuple) -> np.ndarray:
    """
    Generate a random sample within the given limits.

    Parameters:
    lim (tuple): A tuple containing two lists or arrays (lower_bounds, upper_bounds).

    Returns:
    np.ndarray: A random sample within the specified limits.
    """
    is_lim_valid(lim)
    return np.random.uniform(lim[0], lim[1])