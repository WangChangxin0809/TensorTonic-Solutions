import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x, p = np.array([x, p])
    if np.sum(p) != 1 :
        raise ValueError
    return np.sum(x * p) 
