import numpy as np


def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.array(y)
    values, counts = np.unique(y, return_counts=True)
    counts = counts / len(y)

    return -np.sum(counts * np.log2(counts))

    pass