import math

import numpy as np


def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    """
    Randomly shuffle a dataset and yield mini-batches (X_batch, y_batch).
    """
    X_np, y_np = np.array(X), np.array(y)
    n = len(y)

    indices = np.arange(start=0, stop=n, step=1)
    if rng is not None:
        rng.shuffle(indices)
    else:
        np.random.shuffle(indices)

    X_shuffled, y_shuffled = X_np[indices], y_np[indices]


    for i in range(math.ceil(n / batch_size) if drop_last == False else math.floor(n / batch_size)):
        batch_X = X_shuffled[i*batch_size : (i + 1) * batch_size]
        batch_y = y_shuffled[i*batch_size : (i + 1) * batch_size]
        yield batch_X, batch_y

