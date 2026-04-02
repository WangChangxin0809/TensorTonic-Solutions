import numpy as np


def kfold_split(N, k, shuffle=False, rng=None):
    """
    Returns: list of length k with tuples (train_idx, val_idx)
    """
    indices = np.arange(N)

    if shuffle:
        if rng is None:
            np.random.shuffle(indices)
        else:
            rng.shuffle(indices)

    folds = np.array_split(indices, k)

    kfold = []
    for i in range(k):
        val = folds[i]
        train = np.concatenate([fold for j, fold in enumerate(folds) if j != i])
        kfold.append((train, val))
        
    return kfold

