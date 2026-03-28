import numpy as np

def impute_missing(X, strategy='mean'):
    X = np.array(X, dtype=float)
    func = np.nanmean if strategy == 'mean' else np.nanmedian

    if X.ndim == 1:
        mask = np.isnan(X)
        X[mask] = 0 if np.all(mask) else func(X)
        return X

    for col in range(X.shape[1]):
        c = X[:, col]
        if np.all(np.isnan(c)):
            c[:] = 0
        else:
            c[np.isnan(c)] = func(c)

    return X