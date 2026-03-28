import numpy as np


def autocorrelation(series, max_lag):
    """
    Compute the autocorrelation of a time series for lags 0 to max_lag.
    """
    n = len(series)
    series = np.array(series, dtype=float)
    series -= np.mean(series)

    lags = np.zeros(max_lag+1)
    var = np.var(series) * n

    if var < 1e-10:
        lags[0] = 1.0 
        return list(lags)
        
    for i in range(max_lag+1):
        lags[i] = np.sum(series[:n - i] * series[i:]) / var

    return list(lags)