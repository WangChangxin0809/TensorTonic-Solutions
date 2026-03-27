import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    x = np.array(x)
    gap = []
    if x.ndim == 3:
        for channel in x:
            gap.append(np.mean(channel))
    elif x.ndim == 4:
        for n in x:
            tmp = []
            for channel in n:
                tmp.append(np.mean(channel))
            gap.append(tmp)
    else:
        raise ValueError
    return gap

    pass