import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x = np.array(x)

    if rng is not None and isinstance(rng, np.random.Generator):
        rand_vals = rng.random(x.shape)
    else:
        rand_vals = np.random.random(x.shape)

    scale = 1.0 / (1.0 - p)
    dropout_pattern = np.where(rand_vals > p, scale, 0.0)

    output = x * dropout_pattern

    return output, dropout_pattern