import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    w, g, s = np.array((w, g, s))
    s = beta * s + (1 - beta) * g * g
    w = w - lr * g / np.pow(s + eps, 0.5)
    return w, s
    
    pass