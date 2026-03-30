import numpy as np


def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    x_t, h_prev, Wx, Wh, b = np.array(x_t), np.array(h_prev), np.array(Wx), np.array(Wh), np.array(b)
    
    h_cur = np.tanh(np.dot(x_t, Wx) + np.dot(h_prev, Wh) + b)
    
    return h_cur
    
    
