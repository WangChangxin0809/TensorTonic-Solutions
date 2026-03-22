import numpy as np


def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    pos = np.arange(seq_len).reshape(seq_len, 1)  # (seq_len, 1)
    pos_matrix = np.repeat(pos, d_model, axis=1)  # (seq_len, d_model)
    i = (np.arange(start=0, stop=d_model, step=1) // 2).reshape(1, d_model)  # (1, d_model)
    i_matrix = np.repeat(a=i, repeats=seq_len, axis=0)  # (seq_len, d_model)

    pe = pos_matrix / (np.pow(base, 2*i_matrix / d_model))

    pe[:, 0::2] = np.sin(pe[:, 0::2])
    pe[:, 1::2] = np.cos(pe[:, 1::2])  
    return pe