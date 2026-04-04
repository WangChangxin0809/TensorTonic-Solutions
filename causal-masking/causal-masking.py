import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)
    Return: masked scores (same shape, dtype=float)
    """
    scores = np.array(scores)
    rows, cols = np.indices(dimensions=scores.shape)[-2:]
    return np.where(rows >= cols, scores, mask_value)