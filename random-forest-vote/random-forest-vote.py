import numpy as np


def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    predictions = np.array(predictions)
    predictions = predictions.swapaxes(0, 1)

    ans = []

    for pred in predictions:
        values, counts = np.unique(pred, return_counts=True)
        idx = np.lexsort((values, -counts))
        ans.append(values[idx[0]])

    return ans