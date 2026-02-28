import numpy as np

def _entropy(y):
    """
    Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0

    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]  # avoid log(0)

    return float(-(p * np.log2(p)).sum())


def information_gain(y, split_mask):
    """
    Compute information gain of a binary split.
    """
    y = np.asarray(y)
    split_mask = np.asarray(split_mask)

    # Split labels
    left = y[split_mask]
    right = y[~split_mask]

    # If one side empty → no information
    if left.size == 0 or right.size == 0:
        return 0.0

    # Parent entropy
    H_parent = _entropy(y)

    # Weighted child entropy
    n = len(y)
    H_children = (
        (len(left)/n) * _entropy(left) +
        (len(right)/n) * _entropy(right)
    )

    return H_parent - H_children