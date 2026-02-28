import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.asarray(y)
    
    if y.size == 0:
        return 0.0
    
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    
    # Remove zero probabilities (numerical safety)
    p = p[p > 0]
    
    return float(-(p * np.log2(p)).sum())