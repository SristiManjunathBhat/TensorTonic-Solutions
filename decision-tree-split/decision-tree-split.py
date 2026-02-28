import numpy as np

def _gini(y):
    if len(y) == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1.0 - np.sum(p ** 2)


def decision_tree_split(X, y):
    X = np.array(X, dtype=float)
    y = np.array(y)
    
    n_samples, n_features = X.shape
    best_gini = float("inf")
    best_feature = None
    best_threshold = None
    
    for j in range(n_features):
        # Possible thresholds = midpoints between sorted unique values
        values = np.sort(np.unique(X[:, j]))
        thresholds = (values[:-1] + values[1:]) / 2
        
        for t in thresholds:
            left_mask = X[:, j] <= t
            right_mask = X[:, j] > t
            
            y_left = y[left_mask]
            y_right = y[right_mask]
            
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            
            gini_left = _gini(y_left)
            gini_right = _gini(y_right)
            
            weighted_gini = (
                (len(y_left) / n_samples) * gini_left +
                (len(y_right) / n_samples) * gini_right
            )
            
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = j
                best_threshold = t
    
    return [best_feature, float(best_threshold)]