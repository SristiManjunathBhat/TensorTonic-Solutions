import numpy as np

def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    
    d = X.shape[1]              # number of features
    I = np.eye(d)               # identity matrix
    
    w = np.linalg.inv(X.T @ X + lam * I) @ X.T @ y
    
    return w.tolist()