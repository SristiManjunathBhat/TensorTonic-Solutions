import numpy as np

def gaussian_naive_bayes(X_train, y_train, X_test):
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train)
    X_test = np.array(X_test, dtype=float)
    
    classes = np.unique(y_train)
    n_samples, n_features = X_train.shape
    
    means = {}
    variances = {}
    priors = {}
    
    # Compute statistics for each class
    for c in classes:
        X_c = X_train[y_train == c]
        means[c] = X_c.mean(axis=0)
        variances[c] = X_c.var(axis=0) + 1e-9  # small value to avoid divide by zero
        priors[c] = len(X_c) / n_samples
    
    predictions = []
    
    # Predict each test sample
    for x in X_test:
        class_scores = {}
        
        for c in classes:
            mean = means[c]
            var = variances[c]
            
            # Log Gaussian likelihood
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * var) +
                ((x - mean) ** 2) / var
            )
            
            log_prior = np.log(priors[c])
            
            class_scores[c] = log_prior + log_likelihood
        
        # Choose class with highest score
        predictions.append(max(class_scores, key=class_scores.get))
    
    return predictions