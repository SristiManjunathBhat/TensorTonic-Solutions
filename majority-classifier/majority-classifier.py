import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # Find unique labels and their counts
    values, counts = np.unique(y_train, return_counts=True)
    
    # Find index of most frequent label
    majority_label = values[np.argmax(counts)]
    
    # Predict that label for all test samples
    return np.full(len(X_test), majority_label)