import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

# Evaluation metrics
nmi = normalized_mutual_info_score  # Normalized Mutual Information
ari = adjusted_rand_score  # Adjusted Rand Index


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy.

    Parameters:
        y_true: Ground truth labels (numpy.array, shape `(n_samples,)`)
        y_pred: Predicted cluster labels (numpy.array, shape `(n_samples,)`)

    Returns:
        Clustering accuracy as a float.
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1  # Determine the number of unique classes
    w = np.zeros((D, D), dtype=np.int64)

    # Build the confusion matrix
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    # Find the optimal label assignment using the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(w.max() - w)

    # Compute the accuracy based on optimal assignment
    return sum([w[i, j] for i, j in zip(row_ind, col_ind)]) * 1.0 / y_pred.size