import numpy as np
from scipy.stats import zscore, spearmanr
from scipy.spatial.distance import pdist, squareform


def rsa_v2(X: np.ndarray, Y: np.ndarray) -> (np.ndarray, float):
    """
    X: np.ndarray of shape (n_samples, n_features)
    """
    X_norm = zscore(X, axis=0, ddof=1)
    condensed = pdist(X_norm, metric="cosine")
    X_rdm = squareform(condensed)

    Y_norm = zscore(Y, axis=0, ddof=1)
    condensed = pdist(Y_norm, metric="cosine")
    Y_rdm = squareform(condensed)

    idx = np.triu_indices_from(X_rdm, k=1)
    rsa_score, pval = spearmanr(X_rdm[idx], Y_rdm[idx])

    return rsa_score, pval
