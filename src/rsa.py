from typing import Literal

import numpy as np

from scipy.stats import kendalltau

def rsa(X: np.ndarray, Y: np.ndarray):

    X_X = X @ X.T
    Y_Y = Y @ Y.T

    J_X = np.ones_like(X_X)
    J_Y = np.ones_like(Y_Y)

    A = J_X - X_X
    B = J_Y - Y_Y

    return kendalltau(A, B), A, B



#Hilbert-Schmidt Independence Criterion

def cka(X: np.ndarray, Y: np.ndarray, kernel: Literal["inner_product"] = "inner_product"):

    if kernel == "inner_product":
        K = X @ X.T
        L = Y @ Y.T
    else:
        raise NotImplementedError

    return K.dot(L) / np.sqrt(np.dot(K, K) * np.dot(L, L))

if __name__ == "__main__":
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    Y = np.array([[5, 6, 5], [7, 8, 6], [9, 10, 11], [12, 13, 14]])

    # print(rsa(X, Y))
    print(cka(X, Y, kernel="inner_product"))