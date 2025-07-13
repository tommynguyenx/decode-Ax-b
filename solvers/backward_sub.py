import numpy as np

def backward_substitution(U, b):
    """
    Solves Ux = b, where U is upper triangular with non-zero diagonal entries.
    """
    n = b.shape[0]
    x = np.zeros(n, dtype=np.complex128)

    for j in range(n-1, -1 , -1):
        if np.isclose(U[j, j], 0):
            raise ValueError("Matrix is singular or needs pivoting.")
        x[j] = (b[j] - np.vdot(U[j, j+1:n], x[j+1:n])) / U[j, j]

    return x