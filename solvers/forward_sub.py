import numpy as np

def forward_substituition(L, b):
    """
    Solves Lx = b, where L is lower triangular with non-zero diagonal entries.
    """
    n = b.shape[0]
    x = np.zeros(n, dtype=np.complex128)

    for j in range(n):
        if np.isclose(L[j, j], 0):
            raise ValueError("Matrix is singular or needs pivoting.")
        x[j] = (b[j] - np.vdot(L[j, :j], x[:j])) / L[j, j]

    return x