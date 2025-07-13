import numpy as np

def diagonal_solve(D, b):
    """
    Solves Dx = b, where D is a diagonal matrix with non-zero diagonal entries.
    """
    d = np.diag(D)
    if np.any(np.isclose(d, 0)):
        raise ValueError("Matrix has zero diagonal entries.")
    x = b / d
    return x