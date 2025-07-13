import numpy as np

def decompose_cholesky(A, method='banachiewicz'):
    if method == 'banachiewicz' or method == 'row_by_row':
        return decompose_cholesky_banachiewicz(A)
    elif method == 'crout' or method == 'col_by_col':
        return decompose_cholesky_crout(A)
    else:
        raise ValueError("Specified method not supported and/or invalid.")


def decompose_cholesky_banachiewicz(A):
    A = A.copy().astype(np.complex128)
    n = len(A)
    L = np.zeros((n, n), dtype=np.complex128)

    for i in range(n):
        # compute entries before the diagonal
        for j in range(i):
            L[i, j] = (A[i, j] - np.vdot(L[i, :j], L[j, :j])) / L[j, j]
        # compute diagonal value
        L[i, i] = np.sqrt(A[i, i] - np.vdot(L[i, :i], L[i, :i]))
        if L[i, i].real <= 0 or not np.isclose(L[i, i].imag, 0):
            raise ValueError("Matrix is not positive-definite.")
        
    return L


def decompose_cholesky_crout(A):
    A = A.copy().astype(np.complex128)
    n = len(A)
    L = np.zeros((n, n), dtype=np.complex128)

    for j in range(n):
        # compute diagonal value
        L[j, j] = np.sqrt(A[j, j] - np.vdot(L[j, :j], L[j, :j]))
        if L[j, j].real <= 0 or not np.isclose(L[j, j].imag, 0):
            raise ValueError("Matrix is not positive-definite.")
        # compute entries below the diagonal
        for i in range(j+1, n):
            L[i, j] = (A[i, j] - np.vdot(L[i, :j], L[j, :j])) / L[j, j]

    return L