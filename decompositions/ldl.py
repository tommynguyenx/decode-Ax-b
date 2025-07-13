import numpy as np

def decompose_LDL(A):
    A = A.copy().astype(np.complex128)
    n = len(A)
    L = np.zeros((n, n), dtype=np.complex128)
    d = np.zeros(n, dtype=np.complex128)

    for i in range(n):
        # compute diagonal entry of D
        d[i] = A[i, i] - np.vdot(L[i, :i], L[i, :i] * d[:i])
        if d[i].real <= 0 or not np.isclose(d[i].imag, 0):
            raise ValueError("Matrix is not positive-definite.")
        # compute lower triangular entries of L
        L[i, i] = 1.0 
        for j in range(i+1, n):
            L[i, j] = (A[i, j] - np.vdot(L[j, :j], L[i, :j] * d[:j])) / d[j]
    
    return L, np.diag(d)