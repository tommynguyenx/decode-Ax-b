import numpy as np

def decompose_lu(A): # partial pivoting
    A = A.copy().astype(float)
    n = len(A)
    perm = np.arange(n)

    for k in range(n - 1):

        # find pivot row
        p = np.argmax(np.abs(A[k:, k])) + k
        if p != k:
            A[[k, p], :] = A[[p, k], :]
            perm[k], perm[p] = perm[p], perm[k]

        if A[k, k] == 0:
            continue

        # eliminate entries below the pivot
        for i in range(k + 1, n):
            A[i, k] /= A[k, k]
            A[i, k + 1:] -= A[i, k] * A[k, k + 1:]

        # get P, L, U matrices
    P = np.eye(n)[perm]
    L = np.tril(A, k=-1) + np.eye(n)
    U = np.triu(A)

    return P, L, U