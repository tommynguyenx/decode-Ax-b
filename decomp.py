import numpy as np

class LU: # only for square matrices
    def __init__(self, A):
        self.A = A
        self.n = len(A)

    def decompose(self): # with partial pivoting
        A = np.copy(self.A).astype(float)
        perm = np.arange(self.n)

        for k in range(self.n - 1):

            # find pivot row
            p = np.argmax(np.abs(A[k:, k])) + k
            if p != k:
                A[[k, p], :] = A[[p, k], :]
                perm[k], perm[p] = perm[p], perm[k]

            if A[k, k] == 0:
                continue

            # eliminate entries below the pivot
            for i in range(k + 1, self.n):
                A[i, k] /= A[k, k]
                A[i, k + 1:] -= A[i, k] * A[k, k + 1:]

            # get P, L, U matrices
            P = np.eye(self.n)[perm]
            L = np.tril(A, k=-1) + np.eye(self.n)
            U = np.triu(A)

        return P, L, U
    

class QR:
    def __init__(self, A):
        self.A = A
        self.m, self.n = A.shape

    def project(self, u, a):
        return (np.dot(u, a) / np.dot(u, u)) * u

    def decomposeGS(self):
        A = np.copy(self.A).astype(float)
        Q = np.zeros((self.m, self.n))
        R = np.zeros((self.n, self.n))

        for j in range(self.n):
            u = A[:, j]
            for i in range(j):
                R[i, j] = np.dot(Q[:, i], A[:, j])
                u -= self.project(Q[:, i], A[:, j])

            R[j, j] = np.linalg.norm(u)
            if R[j, j] == 0:
                raise ValueError("Matrix is not full rank.")
            
            Q[:, j] = u / R[j, j]
        
        return Q, R
    
    def decomposeMGS(self):
        A = np.copy(self.A).astype(float)
        Q = np.zeros((self.m, self.n))
        R = np.zeros((self.n, self.n))

        for j in range(self.n):
            u = A[:, j]
            for i in range(j):
                R[i, j] = np.dot(Q[:, i], u)
                u -= self.project(Q[:, i], u)

            R[j, j] = np.linalg.norm(u)
            if R[j, j] == 0:
                raise ValueError("Matrix is not full rank.")
            
            Q[:, j] = u / R[j, j]
        
        return Q, R
            