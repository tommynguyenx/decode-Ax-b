import numpy as np

def decompose_qr(A, method='mgs'):
    if method == 'mgs':
        return decompose_qr_mgs(A)
    elif method == 'gs':
        return decompose_qr_gs(A)
    else: 
        raise ValueError("Specified method not supported and/or invalid.")


def decompose_qr_gs(A):
    A = A.copy().astype(float) 
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        u = A[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            u -= np.dot(R[i, j], Q[:, i])

        R[j, j] = np.linalg.norm(u)
        if R[j, j] == 0:
            raise ValueError("Matrix is not full rank.")
        
        Q[:, j] = u / R[j, j]
    
    return Q, R


def decompose_qr_mgs(A):
    A = A.copy().astype(float) 
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for j in range(n):
        u_j = A[:, j]
        R[j, j] = np.linalg.norm(u_j)
        if R[j, j] == 0:
            raise ValueError("Matrix is not full rank.")
        Q[:, j] = u_j / R[j, j]
        for k in range(j+1, n):
            R[j, k] = np.dot(Q[:, j], A[:, k]) 
            A[:, k] -= R[j, k] * Q[:, j] # store u_k inside A[:, k]
            
    return Q, R