import numpy as np
from decompositions import decompose_lu, decompose_qr, decompose_cholesky, decompose_ldl
from solvers import forward_substituition, backward_substitution, diagonal_solve
 
def solve_lu(A, b):
    """
    Solves Ax = b using LU decomposition.
    """
    P, L, U = decompose_lu(A)
    y = forward_substituition(L, P @ b)
    x = backward_substitution(U, y)
    return x

def solve_qr(A, b, method='mgs'):
    """
    Solves Ax = b using QR decomposition.
    """
    Q, R = decompose_qr(A, method=method)
    x = backward_substitution(R, Q.T @ b)
    return x

def solve_cholesky(A, b, method='banachiewicz'):
    """
    Solves Ax = b using Cholesky decompisition.
    """
    L = decompose_cholesky(A, method=method)
    y = forward_substituition(L, b)
    x = backward_substitution(L.H, y)
    return x

def solve_ldl(A, b):
    """
    Solve Ax = b using LDL decomposition.
    """
    L, D = decompose_ldl(A)
    z = forward_substituition(L, b)
    y = diagonal_solve(D, z)
    x = backward_substitution(L.H, y)
    return x