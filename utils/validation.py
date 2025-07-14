import numpy as np

def validate_matrix_square(A):
    return A.shape[0] == A.shape[1]

def validate_matrix_full_col_rank(A):
    return np.linalg.matrix_rank(A) == A.shape[0]

def validate_matrix_positive_definite(A):
    return validate_matrix_square(A) and np.all(np.linalg.eigvals(A) > 0)