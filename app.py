import streamlit as st
import numpy as np
from decompositions import decompose_lu, decompose_qr, decompose_cholesky, decompose_ldl
from solvers import forward_substituition, backward_substitution
from utils import validate_matrix_square, validate_matrix_full_col_rank, validate_matrix_positive_definite


def latex_matrix(name, matrix, decimals=3):
    return f"{name} = " + r"\begin{bmatrix}" + \
        r" \\".join([" & ".join([f"{x:.{decimals}f}" for x in row]) for row in matrix]) + \
        r"\end{bmatrix}"


st.title("Matrix Decomposition")

input_mode = st.radio("Choose input mode:", ["Text (Python list)", "Editable Matrix"])

if input_mode == "Text (Python list)":
    A_input = st.text_area("Enter matrix A (Python list format)", "[[4, 1], [1, 3]]")
    A = np.array(eval(A_input), dtype=float)
else:
    num_rows = st.number_input("Number of rows", min_value=2, value=2)
    num_cols = st.number_input("Number of columns", min_value=2, value=2)
    default_matrix = np.zeros((int(num_rows), int(num_cols)), dtype=float)
    A_df = st.data_editor(
        default_matrix,
        num_rows="dynamic",
        use_container_width=True
    )
    A = np.array(A_df, dtype=float)

st.header("LU Decomposition Results")
if validate_matrix_square(A):
    P, L, U = decompose_lu(A)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.latex(latex_matrix("P", P, decimals=0))
    with col2:
        st.latex(latex_matrix("L", L))
    with col3:
        st.latex(latex_matrix("U", U))
    st.write("where A = P.T @ L @ U")
else: 
    st.error("Matrix A must be square for LU decomposition.")



st.header("QR Decomposition Results")
if validate_matrix_full_col_rank(A):
    Q, R = decompose_qr(A)
    col1, col2 = st.columns(2)
    with col1:
        st.latex(latex_matrix("Q", Q))
    with col2:
        st.latex(latex_matrix("R", R))
else:
    st.error("Matrix A must be full column rank for QR decomposition.")


st.header("Cholesky Decomposition Results")
if validate_matrix_positive_definite(A):
    L = decompose_cholesky(A)
    st.latex(latex_matrix("L", L))
else:
    st.error("Matrix A must be positive definite for Cholesky decomposition.")