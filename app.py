import streamlit as st
import numpy as np
from decompositions import decompose_lu
from solvers import forward_substituition, backward_substitution

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

st.write("**Matrix A:**")
latex_matrix_A = r"\begin{bmatrix}" + \
    r" \\".join([" & ".join(map(str, row)) for row in A]) + \
    r"\end{bmatrix}"
st.latex(latex_matrix_A)

P, L, U = decompose_lu(A)

st.write("**Matrix P:**")
latex_matrix_P = r"\begin{bmatrix}" + \
    r" \\".join([" & ".join(map(str, row)) for row in P]) + \
    r"\end{bmatrix}"
st.latex(latex_matrix_P)

st.write("**Matrix L:**")
latex_matrix_L = r"\begin{bmatrix}" + \
    r" \\".join([" & ".join(map(str, row)) for row in L]) + \
    r"\end{bmatrix}"
st.latex(latex_matrix_L)

st.write("**Matrix U:**")
latex_matrix_U = r"\begin{bmatrix}" + \
    r" \\".join([" & ".join(map(str, row)) for row in U]) + \
    r"\end{bmatrix}"
st.latex(latex_matrix_U)

st.write("**Reconstruction of A = P.T @ L @ U:**")
A_reconstructed = P.T @ L @ U
latex_matrix_re = r"\begin{bmatrix}" + \
    r" \\".join([" & ".join(map(str, row)) for row in A_reconstructed]) + \
    r"\end{bmatrix}"
st.latex(latex_matrix_re)