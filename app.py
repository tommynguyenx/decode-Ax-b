import streamlit as st
import numpy as np
from decompositions import decompose_lu, decompose_qr, decompose_cholesky, decompose_ldl
from solvers import forward_substituition, backward_substitution
from utils import validate_matrix_square, validate_matrix_full_col_rank, validate_matrix_positive_definite

# function to format matrices for Latex display
def latex_matrix(name, matrix, decimals=3):
    return f"{name} = " + r"\begin{bmatrix}" + \
        r" \\".join([" & ".join([f"{x:.{decimals}f}" for x in row]) for row in matrix]) + \
        r"\end{bmatrix}"

# function to create a section header
def section_header(text, color="#f0f2f6", text_color="#222", border_color="#ffa500"):
    st.markdown(
        f"""
        <div style="
            background-color:{color};
            padding:12px;
            border-radius:8px;
            margin-bottom:8px;
            border: 2px solid {border_color};
        ">
            <h3 style="margin:0;color:{text_color};">{text}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )


st.title("Matrix Decomposition")

# Input mode selection
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


st.write("")
st.write("")
# Add a button to trigger decomposition
decomp_options = st.multiselect(
    "Choose decompositions to display:",
    ["LU", "QR", "Cholesky"],
    default=["LU"]
)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    decompose_clicked = st.button("Decompose", use_container_width=True)


st.write("")
st.write("")
st.write("")
if decompose_clicked:
    section_color = "#ffeacc"  # Light orange

    if "LU" in decomp_options:
        section_header("LU Decomposition Results", section_color, "#222")
        if validate_matrix_square(A):
            P, L, U = decompose_lu(A)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.latex(latex_matrix("P", P, decimals=0))
            with col2:
                st.latex(latex_matrix("L", L))
            with col3:
                st.latex(latex_matrix("U", U))
        else: 
            st.error("Matrix A must be square for LU decomposition.")
        st.write("")  # Adds a little space
        st.write("")

    if "QR" in decomp_options:
        section_header("QR Decomposition Results", section_color, "#222")
        if validate_matrix_full_col_rank(A):
            Q, R = decompose_qr(A)
            col1, col2 = st.columns(2)
            with col1:
                st.latex(latex_matrix("Q", Q))
            with col2:
                st.latex(latex_matrix("R", R))
        else:
            st.error("Matrix A must be full column rank for QR decomposition.")
        st.write("")  # Adds a little space
        st.write("")

    if "Cholesky" in decomp_options:
        section_header("Cholesky Decomposition Results", section_color, "#222")
        if validate_matrix_positive_definite(A):
            L = decompose_cholesky(A)
            st.latex(latex_matrix("L", L))
        else:
            st.error("Matrix A must be positive definite for Cholesky decomposition.")
        st.write("")  # Adds a little space
        st.write("")