import numba
# from numba.np.ufunc import parallel
import numpy as np
from numba import cuda
# print(cuda.detect())

# numba.config.THREADING_LAYER = 'tbb'
# multiplication
@numba.jit(nopython=True, parallel=True,cache=True)
def csr_matvec(M, N, indptr, indices, data, other):
    y = np.zeros(M, dtype=np.float64)

    for row_index in numba.prange(M):
        col_start = indptr[row_index]
        col_end = indptr[row_index + 1]
        for col_index in range(col_start, col_end):
            y[row_index] += data[col_index] * other[indices[col_index]]
    return y

@numba.jit(nopython=True, parallel=True,cache=True)
def csr_elmul_csr(n_row, 
                  n_col,
                    Ap,
                    Aj,
                    Ax,
                    Bp,
                    Bj,
                    Bx,
                    Cp, 
                    Cj, 
                    Cx):
    Cp[0] = 0
    nnz = 0
    for i in numba.prange(n_row):
        A_pos = Ap[i]
        B_pos = Bp[i]
        A_end = Ap[i+1]
        B_end = Bp[i+1]
        #while not finished with either row
        while(A_pos < A_end and B_pos < B_end):
            A_j = Aj[A_pos]
            B_j = Bj[B_pos]
            if(A_j == B_j):
                result = Ax[A_pos]*Bx[B_pos]
                if(result != 0):
                    Cj[nnz] = A_j
                    Cx[nnz] = result
                    nnz += 1
                A_pos += 1
                B_pos += 1
            elif (A_j < B_j):
                A_pos += 1
            else:
                B_pos += 1
        Cp[i+1] = nnz


def csr_matmat():
    pass

# def csr_

def csr_ne_csr():
    pass

def csr_add_scr():
    pass

def csr_minus_csr():
    pass
