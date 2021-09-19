import numba
# from numba.np.ufunc import parallel
import numpy as np
from numba import cuda
# print(cuda.detect())

# numba.config.THREADING_LAYER = 'tbb'
# multiplication
@numba.jit(nopython=True, parallel=True)
def csr_matvec(M, N, indptr, indices, data, other):
    y = np.zeros(M, dtype=np.float64)

    for row_index in numba.prange(M):
        col_start = indptr[row_index]
        col_end = indptr[row_index + 1]
        for col_index in range(col_start, col_end):
            y[row_index] += data[col_index] * other[indices[col_index]]
    return y

def csr_matmat_maxnnz(M,N,):
    pass

def csr_elmul_csr():
    pass

def csr_matmat():
    pass

# def csr_

def csr_ne_csr():
    pass

def csr_add_scr():
    pass

def csr_minus_csr():
    pass