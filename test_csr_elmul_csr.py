import numba
import numba as nb
import numpy as np

import scipy as sp
from scipy import sparse
from scipy.sparse import _sparsetools
from scipy.sparse.sputils import (asmatrix, check_shape, downcast_intp_index,
                                  get_index_dtype, get_sum_dtype, getdtype,
                                  is_pydata_spmatrix, isdense, isintlike,
                                  isscalarlike, isshape, matrix, to_native,
                                  upcast, upcast_char)
import time
from prettytable import PrettyTable

def np_csr_elmul_csr(n_row, n_col,
           Ap,
           Aj,
           Ax,
           Bp,
           Bj,
           Bx,
           Cp, 
           Cj, 
           Cx):
    Cp[0] = 0;
    nnz = 0;
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
                    nnz+=1
                A_pos+=1
                B_pos+=1
            elif (A_j < B_j):
                A_pos+=1
            else:
                B_pos+=1
        Cp[i+1] = nnz;

@numba.jit(nopython=True, parallel=True)
def csr_elmul_csr(n_row, n_col,
           Ap,
           Aj,
           Ax,
           Bp,
           Bj,
           Bx,
           Cp, 
           Cj, 
           Cx):
    Cp[0] = 0;
    nnz = 0;
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
                    nnz+=1
                A_pos+=1
                B_pos+=1
            elif (A_j < B_j):
                A_pos+=1
            else:
                B_pos+=1
        Cp[i+1] = nnz;


@numba.jit(nopython=True, parallel=True,cache=True)
def csr_elmul_csr(n_row, n_col,
           Ap,
           Aj,
           Ax,
           Bp,
           Bj,
           Bx,
           Cp, 
           Cj, 
           Cx):
    Cp[0] = 0;
    nnz = 0;
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
                    nnz+=1
                A_pos+=1
                B_pos+=1
            elif (A_j < B_j):
                A_pos+=1
            else:
                B_pos+=1
        Cp[i+1] = nnz;


@numba.jit(nopython=True, parallel=True)
def csr_elmul_csr_dev(n_row, n_col,
           Ap,
           Aj,
           Ax,
           Bp,
           Bj,
           Bx,
           Cp, 
           Cj, 
           Cx):
    Cp[0] = 0;
    nnz = 0;
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
                    nnz+=1
                A_pos+=1
                B_pos+=1
            elif (A_j < B_j):
                A_pos+=1
            else:
                B_pos+=1
        Cp[i+1] = nnz;

scipy_csr_elmul_csr = getattr(_sparsetools, "csr_elmul_csr")

m = 200
n = 400
m1 = np.random.randint(0,100,(m,n))
m1[m1>50]=0
m2 = np.random.randint(0,100,(m,n))
m2[m2>50]=0

m1x = sparse.csr_matrix(m1)
m2x = sparse.csr_matrix(m2)

maxnnz = m1x.nnz + m2x.nnz
idx_dtype = get_index_dtype((m1x.indptr, m1x.indices,
                             m2x.indptr, m2x.indices),
                            maxval=maxnnz)
indptr = np.empty(m1x.indptr.shape, dtype=idx_dtype)
indices = np.empty(maxnnz, dtype=idx_dtype)
bool_ops = ['_ne_', '_lt_', '_gt_', '_le_', '_ge_']
if 'enul' in bool_ops:
    data = np.empty(maxnnz, dtype=np.bool_)
else:
    data = np.empty(maxnnz, dtype=upcast(m1x.dtype, m2x.dtype))
    
    
n_row,n_col = m1.shape
Ap = m1x.indptr
Aj = m1x.indices
Ax = m1x.data
Bp = m2x.indptr
Bj = m2x.indices
Bx = m2x.data
Cp = indptr
Cj = indices
Cx = data
loops = [100,500,1000]
tb =  PrettyTable()
tb.field_names = [""]+["{} loops".format(loop) for loop in loops]

times = []
for loop in loops:
    t0 = time.time()
    for i in range(loop):
        np_csr_elmul_csr(n_row, n_col,
           Ap,
           Aj,
           Ax,
           Bp,
           Bj,
           Bx,
           Cp, 
           Cj, 
           Cx)
    t1 = time.time()
    t_sc = t1-t0
    times.append(t_sc/loop)
tb.add_row(["numpy"]+["{:0.4f} ms".format(time*1000) for time in times])

times = []
for loop in loops:
    t0 = time.time()
    for i in range(loop):
        scipy_csr_elmul_csr(n_row, n_col,
           Ap,
           Aj,
           Ax,
           Bp,
           Bj,
           Bx,
           Cp, 
           Cj, 
           Cx)
    t1 = time.time()
    t_sc = t1-t0
    times.append(t_sc/loop)
tb.add_row(["scipy"]+["{:0.4f} ms".format(time*1000) for time in times])
times = []

csr_elmul_csr(n_row, n_col,
           Ap,
           Aj,
           Ax,
           Bp,
           Bj,
           Bx,
           Cp, 
           Cj, 
           Cx)
for loop in loops:
    t0 = time.time()
    for i in range(loop):
        csr_elmul_csr(n_row, n_col,
           Ap,
           Aj,
           Ax,
           Bp,
           Bj,
           Bx,
           Cp, 
           Cj, 
           Cx)
    t1 = time.time()
    t_sc = t1-t0
    times.append(t_sc/loop)
tb.add_row(["numba"]+["{:0.4f} ms".format(time*1000) for time in times])

# tb.add_row(["scipy"]+["{:0.4f} ms".format(time*1000) for time in scipy_times])

print("Test csr_elmul_csr ({}) speed betweend numba and scipy".format("x".join([str(s) for s in m1.shape])))
print(tb)