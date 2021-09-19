import numpy as np
import time
# from scipy.sparse import nb_csr
from scipy.sparse import coo_matrix, csr

from prettytable import PrettyTable

def discretise_poisson(N):
    """Generate the matrix and rhs associated with the discrete Poisson operator."""

    nelements = 5 * N**2 - 16 * N + 16

    row_ind = np.empty(nelements, dtype=np.float64)
    col_ind = np.empty(nelements, dtype=np.float64)
    data = np.empty(nelements, dtype=np.float64)

    f = np.empty(N * N, dtype=np.float64)

    count = 0
    for j in range(N):
        for i in range(N):
            if i == 0 or i == N - 1 or j == 0 or j == N - 1:
                row_ind[count] = col_ind[count] = j * N + i
                data[count] = 1
                f[j * N + i] = 0
                count += 1

            else:
                row_ind[count : count + 5] = j * N + i
                col_ind[count] = j * N + i
                col_ind[count + 1] = j * N + i + 1
                col_ind[count + 2] = j * N + i - 1
                col_ind[count + 3] = (j + 1) * N + i
                col_ind[count + 4] = (j - 1) * N + i

                data[count] = 4 * (N - 1)**2
                data[count + 1 : count + 5] = - (N - 1)**2
                f[j * N + i] = 1

                count += 5
    # print(data.shape, (row_ind.max(), col_ind.max()),N**2)
    return coo_matrix((data, (row_ind, col_ind)), shape=(N**2, N**2)).tocsr(), f


scipy_times = []
nb_times= []
N = 1000

A, _ = discretise_poisson(N)
# A_nb = nb_csr.csr_matrix(A)

rand = np.random.RandomState(0)
x = rand.randn(N * N)
loops = [100,200,500,1000]
for loop in loops:
    t0 = time.time()
    for i in range(loop):
        y = A@x
    t1 = time.time()
    t_sc = t1-t0
    scipy_times.append(t_sc/loop)
    

tb =  PrettyTable()
tb.field_names = [""]+["{} loops".format(loop) for loop in loops]
# tb.add_row(["scipy"]+["{:0.4f} ms".format(time*1000) for time in scipy_times])
tb.add_row(["numba"]+["{:0.4f} ms".format(time*1000) for time in nb_times])
print("Test Matvec speed betweend numba and scipy")
print(tb)