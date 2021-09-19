import numpy as np
import time
from scipy.sparse import nb_csr
from scipy.sparse import coo_matrix, csr

from prettytable import PrettyTable

src = np.random.randint(0,100,(100,100))
src[src>50]=0
loops = [100,200,500,1000]
scipy_times = []
nb_times= []
N = 100

A = csr.csr_matrix(src)
A_nb = nb_csr.csr_matrix_nb(src)

print(A_nb.toarray())

for loop in loops:
    t0 = time.time()
    for i in range(loop):
        y = A_nb+A_nb
        print(y.toarray())
    t1 = time.time()
    t_nb = t1-t0
    nb_times.append(t_nb/loop)
    t0 = time.time()
    for i in range(loop):
        y = A*A
    t1 = time.time()
    t_sc = t1-t0
    scipy_times.append(t_sc/loop)
    

tb =  PrettyTable()
tb.field_names = [""]+["{} loops".format(loop) for loop in loops]
tb.add_row(["scipy"]+["{:0.4f} ms".format(time*1000) for time in scipy_times])
tb.add_row(["numba"]+["{:0.4f} ms".format(time*1000) for time in nb_times])
print("Test Matvec speed betweend numba and scipy")
print(tb)