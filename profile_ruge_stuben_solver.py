import pyamg
import numpy as np
import  time
from prettytable import PrettyTable

A = pyamg.gallery.poisson((500,500), format='csr')  # 2D Poisson problem on 500x500 grid
# print('A : ', A )
# ml_nb = pyamg.ruge_stuben_solver(A,use_nb=True)     
ml = pyamg.ruge_stuben_solver(A) # construct the multigrid hierarchy
# print(ml)                                           # print hierarchy information
b = np.random.rand(A.shape[0])                      # pick a random right hand side
# t0=
                        # solve Ax=b to a tolerance of 1e-10
# print("residual: ", np.linalg.norm(b-A*x))  
y = ml.solve(b, tol=1e-10)
# loops = [100]
# nb_times = []
# scipy_times=[]
# for loop in loops:
#     t0 = time.time()
#     for i in range(loop):
        
#         x = ml_nb.solve(b, tol=1e-10)  
#     t1 = time.time()
#     t_nb = t1-t0
#     print('t_nb: ', t_nb)
#     nb_times.append(t_nb/loop)
#     t0 = time.time()
#     for i in range(loop):
#         y = ml.solve(b, tol=1e-10)  
#     t1 = time.time()
#     t_sc = t1-t0
#     scipy_times.append(t_sc/loop)

# tb =  PrettyTable()
# tb.field_names = [""]+["{} loops".format(loop) for loop in loops]
# tb.add_row(["scipy"]+["{:0.4f} ms".format(time*1000) for time in scipy_times])
# tb.add_row(["numba"]+["{:0.4f} ms".format(time*1000) for time in nb_times])
# print("Test Matvec speed betweend numba and scipy")
# print(tb)