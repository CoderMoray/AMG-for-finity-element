import pyamg
import numpy as np
import  time
from prettytable import PrettyTable

A = pyamg.gallery.poisson((500,500), format='csr')  # 2D Poisson problem on 500x500 grid
# print('A : ', A )
ml = pyamg.ruge_stuben_solver(A)     
b = np.random.rand(A.shape[0])                      # pick a random right hand side
# t0=
                        # solve Ax=b to a tolerance of 1e-10
# print("residual: ", np.linalg.norm(b-A*x))  
loops = [100]
nb_times = []
times=[]
for loop in loops:

    t0 = time.time()
    for i in range(loop):
        y = ml.solve(b, tol=1e-10)  
    t1 = time.time()
    t_sc = t1-t0
    times.append(t_sc/loop)

tb =  PrettyTable()
tb.field_names = [""]+["{} loops".format(loop) for loop in loops]
tb.add_row(["ruge stuben"]+["{:0.4f} ms".format(time*1000) for time in times])
print(tb)