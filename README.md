# Introductions
The currently available tools for sparse linear algebra in Scipy are not very good for larger problems. The goal is to develop a high-performance sparse linear algebra interface for a single workstation that integrates fully within Scipy.

•	Interfacing to algebraic multigrid (AMG) solvers and other preconditioners. Create interfaces to widely used algebraic multigrid packages (e.g. ML, others) and preconditioners.

Potentially write a custom AMG Implementation referred on PyAMG git-repository https://github.com/pyamg.

Our aim is to speed up PyAMG on the using of finite elements.

All software developments must be well implemented using PEP8 Python coding standards, easy to use, well documented and easy to install using conda or pip.

# Installation

`python 3.6`
### python packages
conda install numba tbb cudatoolkit
pip install -r requirements.txt
### build scipy
pip install numba Cython
sudo apt-get install gcc gfortran python3-dev libopenblas-dev liblapack-dev
cd scipy
<!-- git checkout maintenance/1.5.x -->
python setup.py install

## tests
python test_csr_matvec.py

## Step 1: rewrite csr_matrix operator

### opretors that supports csr matrix in SciPy

|operator|A|B|scipy sparse|
|-|-|-|-|
|*,@|sparse matrix|vector|matrix multiplication,dot,call `csr_matvec`|
|*,@|sparse matrix|sparse matrix|matrix multiplication, dot, call `_mul_sparse_matrix`->`csr_matmat`|
|`multiply`|sparse matrix(M,N)|sparse matrix(M,N)|(M,N)*(M,N)，`multiply`-> `_binopt`->`csr_elmul_csr`|
|`multiply`|sparse matrix(M,1)|sparse matrix(1,N)|(M,1)@(1,N),`multiply`-> `_mul_sparse_matrix`|
|`multiply`|sparse matrix(1,N)|sparse matrix(M,1)|(M,1)@(1,N)，`multiply`-> `_mul_sparse_matrix`|
|`multiply`|sparse matrix(M,N)|sparse matrix(1,N)|(M,N)*(1,N)，`multiply`-> `_mul_sparse_matrix`|
|`multiply`|sparse matrix(M,N)|sparse matrix(M,1)|(M,N)*(M,1)，`multiply`-> `_mul_sparse_matrix`|
|`multiply`|sparse matrix(M,1)|sparse matrix(M,N)|(M,1)*(M,N)，`multiply`-> `_mul_sparse_matrix`|
|+|sparse matrix(M,N)|sparse matrix(M,N)|(M,N)+(M,N), `_add_sparse`->`_binopt`->`csr_plus_csr`|
|-|sparse matrix(M,N)|sparse matrix(M,N)|(M,N)+(M,N), `_add_sparse`->`_binopt`->`csr_minus_csr`|
### TODO
- [ ] matvec

###  matvec
The calling interface is `_mul_vector` of `nb_compresed`
The implementaation operator is `csr_matvec` of `nb_sparsetools`

Ref to `Sparse Matrix data structures`,it is found that the implementation speed of numba is very slow.
    
    -Install numba tbb with conda instead of pip.
      - When the number of calculations is < 100, it is slightly slower than SciPy
      - Between 100 and 200, speed is almost the same
      - Calculation times > 200, slightly faster than SciPy
    
### mul_scalar

Little acceleration effect
### * is dot, call `_mul_sparse_matrix` function

### * -> multi
## issues

#### NumbaWarning: The TBB threading layer requires TBB version 2019.5 or later i.e., TBB_INTERFACE_VERSION >= 11005. Found TBB_INTERFACE_VERSION = 9002. The TBB threading layer is disabled.

[numba docs](https://numba.pydata.org/numba-doc/latest/user/threading-layer.html#setting-the-threading-layer)

see `numba -s` has a section __Threading Layer Information__

Solution: Use `conda` to install rather than `pip`