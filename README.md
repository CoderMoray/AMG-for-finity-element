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
