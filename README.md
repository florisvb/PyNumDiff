# PyNumDiff
Methods for numerical differentiation of noisy data

## Installation
To install this package, run `python ./setup.py install` from inside this directory.

## Requirements
Minimal requirements: numpy, scipy, matplotlib
Certain methods require additional packages:
* Total Variation Regularization methods: [cvxpy](http://www.cvxpy.org/install/index.html), and a convex solver like [MOSEK](https://www.mosek.com/products/academic-licenses/) (free academic license available)
* Linear Model DMD: https://github.com/florisvb/PyDMD
* Linear Model Chebychev: [pychebfun](https://github.com/pychebfun/pychebfun/)

## Usage
`x_hat, dxdt_hat = pynumdiff.sub_module.method(x, dt, params, options)` 

## Examples
See [notebooks/1_basic_tutorial.ipynb](https://github.com/florisvb/PyNumDiff/blob/master/notebooks/1_basic_tutorial.ipynb)
