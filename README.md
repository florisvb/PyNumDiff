<p align="center">
  <a href="https://pynumdiff.readthedocs.io/en/latest/" target="_blank" >
    <img alt="Python for Numerical Differentiation of noisy time series data" src="docs/source/_static/logo_PyNumDiff.png" width="300" height="200" />
  </a>
</p>

<p align="center">
    <a href='https://pynumdiff.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/pynumdiff/badge/?version=latest' alt='Documentation Status' /></a>
    <a href="#travis" alt="Travis Build Status">
        <img src="https://travis-ci.com/luckystarufo/PyNumDiff.svg?branch=upgrade"/></a>
</p>


**PyNumDiff**: Python for Numerical Differentiation of noisy time series data

## Table of contents
* [Introduction](#introduction)
* [Structure](#structure)
* [Getting Started](#getting-started)
    * [Prerequisite](#prerequisite)
    * [Installing](#installing)
* [Usage](#usage)
    * [Examples](#examples)
    * [Running the tests](#running-the-tests)
* [Citation](#citation)
* [Contributing](#contributing)
* [License](#license)
* [Acknowledgments](#acknowledgments)

## Introduction

## Structure

## Getting Started

### Installing via pip

`pip install pynumdiff`

### Installing from source

To install this package, run `python ./setup.py install` from inside this directory.

#### Requirements
Python version: 2.7+

Minimal requirements: `numpy, scipy, matplotlib`

Certain methods require additional packages:
* Total Variation Regularization methods: [cvxpy](http://www.cvxpy.org/install/index.html), and a convex solver like [MOSEK](https://www.mosek.com/products/academic-licenses/) (free academic license available)
* Linear Model DMD: https://github.com/florisvb/PyDMD
* Linear Model Chebychev: [pychebfun](https://github.com/pychebfun/pychebfun/)

To run the notebooks and generate figures in notebooks/paper_figures you will also need:
* [figurefirst](https://github.com/FlyRanch/figurefirst)

## Usage

* Basic Usage: you provide the parameters
`x_hat, dxdt_hat = pynumdiff.sub_module.method(x, dt, params, options)` 

* For example, a few favorites:
    x_hat, dxdt_hat = pynumdiff.linear_model.savgoldiff(x, dt, [3, 20, 25])
    x_hat, dxdt_hat = pynumdiff.kalman_smooth.constant_acceleration(x, dt, [1e-1, 1e-2])
    x_hat, dxdt_hat = pynumdiff.total_variation_regularization.jerk(x, dt, [10])
    x_hat, dxdt_hat = pynumdiff.smooth_finite_difference.butterdiff(x, dt, [3, 0.07])

* Comprehensive examples:
In jupyter notebook form: [notebooks/1_basic_tutorial.ipynb](https://github.com/florisvb/PyNumDiff/blob/master/notebooks/1_basic_tutorial.ipynb)


* Advanced usage: automated parameter selection through multi-objective optimization
This approach solves a loss function that balances the faithfulness and smoothness of the derivative estimate, and relies on a single hyperparameter, gamma, or `tvgamma` in the code. See the paper for more detail, but a brief overview is given in the example notebooks linked below.

    params, val = pynumdiff.optimize.sub_module.method(x, dt, params=None, 
                                                       tvgamma=tvgamma, # hyperparameter
                                                       dxdt_truth=None, # no ground truth data
                                                       options={})
    print('Optimal parameters: ', params)
    x_hat, dxdt_hat = pynumdiff.sub_module.method(x, dt, params, options={'smooth': True})

* Important points:
* Larger values of `tvgamma` produce smoother derivatives
* The value of `tvgamma` is largely universal across methods, making it easy to compare method results
* The optimization is not fast. Run it on subsets of your data if you have a lot of data. It will also be much faster with faster differentiation methods, like savgoldiff and butterdiff, and probably too slow for sliding methods like sliding DMD and sliding LTI fit. 
* The following heuristic works well for choosing `tvgamma`, where `cutoff_frequency` is the highest frequency content of the signal in your data, and `dt` is the timestep. 

`tvgamma = np.exp(   -1.6*np.log(cutoff_frequency) -0.71*np.log(dt) - 5.1   )`



### Examples

* Ground truth data known:  [notebooks/2a_optimizing_parameters_with_dxdt_known.ipynb](https://github.com/florisvb/PyNumDiff/blob/master/notebooks/2a_optimizing_parameters_with_dxdt_known.ipynb)
* Ground truth data NOT known (real world):  [notebooks/2b_optimizing_parameters_with_dxdt_unknown.ipynb](https://github.com/florisvb/PyNumDiff/blob/master/notebooks/2b_optimizing_parameters_with_dxdt_unknown.ipynb)

### Running the tests

## Citation

If you use this package for your differentiation needs, please cite [this paper](https://doi.org/10.1109/ACCESS.2020.3034077).

F. van Breugel, J. Nathan Kutz and B. W. Brunton, "Numerical differentiation of noisy data: A unifying multi-objective optimization framework," in IEEE Access, doi: 10.1109/ACCESS.2020.3034077.

@ARTICLE{9241009, author={F. {van Breugel} and J. {Nathan Kutz} and B. W. {Brunton}}, journal={IEEE Access}, title={Numerical differentiation of noisy data: A unifying multi-objective optimization framework}, year={2020}, volume={}, number={}, pages={1-1}, doi={10.1109/ACCESS.2020.3034077}}

## Contributing

## License

## Acknowledgments