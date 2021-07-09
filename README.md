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


**PyNumDiff**: Python for Numerical Differentiation of noisy time series data.

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

PyNumDiff is a Python package that implements various methods for computing numerical derivatives of noisy data, which 
can be a critical step in developing dynamic models or designing control. There are four different families of methods 
implemented in this repository: smoothing followed by finite difference calculation, local approximation with linear 
models, Kalman filtering based methods and total variation regularization methods. Most of these methods have multiple
parameters involved to tune. We take a principled approach and propose a multi-objective optimization framework for 
choosing parameters that minimize a loss function to balance the faithfulness and smoothness of the derivative estimate.
For more details, refer to [this paper](https://doi.org/10.1109/ACCESS.2020.3034077).

## Structure

    PyNumDiff/
      |- README.md
      |- pynumdiff/
         |- __init__.py
         |- __version__.py
         |- finite_difference/
            |- ...
         |- kalman_smooth/
            |- ...
         |- linear_model/
            |- ...
         |- smooth_finite_difference/
            |- ...
         |- total_variation_regularization/
            |- ...
         |- utils/
            |- ...
         |- optimize/
            |- __init__.py
            |- __optimize__.py
            |- finite_difference/
                |- ...
            |- kalman_smooth/
                |- ...
            |- linear_model/
                |- ...
            |- smooth_finite_difference/
                |- ...
            |- total_variation_regularization/
                |- ...
         |- tests/
            |- ...
      |- examples
         |- 1_basic_tutorial.ipynb
         |- 2a_optimizing_parameters_with_dxdt_known.ipynb
         |- 2b_optimizing_parameters_with_dxdt_unknown.ipynb
      |- docs/
         |- Makefile
         |- make.bat
         |- build/
            |- ...
         |- source/
            |- _static
            |- _summaries
            |- conf.py
            |- index.rst
            |- ...
      |- setup.py
      |- .gitignore
      |- .travis.yml
      |- LICENSE
      |- requirements.txt


## Getting Started

PyNumDiff requires common packages like `numpy`, `scipy`, `matplotlib`, `pytest` (for unittests), `pylint` 
(for PEP8 style check). For a full list, you can check the file [requirements.txt](requirements.txt)

In addition, it also requires certain additional packages:
* Total Variation Regularization methods: [`cvxpy`](http://www.cvxpy.org/install/index.html)
* Linear Model Chebychev: [`pychebfun`](https://github.com/pychebfun/pychebfun/)

When using `cvxpy`, our default solver is set to be `MOSEK` (highly recommended), you would need to download their 
free academic license from their [website](https://www.mosek.com/products/academic-licenses/). Otherwise, you can also 
use other solvers which are listed [here](https://www.cvxpy.org/tutorial/advanced/index.html).

The code is compatible with Python 3.x. It can be installed using pip or directly from the source code.

### Installing via pip

`pip install pynumdiff`

### Installing from source

To install this package, run `python ./setup.py install` from inside this directory.


## Usage

**PyNumDiff** uses [Sphinx](http://www.sphinx-doc.org/en/stable/) for code documentation.
So you can see more details about the API usage [there](https://pysindy.readthedocs.io/en/latest/).

### Basic usages

* Basic Usage: you provide the parameters
```bash
        x_hat, dxdt_hat = pynumdiff.sub_module.method(x, dt, params, options)     
```
* Advanced usage: automated parameter selection through multi-objective optimization
```bash
        params, val = pynumdiff.optimize.sub_module.method(x, dt, params=None, 
                                                           tvgamma=tvgamma, # hyperparameter
                                                           dxdt_truth=None, # no ground truth data
                                                           options={})
        print('Optimal parameters: ', params)
        x_hat, dxdt_hat = pynumdiff.sub_module.method(x, dt, params, options={'smooth': True})`
```

### Notebook examples

We will frequently update simple examples for demo purposes, and here are currently exisiting ones:
* Differentiaion with different methods: [1_basic_tutorial.ipynb](examples/1_basic_tutorial.ipynb)
* Parameter Optimization with known ground truth (only for demonstration purpose):  [2a_optimizing_parameters_with_dxdt_known.ipynb](examples/2a_optimizing_parameters_with_dxdt_known.ipynb)
* Parameter Optimization with unknown ground truth:  [2b_optimizing_parameters_with_dxdt_unknown.ipynb](./examples/2b_optimizing_parameters_with_dxdt_unknown.ipynb)


### Important notes

* Larger values of `tvgamma` produce smoother derivatives
* The value of `tvgamma` is largely universal across methods, making it easy to compare method results
* The optimization is not fast. Run it on subsets of your data if you have a lot of data. It will also be much faster with faster differentiation methods, like savgoldiff and butterdiff, and probably too slow for sliding methods like sliding DMD and sliding LTI fit. 
* The following heuristic works well for choosing `tvgamma`, where `cutoff_frequency` is the highest frequency content of the signal in your data, and `dt` is the timestep: `tvgamma=np.exp(-1.6*np.log(cutoff_frequency)-0.71*np.log(dt)-5.1)`


### Running the tests

We are using Travis CI for continuous intergration testing. You can check out the current status 
[here](https://travis-ci.com/github/luckystarufo/PyNumDiff).

To run tests locally, type:
```bash
> pytest pynumdiff
```


## Citation

@ARTICLE{9241009, author={F. {van Breugel} and J. {Nathan Kutz} and B. W. {Brunton}}, journal={IEEE Access}, title={Numerical differentiation of noisy data: A unifying multi-objective optimization framework}, year={2020}, volume={}, number={}, pages={1-1}, doi={10.1109/ACCESS.2020.3034077}}


## License
This project utilizes the [MIT LICENSE](LICENSE).
100% open-source, feel free to utilize the code however you like. 