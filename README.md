# PyNumDiff

Python methods for numerical differentiation of noisy data, including multi-objective optimization routines for automated parameter selection.

<p align="center">
  <a href="https://pynumdiff.readthedocs.io/en/master/" target="_blank" >
    <img alt="Python for Numerical Differentiation of noisy time series data" src="docs/source/_static/logo_PyNumDiff.png" width="300" height="200" />
  </a>
</p>

<p align="center">
    <a href='https://pynumdiff.readthedocs.io/en/master/?badge=master'>
        <img src='https://readthedocs.org/projects/pynumdiff/badge/?version=master' alt='Documentation Status' /></a>
    <a href="https://badge.fury.io/py/pynumdiff">
        <img src="https://badge.fury.io/py/pynumdiff.svg" alt="PyPI version" height="18"></a>
    <a href="https://zenodo.org/badge/latestdoi/159711175">
        <img src="https://zenodo.org/badge/159711175.svg" alt="DOI"></a>
    <a href="https://joss.theoj.org/papers/102257ee4b0142bf49bc18d7c810e9d5">
        <img src="https://joss.theoj.org/papers/102257ee4b0142bf49bc18d7c810e9d5/status.svg"></a>
</p>

## Table of contents
- [PyNumDiff](#pynumdiff)
  - [Table of contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Structure](#structure)
  - [Citation](#citation)
      - [PyNumDiff python package:](#pynumdiff-python-package)
      - [Optimization algorithm:](#optimization-algorithm)
  - [Getting Started](#getting-started)
    - [Prerequisite](#prerequisite)
    - [Installing](#installing)
  - [Usage](#usage)
    - [Basic usages](#basic-usages)
    - [Notebook examples](#notebook-examples)
    - [Important notes](#important-notes)
    - [Running the tests](#running-the-tests)
  - [License](#license)

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
         |- kalman_smooth/
         |- linear_model/
         |- smooth_finite_difference/
         |- total_variation_regularization/
         |- utils/
         |- optimize/
            |- __init__.py
            |- __optimize__.py
            |- finite_difference/
            |- kalman_smooth/
            |- linear_model/
            |- smooth_finite_difference/
            |- total_variation_regularization/
         |- tests/
      |- examples
         |- 1_basic_tutorial.ipynb
         |- 2a_optimizing_parameters_with_dxdt_known.ipynb
         |- 2b_optimizing_parameters_with_dxdt_unknown.ipynb
      |- docs/
         |- Makefile
         |- make.bat
         |- build/
         |- source/
            |- _static
            |- _summaries
            |- conf.py
            |- index.rst
            |- ...
      |- .gitignore
      |- LICENSE.txt
      |- pyproject.toml

## Citation

See CITATION.cff file as well as the following references.

#### PyNumDiff python package:

    @article{PyNumDiff2022,
      doi = {10.21105/joss.04078},
      url = {https://doi.org/10.21105/joss.04078},
      year = {2022},
      publisher = {The Open Journal},
      volume = {7},
      number = {71},
      pages = {4078},
      author = {Floris van Breugel and Yuying Liu and Bingni W. Brunton and J. Nathan Kutz},
      title = {PyNumDiff: A Python package for numerical differentiation of noisy time-series data},
      journal = {Journal of Open Source Software}
    }


#### Optimization algorithm:

    @article{ParamOptimizationDerivatives2020, 
    doi={10.1109/ACCESS.2020.3034077}
    author={F. {van Breugel} and J. {Nathan Kutz} and B. W. {Brunton}}, 
    journal={IEEE Access}, 
    title={Numerical differentiation of noisy data: A unifying multi-objective optimization framework}, 
    year={2020}
    }

## Getting Started

### Prerequisite

PyNumDiff requires common packages like `numpy`, `scipy`, `matplotlib`, `pytest` (for unittests), `pylint` 
(for PEP8 style check). For a full list, you can check the file [pyproject.toml](pyproject.toml)

In addition, it also requires certain additional packages for select functions, though these are not required for a successful install of PyNumDiff:
* Total Variation Regularization methods: [`cvxpy`](http://www.cvxpy.org/install/index.html)

When using `cvxpy`, our default solver is set to be `MOSEK` (highly recommended), you would need to download their 
free academic license from their [website](https://www.mosek.com/products/academic-licenses/). Otherwise, you can also 
use other solvers which are listed [here](https://www.cvxpy.org/tutorial/advanced/index.html).

### Installing

The code is compatible with >=Python 3.5. It can be installed using pip or directly from the source code. Basic installation options include:

* From PyPI using pip: `pip install pynumdiff`.
* From source using pip git+: `pip install git+https://github.com/florisvb/PyNumDiff`
* From local source code using setup.py: Run `pip install .` from inside this directory. See below for example.

For additional solvers, run `pip install pynumdiff[advanced]`.  This includes `cvxpy`,
which can be tricky when compiling from source.  If an error occurs in installing
`cvxpy`, see [cvxpy install documentation](https://www.cvxpy.org/install/), install
`cvxpy` according to those instructions, and try `pip install pynumdiff[advanced]`
again.

<em>Note: If using the optional MOSEK solver for cvxpy you will also need a [MOSEK license](https://www.mosek.com/products/academic-licenses/), free academic license.</em>


## Usage

**PyNumDiff** uses [Sphinx](http://www.sphinx-doc.org/en/stable/) for code documentation.
So you can see more details about the API usage [there](https://pynumdiff.readthedocs.io/en/latest/).

### Basic usages

* Basic Usage: you provide the parameters
```bash
        x_hat, dxdt_hat = pynumdiff.sub_module.method(x, dt, params, options)     
```
* Intermediate usage: automated parameter selection through multi-objective optimization
```bash
        params, val = pynumdiff.optimize.sub_module.method(x, dt, params=None, 
                                                           tvgamma=tvgamma, # hyperparameter
                                                           dxdt_truth=None, # no ground truth data
                                                           options={})
        print('Optimal parameters: ', params)
        x_hat, dxdt_hat = pynumdiff.sub_module.method(x, dt, params, options={'smooth': True})`
```
* Advanced usage: automated parameter selection through multi-objective optimization using a user-defined cutoff frequency
```bash
        # cutoff_freq: estimate by (a) counting the number of true peaks per second in the data or (b) look at power spectra and choose cutoff
        log_gamma = -1.6*np.log(cutoff_frequency) -0.71*np.log(dt) - 5.1 # see: https://ieeexplore.ieee.org/abstract/document/9241009
        tvgamma = np.exp(log_gamma) 

        params, val = pynumdiff.optimize.sub_module.method(x, dt, params=None, 
                                                           tvgamma=tvgamma, # hyperparameter
                                                           dxdt_truth=None, # no ground truth data
                                                           options={})
        print('Optimal parameters: ', params)
        x_hat, dxdt_hat = pynumdiff.sub_module.method(x, dt, params, options={'smooth': True})`
```

### Notebook examples

We will frequently update simple examples for demo purposes, and here are currently exisiting ones:
* Differentiation with different methods: [1_basic_tutorial.ipynb](examples/1_basic_tutorial.ipynb)
* Parameter Optimization with known ground truth (only for demonstration purpose):  [2a_optimizing_parameters_with_dxdt_known.ipynb](examples/2a_optimizing_parameters_with_dxdt_known.ipynb)
* Parameter Optimization with unknown ground truth:  [2b_optimizing_parameters_with_dxdt_unknown.ipynb](./examples/2b_optimizing_parameters_with_dxdt_unknown.ipynb)


### Important notes

* Larger values of `tvgamma` produce smoother derivatives
* The value of `tvgamma` is largely universal across methods, making it easy to compare method results
* The optimization is not fast. Run it on subsets of your data if you have a lot of data. It will also be much faster with faster differentiation methods, like savgoldiff and butterdiff, and probably too slow for sliding methods like sliding DMD and sliding LTI fit. 
* The following heuristic works well for choosing `tvgamma`, where `cutoff_frequency` is the highest frequency content of the signal in your data, and `dt` is the timestep: `tvgamma=np.exp(-1.6*np.log(cutoff_frequency)-0.71*np.log(dt)-5.1)`


### Running the tests

We are using Travis CI for continuous intergration testing. You can check out the current status 
[here](https://travis-ci.com/github/florisvb/PyNumDiff).

To run tests locally, type:
```bash
> pytest pynumdiff
```


## License

This project utilizes the [MIT LICENSE](LICENSE.txt).
100% open-source, feel free to utilize the code however you like. 
