# PyNumDiff

Python methods for numerical differentiation of noisy data, including multi-objective optimization routines for automated parameter selection.

<p align="center">
  <a href="https://pynumdiff.readthedocs.io/master/">
    <img alt="Python for Numerical Differentiation of noisy time series data" src="https://raw.githubusercontent.com/florisvb/PyNumDiff/master/logo.png" width="300" height="200" />
  </a>
</p>

<p align="center">
    <img src='https://github.com/florisvb/pynumdiff/actions/workflows/test.yml/badge.svg'/>
    <a href='https://pynumdiff.readthedocs.io/master/'>
        <img src='https://app.readthedocs.org/projects/pynumdiff/badge/?version=master' alt='Documentation Status' /></a>
    <a href='https://coveralls.io/github/florisvb/PyNumDiff?branch=master'>
        <img src='https://coveralls.io/repos/github/florisvb/PyNumDiff/badge.svg?branch=master' alt='Coverage Status' /></a>
    <a href="https://badge.fury.io/py/pynumdiff">
        <img src="https://badge.fury.io/py/pynumdiff.svg" alt="PyPI"></a>
    <a href="https://zenodo.org/badge/latestdoi/159711175">
        <img src="https://zenodo.org/badge/159711175.svg" alt="DOI"></a>
    <a href="https://joss.theoj.org/papers/102257ee4b0142bf49bc18d7c810e9d5">
        <img src="https://joss.theoj.org/papers/102257ee4b0142bf49bc18d7c810e9d5/status.svg"></a>
</p>

## Introduction

PyNumDiff is a Python package that implements various methods for computing numerical derivatives of noisy data, which can be a critical step in developing dynamic models or designing control. There are seven different families of methods implemented in this repository:

1. convolutional smoothing followed by finite difference calculation
2. polynomial-fit-based methods
3. iterated finite differencing
4. total variation regularization of a finite difference derivative
5. Kalman (RTS) smoothing
6. basis-function-based methods
7. linear local approximation with linear model

Most of these methods have multiple parameters, so we take a principled approach and propose a multi-objective optimization framework for choosing parameters that minimize a loss function to balance the faithfulness and smoothness of the derivative estimate. For more details, refer to [this paper](https://doi.org/10.1109/ACCESS.2020.3034077).

## Installing

Dependencies are listed in [pyproject.toml](https://github.com/florisvb/PyNumDiff/blob/master/pyproject.toml). They include the usual suspects like `numpy` and `scipy`, but also optionally `cvxpy`.

The code is compatible with >=Python 3.10. Install from PyPI with `pip install pynumdiff`, from source with `pip install git+https://github.com/florisvb/PyNumDiff`, or from local download with `pip install .`. Call `pip install pynumdiff[advanced]` to automatically install optional dependencies from the advanced list, like [CVXPY](https://www.cvxpy.org).

## Usage

For more details, read our [Sphinx documentation](https://pynumdiff.readthedocs.io/master/). The basic pattern of all differentiation methods is:

```python
somethingdiff(x, dt, **kwargs)
```

where `x` is data, `dt` is a step size, and various keyword arguments control the behavior. Some methods support variable step size, in which case the second parameter is renamed `_t` and can receive either a constant step size or an array of values to denote sample locations.

You can provide the parameters:
```python
from pynumdiff.submodule import method

x_hat, dxdt_hat = method(x, dt, param1=val1, param2=val2, ...)     
```

Or you can find parameter by calling the multi-objective optimization algorithm from the `optimize` module:
```python
from pynumdiff.optimize import optimize

# estimate cutoff_frequency by (a) counting the number of true peaks per second in the data or (b) look at power spectra and choose cutoff
tvgamma = np.exp(-1.6*np.log(cutoff_frequency) -0.71*np.log(dt) - 5.1) # see https://ieeexplore.ieee.org/abstract/document/9241009

params, val = optimize(somethingdiff, x, dt, tvgamma=tvgamma, # smoothness hyperparameter which defaults to None if dxdt_truth given
            dxdt_truth=None, # give ground truth data if available, in which case tvgamma goes unused
            search_space_updates={'param1':[vals], 'param2':[vals], ...})

print('Optimal parameters: ', params)
x_hat, dxdt_hat = somethingdiff(x, dt, **params)
```
If no `search_space_updates` is given, a default search space is used. See the top of `_optimize.py`.

The following heuristic works well for choosing `tvgamma`, where `cutoff_frequency` is the highest frequency content of the signal in your data, and `dt` is the timestep: `tvgamma=np.exp(-1.6*np.log(cutoff_frequency)-0.71*np.log(dt)-5.1)`. Larger values of `tvgamma` produce smoother derivatives. The value of `tvgamma` is largely universal across methods, making it easy to compare method results. Be aware the optimization is a fairly heavy process.

### Notebook examples

Much more extensive usage is demonstrated in Jupyter notebooks:
* Differentiation with different methods: [1_basic_tutorial.ipynb](https://github.com/florisvb/PyNumDiff/blob/master/examples/1_basic_tutorial.ipynb)
* Parameter Optimization with known ground truth (only for demonstration purpose):  [2a_optimizing_parameters_with_dxdt_known.ipynb](https://github.com/florisvb/PyNumDiff/blob/master/examples/2a_optimizing_parameters_with_dxdt_known.ipynb)
* Parameter Optimization with unknown ground truth: [2b_optimizing_parameters_with_dxdt_unknown.ipynb](https://github.com/florisvb/PyNumDiff/blob/master/examples/2b_optimizing_parameters_with_dxdt_unknown.ipynb)
* Automatic method suggestion: [3_automatic_method_suggestion.ipynb](https://github.com/florisvb/PyNumDiff/blob/master/examples/3_automatic_method_suggestion.ipynb)

## Repo Structure

- `.github/workflows` contains `.yaml` that configures our GitHub Actions continuous integration (CI) runs.
- `docs/` contains `make` files and `.rst` files to govern the way `sphinx` builds documentation, either locally by navigating to this folder and calling `make html` or in the cloud by `readthedocs.io`.
- `examples/` contains Jupyter notebooks that demonstrate some usage of the library.
- `pynumdiff/` contains the source code. For a full list of modules and further navigation help, see the readme in this subfolder.
- `.editorconfig` ensures tabs are displayed as 4 characters wide.
- `.gitignore` ensures files generated by local `pip install`s, Jupyter notebook runs, caches from code runs, virtual environments, and more are not picked up by `git` and accidentally added to the repo.
- `.pylintrc` configures `pylint`, a tool for autochecking code quality.
- `.readthedocs.yaml` configures `readthedocs` and is necessary for documentation to get auto-rebuilt.
- `CITATION.cff` is citation information for the Journal of Open-Source Software (JOSS) paper associated with this project.
- `LICENSE.txt` allows free usage of this project.
- `README.md` is the text you're reading, hello.
- `linting.py` is a script to run `pylint`.
- `pyproject.toml` governs how this package is set up and installed, including dependencies.

## Citation

See CITATION.cff file as well as the following references.

### PyNumDiff python package:

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

### Optimization algorithm:

    @article{ParamOptimizationDerivatives2020, 
    doi={10.1109/ACCESS.2020.3034077}
    author={F. {van Breugel} and J. {Nathan Kutz} and B. W. {Brunton}}, 
    journal={IEEE Access}, 
    title={Numerical differentiation of noisy data: A unifying multi-objective optimization framework}, 
    year={2020}
    }

## Running the tests

We are using GitHub Actions for continuous intergration testing.

Run tests locally by navigating to the repo in a terminal and calling
```bash
> pytest -s
```

Add the flag `--plot` to see plots of the methods against test functions. Add the flag `--bounds` to print $\log$ error bounds (useful when changing method behavior).

## License

This project utilizes the [MIT LICENSE](https://github.com/florisvb/PyNumDiff/blob/master/LICENSE.txt).
100% open-source, feel free to utilize the code however you like. 
