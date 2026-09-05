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
    <!--a href="https://doi.org/10.5281/zenodo.6374098">
        <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.6374098.svg" alt="DOI"></a-->
    <a href="https://joss.theoj.org/papers/102257ee4b0142bf49bc18d7c810e9d5">
        <img src="https://joss.theoj.org/papers/102257ee4b0142bf49bc18d7c810e9d5/status.svg"></a>
</p>

## Introduction

PyNumDiff is a Python package that implements many methods for computing numerical derivatives and smooth estimates from noisy data, which can be a critical step in developing dynamic models or designing control. There are seven different families of methods in this repository:

1. prefiltering followed by finite difference calculation
2. iterated finite differencing
3. polynomial fit methods
4. basis function fit methods
5. total variation regularization of a finite difference derivative
6. generalized Kalman smoothing
7. local approximation with linear model

All are ultimately smoothing with similar runtime and accuracy, but some have flexibility advantages over others, summarized in the table under [Usage](#usage) below. For further details and comparison, see section 7 of our [Taxonomy Paper](https://arxiv.org/abs/2512.09090).

All methods have hyperparameters, described in the [Sphinx documentation](https://pynumdiff.readthedocs.io/master/). We take a principled approach and propose a multi-objective optimization framework for choosing settings that minimize a loss function that balances faithfulness to data with smoothness of the derivative estimate. For more details, refer to [this paper](https://doi.org/10.1109/ACCESS.2020.3034077).

![Three simulated signals and their derivatives, estimated by six of the seven method families, with hyperparameters chosen by `pynumdiff.optimize`.](paper/methods_comparison.png)

Above, three simulated signals are differentiated by two methods each, drawn from six of the seven families, with hyperparameters chosen automatically by `pynumdiff.optimize`. Reproduce with `python paper/make_figure.py`.

## Installing

Dependencies are listed in [pyproject.toml](https://github.com/florisvb/PyNumDiff/blob/master/pyproject.toml). They include the usual suspects like `numpy` and `scipy`, plus `pywavelets` for `waveletdiff`, `tqdm` for the optimizer, and `cvxpy` for `robustdiff` and `tvrdiff`.

The code is compatible with >=Python 3.11. Install from PyPI with `pip install pynumdiff`, from source with `pip install git+https://github.com/florisvb/PyNumDiff`, or from local download with `pip install .`. Call `pip install pynumdiff[advanced]` to automatically install optional dependencies from the advanced list, like [CVXPY](https://www.cvxpy.org).

## Usage

For more details, read our [Sphinx documentation](https://pynumdiff.readthedocs.io/master/). The basic pattern of all differentiation methods is:
```python
x_hat, dxdt_hat = somethingdiff(x, dt, **kwargs)
```
where `x` is data, `dt` is a step size, and various keyword arguments control the behavior. Methods marked multidimensional take an `axis` argument selecting which dimension of a block to differentiate along, and those supporting variable step size rename the second parameter `dt_or_t`, which accepts either a constant step size or an array of sample locations. Handing a method data it doesn't support raises a `ValueError` explaining why.

| Method | Multidim | Variable step | Missing data | Outliers | Circular domain | Needs CVXPY |
| --- | :-: | :-: | :-: | :-: | :-: | :-: |
| `kerneldiff` | ✓ | | | | | |
| `butterdiff` | ✓ | | | | | |
| `finitediff` | ✓ | | | | | |
| `polydiff` | ✓ | ✓ | ✓ | | | |
| `savgoldiff` | ✓ | | | | | |
| `splinediff` | ✓ | ✓ | ✓ | | | |
| `spectraldiff` | ✓ | | | | | |
| `rbfdiff` | ✓ | ✓ | | | | |
| `waveletdiff` | ✓ | | | | | |
| `tvrdiff` | ✓ | | | ✓ | | ✓ |
| `rtsdiff` | ✓ | ✓ | ✓ | | ✓ | |
| `robustdiff` | ✓ | ✓ | ✓ | ✓ | | ✓ |
| `lineardiff` | ✓ | ✓ | ✓ | | | ✓ |

There are also a couple minor methods kept for general interest (`iterative_velocity` and `smooth_acceleration`) but in practice dominated by or redundant with others from the table.

You can set the hyperparameters manually, or you can find hyperparameter settings by calling the multi-objective optimization algorithm from the `optimize` module:
```python
from pynumdiff.optimize import optimize

# estimate bandlimit by (a) counting the number of true peaks per second in the data or (b) look at the power spectrum
params, val = optimize(somethingdiff, x, dt, bandlimit=bandlimit, # smoothness hyper-parameter which defaults to None if dxdt_truth given
            dxdt_truth=None, # give ground truth data if available, in which case bandlimit goes unused
            search_space_updates={'param1':[vals], 'param2':[vals], ...})

print('Optimal parameters: ', params)
x_hat, dxdt_hat = somethingdiff(x, dt, **params)
```
`bandlimit` governs the smoothness targeted by the optimization procedure, with smaller values yielding smoother derivatives. Its value is dependent upon frequency content of the underlying signal, and it is universal across methods, making it possible to compare results post optimization. A default search space is used to initialize and perform optimization, defined at the top of `optimize.py`, with overwrites from `search_space_updates`. Be aware the optimization is a fairly heavy process.

### Notebook examples

Much more extensive usage is demonstrated in Jupyter notebooks, described further in the README in the `notebooks/` folder:
* [1_basic_tutorial.ipynb](https://github.com/florisvb/PyNumDiff/blob/master/notebooks/1_basic_tutorial.ipynb) invokes all the major methods on 1D data.
* [2_optimizing_hyperparameters.ipynb](https://github.com/florisvb/PyNumDiff/blob/master/notebooks/2_optimizing_hyperparameters.ipynb) covers the metrics worth optimizing and how to use `optimize` to choose hyperparameters.
* [3_automatic_method_suggestion.ipynb](https://github.com/florisvb/PyNumDiff/blob/master/notebooks/3_automatic_method_suggestion.ipynb) lets `pynumdiff` pick a method for your data.
* [4_performance_analysis.ipynb](https://github.com/florisvb/PyNumDiff/blob/master/notebooks/4_performance_analysis.ipynb) compares methods' accuracy and bias across simulations.
* [5_robust_outliers_demo.ipynb](https://github.com/florisvb/PyNumDiff/blob/master/notebooks/5_robust_outliers_demo.ipynb) puts `rtsdiff` and `robustdiff` head to head on data with outliers.
* [6_multidimensionality_demo.ipynb](https://github.com/florisvb/PyNumDiff/blob/master/notebooks/6_multidimensionality_demo.ipynb) differentiates multidimensional data along particular axes.
* [7_circular_domain.ipynb](https://github.com/florisvb/PyNumDiff/blob/master/notebooks/7_circular_domain.ipynb) handles data on a wrapped domain, like angles, with `rtsdiff`.

## Repo Structure

- `.github/workflows` contains `.yaml` that configures our GitHub Actions continuous integration (CI) runs.
- `docs/` contains `make` files and `.rst` files to govern the way `sphinx` builds documentation, either locally by navigating to this folder and calling `make html` or in the cloud by `readthedocs.io`.
- `notebooks/` contains Jupyter notebooks that demonstrate some usage of the library.
- `pynumdiff/` contains the source code. For a full list of modules and further navigation help, see the readme in this subfolder.
- `.coveragerc` governs `coverage` runs, listing files and functions/lines that should be excluded, e.g. plotting code.
- `.editorconfig` ensures tabs are displayed as 4 characters wide.
- `.gitignore` ensures files generated by local `pip install`s, Jupyter notebook runs, caches from code runs, virtual environments, and more are not picked up by `git` and accidentally added to the repo.
- `.pylintrc` configures `pylint`, a tool for autochecking code quality.
- `.readthedocs.yaml` configures `readthedocs` and is necessary for documentation to get auto-rebuilt.
- `CITATION.cff` is citation information for the Journal of Open-Source Software (JOSS) paper associated with this project.
- `LICENSE.txt` allows free usage of this project.
- `README.md` is the text you're reading, hello.
- `pyproject.toml` governs how this package is set up and installed, including dependencies.

## Citation

See CITATION.cff file, but here are some possible BibTeX entries for convenience.

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

### Collection of numerical differentiation methods:

    @misc{komarov2025taxonomynumericaldifferentiationmethods,
      title={A Taxonomy of Numerical Differentiation Methods},
      author={Pavel Komarov and Floris van Breugel and J. Nathan Kutz},
      year={2025},
      eprint={2512.09090},
      archivePrefix={arXiv},
      primaryClass={math.NA},
      url={https://arxiv.org/abs/2512.09090}
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

We are using GitHub Actions for continuous integration testing.

Run tests locally by navigating to the repo in a terminal and calling
```bash
> pytest -s
```
Add the flag `--plot` to see plots of the methods against test functions. Add the flag `--bounds` to print $\log$ error bounds (useful when changing method behavior).

## License

This project utilizes the [MIT LICENSE](https://github.com/florisvb/PyNumDiff/blob/master/LICENSE.txt).
100% open-source, feel free to utilize the code however you like. 
