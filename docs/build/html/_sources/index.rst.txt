Welcome to PyNumDiff's documentation!
=====================================


[Py]thon [Num]erical [Diff]erentiation for noisy time series measurements.

Description
-----------
PyNumDiff is a Python package that implements various methods for computing numerical derivatives of noisy data, which
can be a critical step in developing dynamic models or designing control. There are four different families of methods
implemented in this repository: smoothing followed by finite difference calculation, local approximation with linear
models, Kalman filtering based methods and total variation regularization methods. Most of these methods have multiple
parameters involved to tune. We take a principled approach and propose a multi-objective optimization framework for
choosing parameters that minimize a loss function to balance the faithfulness and smoothness of the derivative estimate.
For more details, refer to `this paper`_.


Installation
------------
Prerequisites
^^^^^^^^^^^^^
PyNumDiff requires common packages like `numpy`, `scipy`, `matplotlib`, `pytest` (for unittests), `pylint`
(for PEP8 style check). For a full list, you can check the file `requirements.txt`_

In addition, it also requires certain additional packages:

- Total Variation Regularization methods: `cvxpy`_
- Linear Model Chebychev: `pychebfun`_

When using `cvxpy`, our default solver is set to be `MOSEK` (highly recommended), you would need to download their
free academic license from their website_. Otherwise, you can also
use other solvers which are listed here_.

The code is compatible with Python 3.x. It can be installed using pip or directly from the source code.


Installing via PIP
^^^^^^^^^^^^^^^^^^
Mac and Linux users can install pre-built binary packages using pip.
To install the package just type:
::

    pip install pynumdiff

Installing from source
^^^^^^^^^^^^^^^^^^^^^^
The official distribution is on GitHub, and you can clone the repository using
::

    git clone https://github.com/luckystarufo/PyNumDiff.git

To install the package just type:
::

    python setup.py install



Developer's Guide
-----------------

.. toctree::
   :maxdepth: 1

   code
   contact
   contributing
   LICENSE

Tutorials
---------
Basic Usages
^^^^^^^^^^^^
* Basic Usage: you provide the parameters

::

   x_hat, dxdt_hat = pynumdiff.sub_module.method(x, dt, params, options)

* For example, a few favorites:

::

    x_hat, dxdt_hat = pynumdiff.linear_model.savgoldiff(x, dt, [3, 20, 25])
    x_hat, dxdt_hat = pynumdiff.kalman_smooth.constant_acceleration(x, dt, [1e-1, 1e-2])
    x_hat, dxdt_hat = pynumdiff.total_variation_regularization.jerk(x, dt, [10])
    x_hat, dxdt_hat = pynumdiff.smooth_finite_difference.butterdiff(x, dt, [3, 0.07])

Examples
^^^^^^^^
We made some tutorial examples. Please refer to the official GitHub repository for the last
updates in the examples folder. Here the list of the exported tutorials:

- `1_basic_tutorial.ipynb`_
- `2a_optimizing_parameters_with_dxdt_known.ipynb`_
- `2b_optimizing_parameters_with_dxdt_unknown.ipynb`_

References
----------

- F. van Breugel, J. Nathan Kutz and B. W. Brunton, "Numerical differentiation of noisy data: A unifying multi-objective optimization framework," in IEEE Access, doi: 10.1109/ACCESS.2020.3034077.

.. _this paper: https://doi.org/10.1109/ACCESS.2020.3034077
.. _website: https://www.mosek.com/products/academic-licenses/
.. _cvxpy: http://www.cvxpy.org/install/index.html
.. _pychebfun: https://github.com/pychebfun/pychebfun/
.. _here: https://www.cvxpy.org/tutorial/advanced/index.html
.. _requirements.txt: https://github.com/luckystarufo/PyNumDiff/blob/upgrade/requirements.txt
.. _1_basic_tutorial.ipynb: https://github.com/luckystarufo/PyNumDiff/blob/upgrade/examples/1_basic_tutorial.ipynb
.. _2a_optimizing_parameters_with_dxdt_known.ipynb: https://github.com/luckystarufo/PyNumDiff/blob/upgrade/examples/2a_optimizing_parameters_with_dxdt_known.ipynb
.. _2b_optimizing_parameters_with_dxdt_unknown.ipynb: https://github.com/luckystarufo/PyNumDiff/blob/upgrade/examples/2b_optimizing_parameters_with_dxdt_unknown.ipynb