Welcome to PyNumDiff's documentation!
=====================================


[Py]thon [Num]erical [Diff]erentiation for noisy time series measurements.

Description
-----------
Methods for numerical differentiation of noisy time series data,
including multi-objective optimization routines for automated parameter selection.
blablabla ...


Installation
------------
Some descriptions about the prerequisites. blablabla ...

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

- more coming soon ...

References
----------

- F. van Breugel, J. Nathan Kutz and B. W. Brunton, "Numerical differentiation of noisy data: A unifying multi-objective optimization framework," in IEEE Access, doi: 10.1109/ACCESS.2020.3034077.