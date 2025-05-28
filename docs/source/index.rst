PyNumDiff
=========

Python methods for numerical differentiation of noisy data, including
multi-objective optimization routines for automated parameter selection.

Table of contents
-----------------

-  `Introduction <#introduction>`__
-  `Structure <#structure>`__
-  `Getting Started <#getting-started>`__

   -  `Prerequisite <#prerequisite>`__
   -  `Installing <#installing>`__

-  `Usage <#usage>`__

   -  `Basic usages <#basic-usages>`__
   -  `Notebook examples <#notebook-examples>`__
   -  `Important notes <#important-notes>`__
   -  `Running the tests <#running-the-tests>`__

-  `Citation <#citation>`__
-  `License <#license>`__

Introduction
------------

PyNumDiff is a Python package that implements various methods for
computing numerical derivatives of noisy data, which can be a critical
step in developing dynamic models or designing control. There are four
different families of methods implemented in this repository: smoothing
followed by finite difference calculation, local approximation with
linear models, Kalman filtering based methods and total variation
regularization methods. Most of these methods have multiple parameters
involved to tune. We take a principled approach and propose a
multi-objective optimization framework for choosing parameters that
minimize a loss function to balance the faithfulness and smoothness of
the derivative estimate. For more details, refer to `this
paper <https://doi.org/10.1109/ACCESS.2020.3034077>`__.

Structure
---------

::

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
      |- setup.py
      |- .gitignore
      |- LICENSE.txt
      |- pyproject.toml

Getting Started
---------------

Prerequisite
~~~~~~~~~~~~

PyNumDiff requires common packages like ``numpy``, ``scipy``,
``matplotlib``, ``pytest`` (for unittests), ``pylint`` (for PEP8 style
check). For a full list, you can check the file
`pyproject.toml <https://github.com/florisvb/PyNumDiff/blob/master/pyproject.toml>`__

In addition, it also requires certain additional packages for select
functions, though these are not required for a successful install of
PyNumDiff: 

-  Total Variation Regularization methods: `cvxpy <http://www.cvxpy.org/install/index.html>`__ 

When using ``cvxpy``, our default solver is set to be ``MOSEK`` (highly
recommended), you would need to download their free academic license
from their
`website <https://www.mosek.com/products/academic-licenses/>`__.
Otherwise, you can also use other solvers which are listed
`here <https://www.cvxpy.org/tutorial/advanced/index.html>`__.

Installing
~~~~~~~~~~

The code is compatible with >=Python 3.5. It can be installed using pip
or directly from the source code. Basic installation options include:

-  From PyPI using pip: ``pip install pynumdiff``.
-  From source using pip git+:
   ``pip install git+https://github.com/florisvb/PyNumDiff``
-  From local source code: Run ``pip install .`` from inside this directory.

For additional solvers, run ``pip install pynumdiff[advanced]``.  This includes ``cvxpy``,
which can be tricky when compiling from source.  If an error occurs in installing
``cvxpy``, see `cvxpy install documentation <https://www.cvxpy.org/install/>`__, install
``cvxpy`` according to those instructions, and try ``pip install pynumdiff[advanced]``
again.


Note: If using the optional MOSEK solver for cvxpy you will also need a
`MOSEK license <https://www.mosek.com/products/academic-licenses/>`__,
free academic license.

Usage
-----

Basic usages
~~~~~~~~~~~~

-  Basic Usage: you provide the parameters

   .. code:: bash

           x_hat, dxdt_hat = pynumdiff.sub_module.method(x, dt, params, options)     

-  Advanced usage: automated parameter selection through multi-objective
   optimization

   .. code:: bash

           params, val = pynumdiff.optimize.sub_module.method(x, dt, params=None, 
                                                              tvgamma=tvgamma, # hyperparameter
                                                              dxdt_truth=None, # no ground truth data
                                                              options={})
           print('Optimal parameters: ', params)
           x_hat, dxdt_hat = pynumdiff.sub_module.method(x, dt, params, options={'smooth': True})`

Notebook examples
~~~~~~~~~~~~~~~~~

-  Differentiation with different methods: `1\_basic\_tutorial.ipynb <https://github.com/florisvb/PyNumDiff/tree/master/examples/1_basic_tutorial.ipynb>`__ 
-  Parameter Optimization with known ground truth (only for demonstration purpose): `2a\_optimizing\_parameters\_with\_dxdt\_known.ipynb <https://github.com/florisvb/PyNumDiff/tree/master/examples/2a_optimizing_parameters_with_dxdt_known.ipynb>`__
-  Parameter Optimization with unknown ground truth: `2b\_optimizing\_parameters\_with\_dxdt\_unknown.ipynb <https://github.com/florisvb/PyNumDiff/tree/master/examples/2b_optimizing_parameters_with_dxdt_unknown.ipynb>`__


Important notes
~~~~~~~~~~~~~~~

-  Larger values of ``tvgamma`` produce smoother derivatives
-  The value of ``tvgamma`` is largely universal across methods, making
   it easy to compare method results
-  The optimization is not fast. Run it on subsets of your data if you
   have a lot of data. It will also be much faster with faster
   differentiation methods, like savgoldiff and butterdiff, and probably
   too slow for sliding methods like sliding DMD and sliding LTI fit.
-  The following heuristic works well for choosing ``tvgamma``, where
   ``cutoff_frequency`` is the highest frequency content of the signal
   in your data, and ``dt`` is the timestep:
   ``tvgamma=np.exp(-1.6*np.log(cutoff_frequency)-0.71*np.log(dt)-5.1)``

Running the tests
~~~~~~~~~~~~~~~~~

To run tests locally, type:

.. code:: bash

    > pytest pynumdiff

Citation
--------

@ARTICLE{9241009, author={F. {van Breugel} and J. {Nathan Kutz} and B.
W. {Brunton}}, journal={IEEE Access}, title={Numerical differentiation
of noisy data: A unifying multi-objective optimization framework},
year={2020}, volume={}, number={}, pages={1-1},
doi={10.1109/ACCESS.2020.3034077}}

Developer's Guide
-----------------

.. toctree::
   :maxdepth: 1

   code
   contact
   contributing
   LICENSE