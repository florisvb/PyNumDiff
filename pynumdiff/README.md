- `basis_fit` contains methods based on fitting basis functions, including a variant of the classic Fourier-spectral method.
- `finite_difference` contains a method to do iterative 1st, 2nd or 4th order finite differencing.
- `kalman_smooth` contains classic linear Kalman filter and RTS smoothing code, as well as a constant-derivative naive-model smoothing method.
- `linear_model` contains a method that tries to fit a linear model on a sliding window.
- `optimize` contains code to find best parameter settings for methods, tuned using Nelder-Mead according to the paper "Numerical differentiation of noisy data: A unifying multi-objective optimization framework", as well as a method suggestion metamethod.
- `polynomial_fit` contains methods which explicitly or implicitly fit polynomials over pieces of the data.
- `smooth_finite_difference` contains methods which do a smoothing step followed by simple finite difference.
- `tests` contains `pytest` unit tests of
	1. all the differentiation methods, checking their results against a suite of known analytic functions (including an ability to plot if the `--plot` command is passed to `pytest`, see `conftest.py`)
	2. the optimizer
	3. utilities, auxiliary functions used throughout the code
- `total_variation_regularization` contains code to take the derivative based on a finite differencing scheme which is regularized by shrinking changes of value in some derivative (1st, 2nd, or 3rd order)
- `utils` contains `utility` functions used throughout differentation methods, `evaluate` functions used by the parameter optimizer, and `simulate` examples for demonstrating and testing the methods.
