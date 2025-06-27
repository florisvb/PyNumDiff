- `finite_difference` contains basic first and second order finite differencing methods. The first order method supports iterative application.
- `kalman_smooth` contains Kalman filtering and smoothing methods, currently constant-derivative methods up to 3rd order (jerk) and a classic linear Kalman Filter based on known dynamics.
- `linear_model` is a bit of a miscellaneous module, containing methods which work linearly: `lineardiff`, `polydiff`, `savgoldiff`, and `spectraldiff`.
- `optimize` contains code to find best parameter settings for methods, tuned using Nelder-Mead according to the paper "Numerical differentiation of noisy data: A unifying multi-objective optimization framework"
- `smooth_finite_difference` contains methods which do a smoothing step followed by simple finite difference.
- `tests` contains `pytest` unit tests of
	1. all the differentiation methods, checking their results against a suite of known analytic functions (including an ability to plot if the `--plot` command is passed to `pytest`, see `conftest.py`)
	2. the optimizer
	3. utilities, auxiliary functions used throughout the code
- `total_variation_regularization` contains code to take the derivative based on a finite differencing scheme which is regularized by shrinking changes of value in some derivative (1st, 2nd, or 3rd order)
- `utils` contains `utility` functions used throughout differentation methods, `evaluate` functions used by the parameter optimizer, and `simulate` examples for demonstrating and testing the methods.
