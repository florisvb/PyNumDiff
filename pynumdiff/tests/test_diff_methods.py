"""Unit tests of core differentiation functionality"""
import numpy as np
from pytest import mark

from ..smooth_finite_difference import kerneldiff, butterdiff
from ..finite_difference import finitediff
from ..polynomial_fit import polydiff, savgoldiff, splinediff
from ..basis_fit import spectraldiff, rbfdiff, waveletdiff
from ..total_variation_regularization import iterative_velocity, smooth_acceleration, tvrdiff
from ..kalman_smooth import rtsdiff, robustdiff
from ..linear_model import lineardiff
# Function aliases for testing cases where parameters change the behavior in a big way, so error limits can be indexed in dict
def kerneldiff_uniform(*args, **kwargs): return kerneldiff(*args, **kwargs)
def kerneldiff_median(*args, **kwargs): return kerneldiff(*args, **kwargs)
def kerneldiff_gaussian(*args, **kwargs): return kerneldiff(*args, **kwargs)
def finitediff_1(*args, **kwargs): return finitediff(*args, **kwargs)
def finitediff_4(*args, **kwargs): return finitediff(*args, **kwargs)
def iterated_finitediff(*args, **kwargs): return finitediff(*args, **kwargs)
def iterated_finitediff_4(*args, **kwargs): return finitediff(*args, **kwargs)
def tvrdiff_1(*args, **kwargs): return tvrdiff(*args, **kwargs)
def tvrdiff_3(*args, **kwargs): return tvrdiff(*args, **kwargs)
def spline_irreg_step(*args, **kwargs): return splinediff(*args, **kwargs)
def robust_irreg_step(*args, **kwargs): return robustdiff(*args, **kwargs)
def polydiff_irreg_step(*args, **kwargs): return polydiff(*args, **kwargs)
def rtsdiff_irreg_step(*args, **kwargs): return rtsdiff(*args, **kwargs)
def lineardiff_irreg_step(*args, **kwargs): return lineardiff(*args, **kwargs)
irreg_list = [spline_irreg_step, polydiff_irreg_step, rbfdiff, rtsdiff_irreg_step, robust_irreg_step, lineardiff_irreg_step] # methods to test with irregular time steps

dt = 0.1
t = np.linspace(0, 3, 31) # sample locations, including the endpoint
tt = np.linspace(0, 3) # full domain, for visualizing denser plots
np.random.seed(7) # for repeatability of the test, so we don't get random failures
noise = 0.05*np.random.randn(*t.shape)
t_irreg = t[-1]*np.linspace(0, 1, len(t))**1.5 # ~8x denser at the start than the end. See #228

# Analytic (function, derivative) pairs on which to test differentiation methods.
test_funcs_and_derivs = [
    (0, r"$x(t)=1$",            lambda t: np.ones(t.shape), lambda t: np.zeros(t.shape)),   # constant
    (1, r"$x(t)=2t+1$",         lambda t: 2*t + 1,          lambda t: 2*np.ones(t.shape)),  # affine
    (2, r"$x(t)=t^2-t+1$",      lambda t: t**2 - t + 1,     lambda t: 2*t - 1),             # quadratic
    (3, r"$x(t)=\sin(3t)+1/2$", lambda t: np.sin(3*t) + 1/2, lambda t: 3*np.cos(3*t)),      # sinuoidal
    (4, r"$x(t)=e^t\sin(5t)$",  lambda t: np.exp(t)*np.sin(5*t),                            # growing sinusoidal
                                lambda t: np.exp(t)*(5*np.cos(5*t) + np.sin(5*t))),
    (5, r"$x(t)=\frac{\sin(8t)}{(t+0.1)^{3/2}}$", lambda t: np.sin(8*t)/((t + 0.1)**(3/2)), # steep challenger
                                lambda t: ((0.8 + 8*t)*np.cos(8*t) - 1.5*np.sin(8*t))/(0.1 + t)**(5/2))]

diff_methods_and_params = [
    (kerneldiff_uniform, {'kernel':'uniform', 'window_size':3, 'num_iterations':2}),
    (kerneldiff_median, {'kernel':'median', 'window_size':3, 'num_iterations':2}),
    (kerneldiff_gaussian, {'kernel':'gaussian', 'window_size':5}),
    (kerneldiff, {'window_size':5}), # 'friedrichs' is the default kernel
    (butterdiff, {'filter_order':3, 'cutoff_freq':0.7}),
    (finitediff_1, {'order':1}), (finitediff, {}), (finitediff_4, {'order':4}), # 2 is the default order
    (iterated_finitediff, {'num_iterations':5}), (iterated_finitediff_4, {'num_iterations':10, 'order':4}),
    (polydiff, {'degree':5, 'window_size':15}),
    (polydiff_irreg_step, {'degree':5, 'window_size':15}),
    (savgoldiff, {'degree':2, 'window_size':5, 'smoothing_win':5}),
    (splinediff, {'degree':5, 's':2}),
    (spline_irreg_step, {'degree':5, 's':2}),
    (spectraldiff, {'cutoff_freq':0.2}),
    (rbfdiff, {'sigma':0.5, 'lmbd':0.001}),
    (waveletdiff, {'wavelet':'db8', 'threshold':1.0}),
    (rtsdiff, {'order':2, 'log_qr_ratio':7, 'forwardbackward':True}),
    (rtsdiff_irreg_step, {'order':2, 'log_qr_ratio':7, 'forwardbackward':True}),
    (robustdiff, {'order':3, 'log_q':9, 'log_r':0}),
    (robust_irreg_step, {'order':3, 'log_q':9, 'log_r':0}),
    (tvrdiff_1, {'order':1, 'gamma':0.5}),
    (tvrdiff, {'order':2, 'gamma':1}),
    (tvrdiff_3, {'order':3, 'gamma':10}),
    (iterative_velocity, {'num_iterations':5, 'gamma':0.05}),
    (smooth_acceleration, {'gamma':2, 'window_size':5}),
    (lineardiff, {'order':3, 'gamma':0.003, 'window_size':11}),
    (lineardiff_irreg_step, {'order':3, 'gamma':0.003, 'window_size':11})
    ]

# All the testing methodology follows the exact same pattern; the only thing that changes is the closeness to the
# right answer various methods achieve with the given parameterizations and random seed. So index a big ol' table
# by method, then by test function number, and finally by the truth-result pair being compared. The tuples are order
# of magnitude of (L2,Linf) distances for pairs (x,x_hat), (dxdt,dxdt_hat), (x,x_hat_noisy), (dxdt,dxdt_hat_noisy).
flr = -12 # machine-precision floor: methods this accurate are exact up to round-off, with least-significant bits
          # drifting across BLAS/numpy/Python builds, so we don't assert tighter error.
error_bounds = {
    kerneldiff_uniform: [[(flr, flr), (flr, flr), (0, -1), (0, 0)],
                         [(0, 0), (1, 1), (0, 0), (1, 1)],
                         [(0, 0), (1, 1), (0, 0), (1, 1)],
                         [(0, 0), (1, 1), (0, 0), (1, 1)],
                         [(1, 1), (2, 2), (1, 1), (2, 2)],
                         [(1, 1), (3, 3), (1, 1), (3, 3)]],
    kerneldiff_median: [[(flr, flr), (flr, flr), (-1, -1), (0, 0)],
                        [(0, 0), (1, 1), (0, 0), (1, 1)],
                        [(0, 0), (1, 1), (0, 0), (1, 1)],
                        [(-1, -1), (0, 0), (0, 0), (1, 1)],
                        [(0, 0), (2, 2), (0, 0), (2, 2)],
                        [(1, 1), (3, 3), (1, 1), (3, 3)]],
    kerneldiff_gaussian: [[(flr, flr), (flr, flr), (0, -1), (1, 0)],
                          [(-1, -1), (1, 0), (0, 0), (1, 1)],
                          [(0, 0), (1, 1), (0, 0), (1, 1)],
                          [(0, -1), (1, 1), (0, 0), (1, 1)],
                          [(1, 1), (2, 2), (1, 1), (2, 2)],
                          [(1, 1), (3, 3), (1, 1), (3, 3)]],
    kerneldiff: [[(flr, flr), (flr, flr), (0, -1), (1, 0)],
                 [(-1, -1), (1, 0), (0, 0), (1, 1)],
                 [(0, 0), (1, 1), (0, 0), (1, 1)],
                 [(0, -1), (1, 1), (0, 0), (1, 1)],
                 [(1, 1), (2, 2), (1, 1), (2, 2)],
                 [(1, 1), (3, 3), (1, 1), (3, 3)]],
    butterdiff: [[(flr, flr), (flr, flr), (0, -1), (1, 1)],
                 [(-3, -3), (-2, -2), (0, -1), (1, 1)],
                 [(-2, -3), (-1, -1), (0, -1), (1, 1)],
                 [(-3, -3), (0, -1), (0, -1), (1, 1)],
                 [(0, -1), (1, 1), (0, 0), (1, 1)],
                 [(0, 0), (3, 3), (0, 0), (3, 3)]],
    finitediff_1: [[(flr, flr), (flr, flr), (0, 0), (1, 1)],
                   [(flr, flr), (flr, flr), (0, 0), (1, 1)],
                   [(flr, flr), (0, 0), (0, 0), (1, 1)],
                   [(flr, flr), (1, 0), (0, 0), (1, 1)],
                   [(flr, flr), (2, 2), (0, 0), (2, 2)],
                   [(flr, flr), (3, 3), (0, 0), (3, 3)]],
    finitediff: [[(flr, flr), (flr, flr), (0, 0), (1, 1)],
                 [(flr, flr), (flr, flr), (0, 0), (1, 1)],
                 [(flr, flr), (flr, flr), (0, 0), (1, 1)],
                 [(flr, flr), (0, -1), (0, 0), (1, 1)],
                 [(flr, flr), (1, 1), (0, 0), (1, 1)],
                 [(flr, flr), (3, 3), (0, 0), (3, 3)]],
    finitediff_4: [[(flr, flr), (flr, flr), (0, 0), (1, 1)],
                   [(flr, flr), (flr, flr), (0, 0), (1, 1)],
                   [(flr, flr), (flr, flr), (0, 0), (1, 1)],
                   [(flr, flr), (-2, -2), (0, 0), (1, 1)],
                   [(flr, flr), (1, 0), (0, 0), (1, 1)],
                   [(flr, flr), (2, 2), (0, 0), (2, 2)]],
    iterated_finitediff: [[(flr, flr), (flr, flr), (0, -1), (0, 0)],
                          [(flr, flr), (flr, flr), (0, -1), (0, 0)],
                          [(-1, -1), (0, 0), (0, -1), (0, 0)],
                          [(0, 0), (1, 0), (0, 0), (1, 0)],
                          [(1, 1), (2, 2), (1, 1), (2, 2)],
                          [(1, 1), (3, 3), (1, 1), (3, 3)]],
    iterated_finitediff_4: [[(flr, flr), (flr, flr), (0, -1), (0, 0)],
                            [(flr, flr), (flr, flr), (0, -1), (0, 0)],
                            [(-1, -1), (0, 0), (-1, -1), (0, 0)],
                            [(0, -1), (1, 1), (0, 0), (1, 1)],
                            [(1, 1), (2, 2), (1, 1), (2, 2)],
                            [(1, 1), (3, 3), (1, 1), (3, 3)]],
    polydiff: [[(flr, flr), (flr, flr), (0, -1), (1, 1)],
               [(flr, flr), (flr, flr), (0, -1), (1, 1)],
               [(flr, flr), (flr, flr), (0, -1), (1, 1)],
               [(-3, -3), (-1, -1), (0, -1), (1, 1)],
               [(0, -1), (1, 1), (0, 0), (1, 1)],
               [(0, 0), (2, 2), (0, 0), (3, 3)]],
    polydiff_irreg_step: [[(flr, flr), (flr, flr), (0, -1), (1, 1)],
                          [(flr, flr), (flr, flr), (0, -1), (1, 1)],
                          [(flr, flr), (flr, flr), (0, -1), (1, 1)],
                          [(-3, -3), (-1, -1), (0, -1), (1, 1)],
                          [(-1, -1), (1, 1), (0, -1), (1, 1)],
                          [(0, 0), (2, 2), (0, 0), (2, 2)]],
    savgoldiff: [[(flr, flr), (flr, flr), (0, -1), (0, 0)],
                 [(flr, flr), (flr, flr), (0, -1), (0, 0)],
                 [(-2, -2), (-1, -1), (0, -1), (0, 0)],
                 [(0, -1), (0, 0), (0, 0), (1, 0)],
                 [(1, 1), (2, 2), (1, 1), (2, 2)],
                 [(1, 1), (3, 3), (1, 1), (3, 3)]],
    splinediff: [[(flr, flr), (flr, flr), (-1, -1), (0, 0)],
                 [(flr, flr), (flr, flr), (-1, -1), (0, 0)],
                 [(flr, flr), (flr, flr), (-1, -1), (0, 0)],
                 [(0, -1), (1, 1), (0, 0), (1, 1)],
                 [(1, 1), (2, 2), (1, 1), (2, 2)],
                 [(0, 0), (3, 3), (1, 0), (3, 3)]],
    spline_irreg_step: [[(flr, flr), (flr, flr), (-1, -1), (0, 0)],
                        [(flr, flr), (flr, flr), (-1, -1), (0, 0)],
                        [(flr, flr), (flr, flr), (-1, -1), (0, 0)],
                        [(0, -1), (1, 1), (0, 0), (1, 1)],
                        [(1, 0), (2, 2), (1, 0), (2, 1)],
                        [(1, 0), (2, 2), (1, 0), (3, 2)]],
    spectraldiff: [[(flr, flr), (flr, flr), (0, -1), (0, 0)],
                   [(0, 0), (1, 1), (0, 0), (1, 1)],
                   [(1, 0), (1, 1), (1, 1), (1, 1)],
                   [(0, 0), (1, 1), (0, 0), (1, 1)],
                   [(1, 1), (2, 2), (1, 1), (2, 2)],
                   [(1, 1), (3, 3), (1, 1), (3, 3)]],
    rbfdiff: [[(-2, -2), (0, 0), (0, -1), (1, 0)],
              [(-1, -1), (1, 0), (0, -1), (1, 1)],
              [(-1, -1), (1, 1), (0, -1), (1, 1)],
              [(-2, -2), (0, 0), (0, -1), (1, 0)],
              [(0, 0), (2, 2), (0, 0), (2, 2)],
              [(1, 1), (3, 3), (1, 1), (3, 3)]],
    waveletdiff: [[(flr, flr), (flr, flr), (0, -1), (1, 0)],
                  [(-2, -2), (0, -1), (0, -1), (1, 1)],
                  [(-1, -2), (0, 0), (0, -1), (1, 1)],
                  [(-2, -2), (0, 0), (0, -1), (1, 1)],
                  [(-1, -1), (2, 2), (0, 0), (2, 2)],
                  [(-1, -1), (3, 3), (0, 0), (3, 3)]],
    tvrdiff_1: [[(flr, flr), (flr, flr), (0, -1), (1, 0)],
                [(flr, flr), (-11, flr), (-1, -1), (-1, -2)],
                [(0, -1), (1, 0), (0, -1), (1, 0)],
                [(0, -1), (1, 1), (0, 0), (1, 0)],
                [(1, 0), (2, 2), (1, 0), (2, 2)],
                [(0, 0), (3, 3), (0, 0), (3, 3)]],
    tvrdiff: [[(flr, flr), (flr, flr), (0, -1), (1, 0)],
              [(-10, -10), (-9, -9), (-1, -1), (0, -1)],
              [(-10, -10), (-9, -10), (-1, -1), (0, -1)],
              [(0, -1), (1, 0), (0, -1), (1, 0)],
              [(1, 0), (2, 2), (1, 0), (2, 2)],
              [(0, 0), (3, 3), (0, 0), (3, 3)]],
    tvrdiff_3: [[(flr, flr), (flr, flr), (-1, -1), (0, 0)],
                [(-9, -10), (-9, -9), (-1, -1), (0, 0)],
                [(-10, -10), (-9, -10), (-1, -1), (0, 0)],
                [(0, 0), (1, 1), (0, 0), (1, 1)],
                [(1, 1), (2, 2), (1, 1), (2, 2)],
                [(1, 1), (3, 3), (1, 1), (3, 3)]],
    iterative_velocity: [[(flr, flr), (flr, flr), (0, -1), (0, 0)],
                         [(0, 0), (0, 0), (0, 0), (1, 0)],
                         [(0, 0), (1, 0), (1, 0), (1, 0)],
                         [(1, 0), (1, 1), (1, 0), (1, 1)],
                         [(2, 1), (2, 2), (2, 1), (2, 2)],
                         [(1, 1), (3, 3), (1, 1), (3, 3)]],
    smooth_acceleration: [[(flr, flr), (flr, flr), (0, -1), (0, 0)],
                          [(-10, -11), (-10, -10), (-1, -1), (-1, -1)],
                          [(-2, -2), (-1, -1), (-1, -1), (0, -1)],
                          [(0, 0), (1, 0), (0, -1), (1, 0)],
                          [(1, 1), (2, 2), (1, 1), (2, 2)],
                          [(1, 1), (3, 3), (1, 1), (3, 3)]],
    rtsdiff: [[(flr, flr), (flr, flr), (0, -1), (1, 1)],
              [(-5, -5), (-4, -4), (0, -1), (1, 1)],
              [(-4, -5), (-3, -3), (0, -1), (1, 1)],
              [(-3, -3), (0, 0), (0, -1), (1, 1)],
              [(-1, -1), (1, 1), (0, -1), (1, 1)],
              [(0, 0), (3, 3), (0, 0), (3, 3)]],
    rtsdiff_irreg_step: [[(flr, flr), (flr, flr), (0, -1), (1, 1)],
                         [(-5, -5), (-4, -4), (0, -1), (1, 1)],
                         [(-4, -4), (-3, -3), (0, -1), (1, 1)],
                         [(-3, -3), (0, -1), (0, -1), (1, 1)],
                         [(-1, -1), (1, 1), (0, -1), (1, 1)],
                         [(0, 0), (3, 3), (0, 0), (3, 3)]],
    robustdiff: [[(flr, flr), (flr, flr), (0, -1), (1, 1)],
                 [(flr, flr), (flr, flr), (0, -1), (1, 1)],
                 [(flr, flr), (flr, flr), (0, -1), (1, 1)],
                 [(flr, flr), (-2, -2), (0, -1), (1, 1)],
                 [(-11, -11), (1, 1), (0, 0), (1, 1)],
                 [(0, 0), (3, 3), (0, 0), (3, 2)]],
    robust_irreg_step: [[(flr, flr), (flr, flr), (0, 0), (1, 1)],
                        [(flr, flr), (flr, flr), (0, 0), (1, 1)],
                        [(flr, flr), (flr, flr), (0, 0), (1, 1)],
                        [(flr, flr), (-1, -1), (0, 0), (1, 1)],
                        [(-10, -10), (1, 1), (0, 0), (1, 1)],
                        [(1, 1), (3, 3), (1, 1), (3, 3)]],
    lineardiff: [[(flr, flr), (flr, flr), (0, -1), (1, 1)],
                 [(-1, -1), (0, 0), (0, -1), (1, 0)],
                 [(-1, -1), (1, 1), (0, -1), (1, 1)],
                 [(-1, -1), (0, 0), (0, -1), (1, 1)],
                 [(0, 0), (2, 2), (0, 0), (2, 2)],
                 [(0, -1), (2, 2), (0, -1), (2, 2)]],
    lineardiff_irreg_step: [[(flr, flr), (flr, flr), (0, -1), (1, 1)],
                            [(-1, -1), (1, 0), (0, -1), (1, 0)],
                            [(-1, -1), (0, 0), (0, -1), (1, 1)],
                            [(-1, -2), (0, 0), (0, -1), (1, 0)],
                            [(0, 0), (2, 1), (0, 0), (2, 1)],
                            [(0, -1), (2, 2), (0, 0), (2, 2)]]
}


# Essentially run the cartesian product of [diff methods] x [test functions] through this one test
@mark.parametrize("diff_method_and_params", diff_methods_and_params) # things like splinediff, with their parameters
@mark.parametrize("test_func_and_deriv", test_funcs_and_derivs) # analytic functions, with their true derivatives
def test_diff_method(diff_method_and_params, test_func_and_deriv, request): # request gives access to context
    """Ensure differentiation methods find accurate derivatives"""
    # unpack
    diff_method, params = diff_method_and_params
    i, latex_name, f, df = test_func_and_deriv

    # sample the true function and true derivative, and make noisy samples
    x = f(t) if diff_method not in irreg_list else f(t_irreg)
    dxdt = df(t) if diff_method not in irreg_list else df(t_irreg)
    _t = dt if diff_method not in irreg_list else t_irreg
    x_noisy = x + noise

    # differentiate without and with noise
    x_hat, dxdt_hat = diff_method(x, _t, **params)
    x_hat_noisy, dxdt_hat_noisy = diff_method(x_noisy, _t, **params)

    # plotting code
    if request.config.getoption("--plot"): # Get the plot flag from pytest configuration
        fig, axes = request.config.plots[diff_method] # get the appropriate plot, set up by the store_plots fixture in conftest.py
        t_ = t_irreg if diff_method in irreg_list else t
        axes[i, 0].plot(t_, f(t_))
        axes[i, 0].plot(t_, x, 'C0+')
        axes[i, 0].plot(t_, x_hat, 'C2.', ms=4)
        axes[i, 0].plot(tt, df(tt))
        axes[i, 0].plot(t_, dxdt_hat, 'C1+')
        axes[i, 0].set_ylabel(latex_name, rotation=0, labelpad=50)
        if i < len(test_funcs_and_derivs)-1: axes[i, 0].set_xticklabels([])
        else: axes[i, 0].set_xlabel('t')
        if i == 0: axes[i, 0].set_title('noiseless')
        axes[i, 1].plot(t_, f(t_), label=r"$x(t)$")
        axes[i, 1].plot(t_, x_noisy, 'C0+', label=r"$x_n$")
        axes[i, 1].plot(t_, x_hat_noisy, 'C2.', ms=4, label=r"$\hat{x}_n$")
        axes[i, 1].plot(tt, df(tt), label=r"$\frac{dx(t)}{dt}$")
        axes[i, 1].plot(t_, dxdt_hat_noisy, 'C1+', label=r"$\hat{\frac{dx}{dt}}_n$")
        if i < len(test_funcs_and_derivs)-1: axes[i, 1].set_xticklabels([])
        else: axes[i, 1].set_xlabel('t')
        axes[i, 1].set_yticklabels([])
        if i == 0: axes[i, 1].set_title('with noise')

    # check x_hat and x_hat_noisy are close to known x and that dxdt_hat and dxdt_hat_noisy are close to known dxdt
    if request.config.getoption("--bounds"): print("\n[", end="") # print stuff if the user gave the --bounds flag
    for j,(a,b) in enumerate([(x,x_hat), (dxdt,dxdt_hat), (x,x_hat_noisy), (dxdt,dxdt_hat_noisy)]):
        l2_error = np.linalg.norm(a - b)
        linf_error = np.max(np.abs(a - b))

        if request.config.getoption("--bounds"): # bounds-printing for establishing bounds
            #print(f"({l2_error},{linf_error})", end=", ") # <- in case you want to print actual errors rather than powers
            print(f"({'flr' if np.ceil(np.log10(l2_error)) <= flr else int(np.ceil(np.log10(l2_error)))}, "
                  f"{'flr' if np.ceil(np.log10(linf_error)) <= flr else int(np.ceil(np.log10(linf_error)))})", end=", ")
        else: # bounds checking
            log_l2_bound, log_linf_bound = error_bounds[diff_method][i][j]
            assert l2_error < 10**log_l2_bound
            assert linf_error < 10**log_linf_bound
            # when a method beats its prior performance by an order of magnitude, signal the improvement
            if 10**flr < l2_error < 10**(log_l2_bound - 1) or 10**flr < linf_error < 10**(log_linf_bound - 1):
                print(f"Improvement detected for method {diff_method.__name__}; consider tightening its bound")

    # Differentiation is linear, so a*f(x) == f(a*x) exactly. A hyperparameter carrying absolute units can break this silently. See #222, #218, #220
    if diff_method is not iterative_velocity:
        for a in [1e-3, 1e3]: # both directions, so an absolute threshold can't pass by being tested only one way
            assert np.max(np.abs(diff_method(a*x_noisy, _t, **params)[1]/a - dxdt_hat_noisy))/np.max(np.abs(dxdt_hat_noisy)) < 1e-9


T1, T2 = np.meshgrid(np.linspace(-1, 0.98, 100), np.linspace(-1, 1, 101)) # a 101 x 100 grid, deliberately not square, so a method measuring a
# wrong dimension can't accidentally get an agreeing figure. Shorten the domain rather than point count, so both axes have same dt2 spacing.
dt2 = 0.02 # distance between samples in the 2D T grids
x = T1**2 * np.sin(3/2 * np.pi * T2) # 2D function


multidim_methods_and_params = [ # single parameterizations, chosen to do well on the noiseless 2D example
    (kerneldiff, {'kernel': 'gaussian', 'window_size': 5}),
    (butterdiff, {'filter_order': 3, 'cutoff_freq': 1 - 1e-6}),
    (finitediff, {}),
    (polydiff, {'degree': 2, 'window_size': 5}),
    (savgoldiff, {'degree': 3, 'window_size': 11, 'smoothing_win': 3}),
    (waveletdiff, {'wavelet': 'db8', 'threshold': 1.0}),
    (rtsdiff, {'order':2, 'log_qr_ratio':7, 'forwardbackward':True}),
    (spectraldiff, {'cutoff_freq': 0.25, 'pad_to_zero_dxdt': False}),
    (rbfdiff, {'sigma': 0.5, 'lmbd': 1e-6}),
    (splinediff, {'degree': 9, 's': 0}), # s is now relative to estimated noise, so 0 is how you ask to interpolate
    (robustdiff, {'order':2, 'log_q':9, 'log_r':0}),
    (tvrdiff, {'order': 3, 'gamma': 1e-4}),
    (lineardiff, {'order': 3, 'gamma': 1e-6, 'window_size': 41, 'stride': 41})
]

# Similar to the error_bounds table, index by method first. But then we test against only one 2D function, and only
# in the absence of noise, since the other test covers that. Instead, because multidim derivs can be combined in
# interesting ways, we find d^2 / dt_1 dt_2 and the Laplacian, d^2/dt_1^2 + d^2/dt_2^2. Tuples are again (L2,Linf).
multidim_error_bounds = {
    kerneldiff: [(2, 1), (3, 2)],
    butterdiff: [(0, -1), (1, -1)],
    finitediff: [(0, -1), (1, -1)],
    waveletdiff: [(1, 0), (2, 2)],
    polydiff: [(1, -1), (1, 0)],
    savgoldiff: [(0, -1), (1, 1)],
    rtsdiff: [(1, -1), (1, 0)],
    spectraldiff: [(2, 1), (3, 2)], # lot of Gibbs ringing in 2nd order derivatives along t1 with t_1^2 sin(3 pi t_2 / 2)
    rbfdiff: [(0, -1), (1, 0)],
    splinediff: [(-8, -8), (-6, -7)],
    robustdiff: [(-2, -3), (-1, -2)],
    tvrdiff: [(0, -1), (1, 0)],
    lineardiff: [(0, -1), (1, -1)] # gamma is near zero because this surface is noise-free; there is no noise to trade fidelity against
}

@mark.parametrize("multidim_method_and_params", multidim_methods_and_params)
def test_multidimensionality(multidim_method_and_params, request):
    """Ensure methods with an axis parameter can successfully differentiate in independent directions"""
    diff_method, params = multidim_method_and_params

    # d^2 / dt_1 dt_2
    analytic_d2 = 3 * T1 * np.pi * np.cos(3/2 * np.pi * T2)
    dxdt1 = diff_method(x, dt2, **params, axis=0)[1]
    computed_d2 = diff_method(dxdt1, dt2, **params, axis=1)[1]
    l2_error_d2 = np.linalg.norm(analytic_d2 - computed_d2) # Frobenius norm (2 norm of vectorized array)
    linf_error_d2 = np.max(np.abs(analytic_d2 - computed_d2))

    # Laplacian
    analytic_laplacian = 2 * np.sin(3/2 * np.pi * T2) - 9/4 * np.pi**2 * T1**2 * np.sin(3/2 * np.pi * T2)
    dxdt2 = diff_method(x, dt2, **params, axis=1)[1]
    computed_laplacian = diff_method(dxdt1, dt2, **params, axis=0)[1] + diff_method(dxdt2, dt2, **params, axis=1)[1]
    l2_error_lap = np.linalg.norm(analytic_laplacian - computed_laplacian)
    linf_error_lap = np.max(np.abs(analytic_laplacian - computed_laplacian))

    if request.config.getoption("--bounds"):
        print([(int(np.ceil(np.log10(l2_error_d2))), int(np.ceil(np.log10(linf_error_d2)))),
            (int(np.ceil(np.log10(l2_error_lap))), int(np.ceil(np.log10(linf_error_lap))))])
    else:
        (log_l2_bound_d2, log_linf_bound_d2), (log_l2_bound_lap, log_linf_bound_lap) = multidim_error_bounds[diff_method]
        assert l2_error_d2 < 10**log_l2_bound_d2
        assert linf_error_d2 < 10**log_linf_bound_d2
        assert l2_error_lap < 10**log_l2_bound_lap
        assert linf_error_lap < 10**log_linf_bound_lap

    if request.config.getoption("--plot"):
        from matplotlib import pyplot
        fig = pyplot.figure(figsize=(12, 5), constrained_layout=True)
        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax1.plot_surface(T1, T2, x, cmap='viridis', alpha=0.5)
        ax1.set_title(r'original function, $x$')
        ax1.set_xlabel(r'$t_1$')
        ax1.set_ylabel(r'$t_2$')
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax2.plot_surface(T1, T2, analytic_d2, cmap='viridis', alpha=0.5)
        ax2.set_title(r'$\frac{\partial^2 x}{\partial t_1 \partial t_2}$')
        ax2.set_xlabel(r'$t_1$')
        ax2.set_ylabel(r'$t_2$')
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        ax3.plot_surface(T1, T2, analytic_laplacian, cmap='viridis', alpha=0.5, label='analytic')
        ax3.set_title(r'$\frac{\partial^2}{\partial t_1^2} + \frac{\partial^2}{\partial t_2^2}$')
        ax3.set_xlabel(r'$t_1$')
        ax3.set_ylabel(r'$t_2$')

        ax2.plot_wireframe(T1, T2, computed_d2)
        ax3.plot_wireframe(T1, T2, computed_laplacian, label='computed')
        legend = ax3.legend(bbox_to_anchor=(0.7, 0.8)); legend.legend_handles[0].set_facecolor(pyplot.cm.viridis(0.6))
        fig.suptitle(f'{diff_method.__name__}', fontsize=16)


nan_methods_and_params = [ # List of methods that can handle missing values
    (splinediff, {'degree': 5, 's': 2}),
    (polydiff, {'degree': 2, 'window_size': 9}),
    (rtsdiff, {'order': 2, 'log_qr_ratio': 7, 'forwardbackward': True}),
    (robustdiff, {'order': 3, 'log_q': 7, 'log_r': 2}),
    (lineardiff, {'order': 3, 'gamma': 0.003, 'window_size': 21}), # wider window to straddle the NaN run and still have 2*order observations
]

@mark.parametrize("diff_method_and_params", nan_methods_and_params)
def test_missing_data(diff_method_and_params):
    """Ensure methods that support missing data return finite outputs when NaN values are present"""
    diff_method, params = diff_method_and_params

    x_nan = np.sin(t)
    x_nan[[5, 10, 15]] = np.nan # introduce missing data at several point locations
    x_nan[22:26] = np.nan # and a contiguous run
    x_hat, dxdt_hat = diff_method(x_nan, dt, **params)

    assert np.all(np.isfinite(x_hat))
    assert np.all(np.isfinite(dxdt_hat))


@mark.parametrize("fwdbwd", [False, True])
def test_circular_rtsdiff(request, fwdbwd):
    """Ensure rtsdiff with circular=True correctly differentiates a wrapping angle signal in radians"""
    dthdt = 5 # constant angular velocity in rad/s
    th = dthdt * t # linearly increasing angle, crosses 2*pi boundaries
    th_noisy = np.angle(np.exp(1j * (th + noise))) # add noise and wrap to [-pi, pi]

    th_hat_naive, dthdt_hat_naive = rtsdiff(th_noisy, dt, order=1, log_qr_ratio=1, circular=False, forwardbackward=fwdbwd)
    th_hat, dthdt_hat = rtsdiff(th_noisy, dt, order=1, log_qr_ratio=1, circular=True, forwardbackward=fwdbwd)
    
    naive_rmse = np.sqrt(np.mean((dthdt_hat_naive - dthdt)**2))
    wrapped_rmse = np.sqrt(np.mean((dthdt_hat - dthdt)**2))
    assert wrapped_rmse < naive_rmse

    th_rmse = np.sqrt(np.mean(np.angle(np.exp(1j * (th_hat - th)))**2)) # angular error
    assert th_rmse < 0.1 # the forward and backward estimates must be blended on a common branch, see #208

    if request.config.getoption("--plot"):
        from matplotlib import pyplot
        fig, (ax1, ax2) = pyplot.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax1.plot(t, th_noisy, 'k+', label=r'$\theta$ noisy (wrapped)')
        ax1.plot(t, th_hat_naive, 'C1--', label=r'$\hat{\theta}$ with circular=False')
        ax1.plot(t, th_hat, 'C0', label=r'$\hat{\theta}$ with circular=True')
        ax1.set_ylabel(r'$\theta$ (rad)')
        ax1.legend()
        ax2.axhline(dthdt, color='C2', xmin=0.045, xmax=0.955, label=r'true $\dot{\theta}$')
        ax2.plot(t, dthdt_hat_naive, 'C1--', label=r'$\hat{\dot{\theta}}$ circular=False')
        ax2.plot(t, dthdt_hat, 'C0', label=r'$\hat{\dot{\theta}}$ circular=True')
        ax2.set_ylabel(r'$\dot{\theta}$ (rad/time)')
        ax2.set_xlabel('t')
        ax2.legend()
        fig.suptitle(f'rtsdiff with circular domain, forwardbackward={fwdbwd}', fontsize=16)
