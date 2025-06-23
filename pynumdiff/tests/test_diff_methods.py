import numpy as np
from pytest import mark
from warnings import warn

from ..linear_model import lineardiff, polydiff, savgoldiff, spectraldiff
from ..total_variation_regularization import velocity, acceleration, jerk, iterative_velocity
from ..kalman_smooth import * # constant_velocity, constant_acceleration, constant_jerk, known_dynamics
from ..smooth_finite_difference import * # mediandiff, meandiff, gaussiandiff, friedrichsdiff, butterdiff, splinediff
from ..finite_difference import first_order, second_order


dt = 0.01
t = np.arange(0, 3, dt) # domain to test over
np.random.seed(7) # for repeatability of the test, so we don't get random failures
noise = 0.05*np.random.randn(*t.shape)

diff_methods_and_params = [
    #(lineardiff, {'order':3, 'gamma':5, 'window_size':10, 'solver':'CVXOPT'}),
    (polydiff, {'polynomial_order':2, 'window_size':3}),
    (savgoldiff, {'polynomial_order':2, 'window_size':4, 'smoothing_win':4}),
    (spectraldiff, {'high_freq_cutoff':0.1})
    ]

# Analytic (function, derivative) pairs on which to test differentiation methods.
test_funcs_and_derivs = [
    (0, lambda t: np.ones(t.shape), lambda t: np.zeros(t.shape)), # x(t) = 1
    (1, lambda t: t, lambda t: np.ones(t.shape)),           # x(t) = t
    (2, lambda t: 2*t + 1, lambda t: 2*np.ones(t.shape)),   # x(t) = 2t+1
    (3, lambda t: t**2 - t + 1, lambda t: 2*t - 1),         # x(t) = t^2 - t + 1
    (4, lambda t: np.sin(t) + 1/2, lambda t: np.cos(t))]    # x(t) = sin(t) + 1/2

# All the testing methodology follows the exact same pattern; the only thing that changes is the
# closeness to the right answer various methods achieve with the given parameterizations. So index a
# big ol' table by the method, then the test function, then the pair of quantities we're comparing.
error_bounds = {
    lineardiff: [[(1e-25, 1e-25)]*4]*len(test_funcs_and_derivs),
    polydiff:    [[(1e-14, 1e-15), (1e-12, 1e-13), (1, 0.1), (100, 100)],
                 [(1e-13, 1e-14), (1e-12, 1e-13), (1, 0.1), (100, 100)],
                 [(1e-13, 1e-14), (1e-11, 1e-12), (1, 0.1), (100, 100)],
                 [(1e-13, 1e-14), (1e-12, 1e-12), (1, 0.1), (100, 100)],
                 [(1e-6, 1e-7), (0.001, 0.0001), (1, 0.1), (100, 100)]],
    savgoldiff: [[(1e-7, 1e-8), (1e-12, 1e-13), (1, 0.1), (100, 10)],
                 [(1e-5, 1e-7), (1e-12, 1e-13), (1, 0.1), (100, 10)],
                 [(1e-7, 1e-8), (1e-11, 1e-12), (1, 0.1), (100, 10)],
                 [(0.1, 0.01), (0.1, 0.01), (1, 0.1), (100, 10)],
                 [(0.01, 1e-3), (0.01, 1e-3), (1, 0.1), (100, 10)]],
    spectraldiff: [[(1e-7, 1e-8), ( 1e-25 , 1e-25), (1, 0.1), (100, 10)],
                   [(0.1, 0.1), (10, 10), (1, 0.1), (100, 10)],
                   [(0.1, 0.1), (10, 10), (1, 0.1), (100, 10)],
                   [(1, 1), (100, 10), (1, 1), (100, 10)],
                   [(0.1, 0.1), (10, 10), (1, 0.1), (100, 10)]]
}


@mark.parametrize("diff_method_and_params", diff_methods_and_params)
@mark.parametrize("test_func_and_deriv", test_funcs_and_derivs)
def test_diff_method(diff_method_and_params, test_func_and_deriv):
    diff_method, params = diff_method_and_params # unpack
    i, f, df = test_func_and_deriv

    # some methods rely on cvxpy, and we'd like to allow use of pynumdiff without convex optimization
    if diff_method in [lineardiff, velocity]:
        try: import cvxpy
        except: warn(f"Cannot import cvxpy, skipping {diff_method} test."); return

    x = f(t) # sample the function
    x_noisy = x + noise # add a little noise
    dxdt = df(t) # true values of the derivative

    # differentiate without and with noise
    x_hat, dxdt_hat = diff_method(x, dt, **params) if isinstance(params, dict) else diff_method(x, dt, params)
    x_hat_noisy, dxdt_hat_noisy = diff_method(x_noisy, dt, **params) if isinstance(params, dict) else diff_method(x_noisy, dt, params)
    
    # check x_hat and x_hat_noisy are close to x and dxdt_hat and dxdt_hat_noisy are close to dxdt
    #print("]\n[", end="")
    for j,(a,b) in enumerate([(x,x_hat), (dxdt,dxdt_hat), (x,x_hat_noisy), (dxdt,dxdt_hat_noisy)]):
        l2_error = np.linalg.norm(a - b)
        linf_error = np.max(np.abs(a - b))
        
        #print(f"({10 ** np.ceil(np.log10(l2_error)) if l2_error> 0 else 1e-25}, {10 ** np.ceil(np.log10(linf_error)) if linf_error > 0 else 1e-25})", end=", ")
        l2_bound, linf_bound = error_bounds[diff_method][i][j]
        assert np.linalg.norm(a - b) < l2_bound
        assert np.max(np.abs(a - b)) < linf_bound
