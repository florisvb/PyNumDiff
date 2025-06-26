import numpy as np
from matplotlib import pyplot
from pytest import mark
from warnings import warn

from ..linear_model import lineardiff, polydiff, savgoldiff, spectraldiff
from ..total_variation_regularization import velocity, acceleration, jerk, iterative_velocity
from ..kalman_smooth import * # constant_velocity, constant_acceleration, constant_jerk, known_dynamics
from ..smooth_finite_difference import mediandiff, meandiff, gaussiandiff, friedrichsdiff, butterdiff, splinediff
from ..finite_difference import first_order, second_order
# Function aliases for testing cases where parameters change the behavior in a big way
def iterated_first_order(*args, **kwargs): return first_order(*args, **kwargs)

dt = 0.1
t = np.arange(0, 3+dt, dt) # sample locations, including the endpoint
tt = np.linspace(0, 3) # full domain, for visualizing denser plots
np.random.seed(7) # for repeatability of the test, so we don't get random failures
noise = 0.05*np.random.randn(*t.shape)

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

# Call both ways, with kwargs (new) and with params list with default options dict (legacy), to ensure both work
diff_methods_and_params = [
    (first_order, {}), # empty dictionary for the case of no parameters. no params -> no diff in new vs old
    (iterated_first_order, {'num_iterations':5}), (iterated_first_order, [5], {'iterate':True}),
    (second_order, {}),
    #(lineardiff, {'order':3, 'gamma':5, 'window_size':10, 'solver':'CVXOPT'}),
    (polydiff, {'polynomial_order':2, 'window_size':3}), (polydiff, [2, 3]),
    (savgoldiff, {'polynomial_order':2, 'window_size':4, 'smoothing_win':4}), (savgoldiff, [2, 4, 4]),
    (spectraldiff, {'high_freq_cutoff':0.1}), (spectraldiff, [0.1]),
    (mediandiff, {'window_size':3, 'num_iterations':2}), (mediandiff, [3, 2], {'iterate':True}),
    (meandiff, {'window_size':3, 'num_iterations':2}), (meandiff, [3, 2], {'iterate':True}),
    (gaussiandiff, {'window_size':5}), (gaussiandiff, [5]),
    (friedrichsdiff, {'window_size':5}), (friedrichsdiff, [5]),
    (butterdiff, {'filter_order':3, 'cutoff_freq':0.074}), (butterdiff, [3, 0.074]),
    (splinediff, {'order':5, 's':2}), (splinediff, [5, 2])
    ]
diff_methods_and_params = [(velocity, [0.5], {'solver': 'CVXOPT'})]

# All the testing methodology follows the exact same pattern; the only thing that changes is the
# closeness to the right answer various methods achieve with the given parameterizations. So index a
# big ol' table by the method, then the test function, then the pair of quantities we're comparing.
error_bounds = {
    first_order: [[(-25, -25), (-25, -25), (0, 0), (1, 1)],
                  [(-25, -25), (-13, -14), (0, 0), (1, 1)],
                  [(-25, -25), (0, 0), (0, 0), (1, 0)],
                  [(-25, -25), (0, 0), (0, 0), (1, 1)],
                  [(-25, -25), (2, 2), (0, 0), (2, 2)],
                  [(-25, -25), (3, 3), (0, 0), (3, 3)]],
    iterated_first_order: [[(-8, -9), (-11, -11), (0, -1), (0, 0)],
                           [(-6, -6), (-6, -7), (0, -1), (0, 0)],
                           [(-1, -1), (0, 0), (0, -1), (0, 0)],
                           [(0, 0), (1, 0), (0, 0), (1, 0)],
                           [(1, 1), (2, 2), (1, 1), (2, 2)],
                           [(1, 1), (3, 3), (1, 1), (3, 3)]],
    second_order: [[(-25, -25), (-25, -25), (0, 0), (1, 1)],
                   [(-25, -25), (-13, -13), (0, 0), (1, 1)],
                   [(-25, -25), (-13, -13), (0, 0), (1, 1)],
                   [(-25, -25), (0, -1), (0, 0), (1, 1)],
                   [(-25, -25), (1, 1), (0, 0), (1, 1)],
                   [(-25, -25), (3, 3), (0, 0), (3, 3)]],
    #lineardiff: [TBD when #91 is solved],
    polydiff: [[(-14, -15), (-14, -14), (0, -1), (1, 1)],
               [(-14, -14), (-13, -13), (0, -1), (1, 1)],
               [(-14, -14), (-13, -13), (0, -1), (1, 1)],
               [(-2, -2), (0, 0), (0, -1), (1, 1)],
               [(0, 0), (1, 1), (0, -1), (1, 1)],
               [(0, 0), (3, 3), (0, 0), (3, 3)]],
    savgoldiff: [[(-9, -10), (-13, -14), (0, -1), (0, 0)],
                 [(-9, -10), (-13, -13), (0, -1), (0, 0)],
                 [(-1, -1), (0, -1), (0, -1), (0, 0)],
                 [(0, -1), (0, 0), (0, -1), (1, 0)],
                 [(1, 1), (2, 2), (1, 1), (2, 2)],
                 [(1, 1), (3, 3), (1, 1), (3, 3)]],
    spectraldiff: [[(-9, -10), (-14, -15), (-1, -1), (0, 0)],
                   [(0, 0), (1, 1), (0, 0), (1, 1)],
                   [(1, 0), (1, 1), (1, 1), (1, 1)],
                   [(0, 0), (1, 1), (0, 0), (1, 1)],
                   [(1, 1), (2, 2), (1, 1), (2, 2)],
                   [(1, 1), (3, 3), (1, 1), (3, 3)]],
    mediandiff: [[(-25, -25), (-25, -25), (-1, -1), (0, 0)],
                 [(0, 0), (1, 1), (0, 0), (1, 1)],
                 [(0, 0), (1, 1), (0, 0), (1, 1)],
                 [(-1, -1), (0, 0), (0, 0), (1, 1)],
                 [(0, 0), (2, 2), (0, 0), (2, 2)],
                 [(1, 1), (3, 3), (1, 1), (3, 3)]],
    meandiff: [[(-25, -25), (-25, -25), (0, -1), (0, 0)],
               [(0, 0), (1, 0), (0, 0), (1, 1)],
               [(0, 0), (1, 1), (0, 0), (1, 1)],
               [(0, 0), (1, 1), (0, 0), (1, 1)],
               [(1, 1), (2, 2), (1, 1), (2, 2)],
               [(1, 1), (3, 3), (1, 1), (3, 3)]],
    gaussiandiff: [[(-14, -15), (-14, -14), (0, -1), (1, 0)],
                   [(-1, -1), (0, 0), (0, 0), (1, 0)],
                   [(0, 0), (1, 1), (0, 0), (1, 1)],
                   [(0, -1), (1, 0), (0, 0), (1, 1)],
                   [(1, 1), (2, 2), (1, 1), (2, 2)],
                   [(1, 1), (3, 3), (1, 1), (3, 3)]],
    friedrichsdiff: [[(-25, -25), (-25, -25), (0, -1), (0, 0)],
                     [(-1, -1), (0, 0), (0, 0), (1, 0)],
                     [(0, 0), (1, 1), (0, 0), (1, 1)],
                     [(0, -1), (1, 1), (0, 0), (1, 1)],
                     [(1, 1), (2, 2), (1, 1), (2, 2)],
                     [(1, 1), (3, 3), (1, 1), (3, 3)]],
    butterdiff: [[(-13, -14), (-13, -13), (0, -1), (0, 0)],
                 [(0, -1), (0, 0), (0, -1), (0, 0)],
                 [(0, 0), (1, 1), (0, 0), (1, 1)],
                 [(1, 0), (1, 1), (1, 0), (1, 1)],
                 [(2, 2), (3, 2), (2, 2), (3, 2)],
                 [(2, 1), (3, 3), (2, 1), (3, 3)]],
    splinediff: [[(-14, -15), (-14, -14), (-1, -1), (0, 0)],
                 [(-14, -14), (-13, -14), (-1, -1), (0, 0)],
                 [(-14, -14), (0, 0), (-1, -1), (0, 0)],
                 [(0, 0), (1, 1), (0, 0), (1, 1)],
                 [(1, 0), (2, 2), (1, 0), (2, 2)],
                 [(1, 0), (3, 3), (1, 0), (3, 3)]]
}

# Essentially run the cartesian product of [diff methods] x [test functions] through this one test
@mark.filterwarnings("ignore::DeprecationWarning") # I want to test the old and new functionality intentionally
@mark.parametrize("diff_method_and_params", diff_methods_and_params)
@mark.parametrize("test_func_and_deriv", test_funcs_and_derivs)
def test_diff_method(diff_method_and_params, test_func_and_deriv, request): # request gives access to context
    # unpack
    diff_method, params = diff_method_and_params[:2]
    if len(diff_method_and_params) == 3: options = diff_method_and_params[2] # optionally pass old-style `options` dict
    i, latex_name, f, df = test_func_and_deriv

    # some methods rely on cvxpy, and we'd like to allow use of pynumdiff without convex optimization
    if diff_method in [lineardiff, velocity]:
        try: import cvxpy
        except: warn(f"Cannot import cvxpy, skipping {diff_method} test."); return

    x = f(t) # sample the function
    x_noisy = x + noise # add a little noise
    dxdt = df(t) # true values of the derivative at samples

    # differentiate without and with noise, accounting for new and old styles of calling functions
    x_hat, dxdt_hat = diff_method(x, dt, **params) if isinstance(params, dict) \
        else diff_method(x, dt, params) if (isinstance(params, list) and len(diff_method_and_params) < 3) \
        else diff_method(x, dt, params, options)
    x_hat_noisy, dxdt_hat_noisy = diff_method(x_noisy, dt, **params) if isinstance(params, dict) \
        else diff_method(x_noisy, dt, params) if (isinstance(params, list) and len(diff_method_and_params) < 3) \
        else diff_method(x_noisy, dt, params, options)
    
    # check x_hat and x_hat_noisy are close to x and that dxdt_hat and dxdt_hat_noisy are close to dxdt
    print("]\n[", end="")
    for j,(a,b) in enumerate([(x,x_hat), (dxdt,dxdt_hat), (x,x_hat_noisy), (dxdt,dxdt_hat_noisy)]):
        l2_error = np.linalg.norm(a - b)
        linf_error = np.max(np.abs(a - b))
        #print(f"({l2_error},{linf_error})", end=", ")
        print(f"({int(np.ceil(np.log10(l2_error))) if l2_error > 0 else -25}, {int(np.ceil(np.log10(linf_error))) if linf_error > 0 else -25})", end=", ")
        
        # log_l2_bound, log_linf_bound = error_bounds[diff_method][i][j]
        # assert np.linalg.norm(a - b) < 10**log_l2_bound
        # assert np.max(np.abs(a - b)) < 10**log_linf_bound
        # if 0 < np.linalg.norm(a - b) < 10**(log_l2_bound - 1) or 0 < np.max(np.abs(a - b)) < 10**(log_linf_bound - 1):
        #     print(f"Improvement detected for method {diff_method.__name__}")

    if request.config.getoption("--plot") and not isinstance(params, list): # Get the plot flag from pytest configuration
        fig, axes = request.config.plots[diff_method] # get the appropriate plot, set up by the store_plots fixture in conftest.py
        axes[i, 0].plot(t, f(t), label=r"$x(t)$")
        axes[i, 0].plot(t, x, 'C0+')
        axes[i, 0].plot(tt, df(tt), label=r"$\frac{dx(t)}{dt}$")
        axes[i, 0].plot(t, dxdt_hat, 'C1+')
        axes[i, 0].set_ylabel(latex_name, rotation=0, labelpad=50)
        if i < len(test_funcs_and_derivs)-1: axes[i, 0].set_xticklabels([])
        else: axes[i, 0].set_xlabel('t')
        if i == 0: axes[i, 0].set_title('noiseless')
        axes[i, 1].plot(t, f(t), label=r"$x(t)$")
        axes[i, 1].plot(t, x_noisy, 'C0+')
        axes[i, 1].plot(tt, df(tt), label=r"$\frac{dx(t)}{dt}$")
        axes[i, 1].plot(t, dxdt_hat_noisy, 'C1+')
        if i < len(test_funcs_and_derivs)-1: axes[i, 1].set_xticklabels([])
        else: axes[i, 1].set_xlabel('t')
        axes[i, 1].set_yticklabels([])
        if i == 0: axes[i, 1].set_title('with noise')
