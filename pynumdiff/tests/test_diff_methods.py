import numpy as np
from pytest import mark
from warnings import warn

from ..finite_difference import first_order, second_order, fourth_order
from ..linear_model import lineardiff, polydiff, savgoldiff, spectraldiff
from ..total_variation_regularization import velocity, acceleration, jerk, iterative_velocity, smooth_acceleration, jerk_sliding
from ..kalman_smooth import constant_velocity, constant_acceleration, constant_jerk
from ..smooth_finite_difference import mediandiff, meandiff, gaussiandiff, friedrichsdiff, butterdiff, splinediff
# Function alias for testing a case where parameters change the behavior in a big way
def iterated_first_order(*args, **kwargs): return first_order(*args, **kwargs)

dt = 0.1
t = np.linspace(0, 3, 31) # sample locations, including the endpoint
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

# Call both ways, with kwargs (new) and with params list and optional options dict (legacy), to ensure both work
diff_methods_and_params = [
    (first_order, {}), (second_order, {}), (fourth_order, {}), # empty dictionary for the case of no parameters
    (iterated_first_order, {'num_iterations':2}), (iterated_first_order, [2], {'iterate':True}),
    (lineardiff, {'order':3, 'gamma':5, 'window_size':11, 'solver':'CLARABEL'}), (lineardiff, [3, 5, 11], {'solver':'CLARABEL'}),
    (polydiff, {'poly_order':2, 'window_size':3}), (polydiff, [2, 3]),
    (savgoldiff, {'poly_order':2, 'window_size':5, 'smoothing_win':5}), (savgoldiff, [2, 5, 5]),
    (spectraldiff, {'high_freq_cutoff':0.1}), (spectraldiff, [0.1]),
    (mediandiff, {'window_size':3, 'num_iterations':2}), (mediandiff, [3, 2], {'iterate':True}),
    (meandiff, {'window_size':3, 'num_iterations':2}), (meandiff, [3, 2], {'iterate':True}),
    (gaussiandiff, {'window_size':5}), (gaussiandiff, [5]),
    (friedrichsdiff, {'window_size':5}), (friedrichsdiff, [5]),
    (butterdiff, {'filter_order':3, 'cutoff_freq':0.074}), (butterdiff, [3, 0.074]),
    (splinediff, {'order':5, 's':2}), (splinediff, [5, 2]),
    (constant_velocity, {'r':1e-4, 'q':1e-2}), (constant_velocity, [1e-4, 1e-2]),
    (constant_acceleration, {'r':1e-4, 'q':1e-1}), (constant_acceleration, [1e-4, 1e-1]),
    (constant_jerk, {'r':1e-4, 'q':10}), (constant_jerk, [1e-4, 10]),
    # TODO (known_dynamics), but presently it doesn't calculate a derivative
    (velocity, {'gamma':0.5}), (velocity, [0.5]),
    (acceleration, {'gamma':1}), (acceleration, [1]),
    (jerk, {'gamma':10}), (jerk, [10]),
    (iterative_velocity, {'num_iterations':5, 'gamma':0.05}), (iterative_velocity, [5, 0.05]),
    (smooth_acceleration, {'gamma':2, 'window_size':5}), (smooth_acceleration, [2, 5]),
    (jerk_sliding, {'gamma':1, 'window_size':15}), (jerk_sliding, [1], {'window_size':15})
    ]

# All the testing methodology follows the exact same pattern; the only thing that changes is the
# closeness to the right answer various methods achieve with the given parameterizations. So index a
# big ol' table by the method, then the test function, then the pair of quantities we're comparing.
# The tuples are order of magnitude of (L2,Linf) distances for pairs
# (x,x_hat), (dxdt,dxdt_hat), (x,x_hat_noisy), (dxdt,dxdt_hat_noisy).
error_bounds = {
    first_order: [[(-25, -25), (-25, -25), (0, 0), (1, 1)],
                  [(-25, -25), (-13, -13), (0, 0), (1, 1)],
                  [(-25, -25), (0, 0), (0, 0), (1, 1)],
                  [(-25, -25), (1, 0), (0, 0), (1, 1)],
                  [(-25, -25), (2, 2), (0, 0), (2, 2)],
                  [(-25, -25), (3, 3), (0, 0), (3, 3)]],
    iterated_first_order: [[(-9, -10), (-25, -25), (0, -1), (1, 0)],
                           [(-9, -10), (-13, -14), (0, -1), (1, 0)],
                           [(0, 0), (1, 0), (0, 0), (1, 0)],
                           [(0, 0), (1, 0), (0, 0), (1, 1)],
                           [(1, 1), (2, 2), (1, 1), (2, 2)],
                           [(1, 1), (3, 3), (1, 1), (3, 3)]],
    second_order: [[(-25, -25), (-25, -25), (0, 0), (1, 1)],
                   [(-25, -25), (-13, -13), (0, 0), (1, 1)],
                   [(-25, -25), (-13, -13), (0, 0), (1, 1)],
                   [(-25, -25), (0, -1), (0, 0), (1, 1)],
                   [(-25, -25), (1, 1), (0, 0), (1, 1)],
                   [(-25, -25), (3, 3), (0, 0), (3, 3)]],
    fourth_order: [[(-25, -25), (-25, -25), (0, 0), (1, 1)],
                   [(-25, -25), (-13, -13), (0, 0), (1, 1)],
                   [(-25, -25), (-13, -13), (0, 0), (1, 1)],
                   [(-25, -25), (-2, -2), (0, 0), (1, 1)],
                   [(-25, -25), (1, 0), (0, 0), (1, 1)],
                   [(-25, -25), (2, 2), (0, 0), (2, 2)]],
    lineardiff: [[(-6, -6), (-5, -6), (0, -1), (0, 0)],
                 [(0, 0), (2, 1), (0, 0), (2, 1)],
                 [(1, 0), (2, 2), (1, 0), (2, 2)],
                 [(1, 0), (2, 1), (1, 0), (2, 1)],
                 [(1, 1), (2, 2), (1, 1), (2, 2)],
                 [(1, 1), (3, 3), (1, 1), (3, 3)]],
    polydiff: [[(-14, -15), (-14, -14), (0, -1), (1, 1)],
               [(-14, -14), (-13, -13), (0, -1), (1, 1)],
               [(-14, -14), (-13, -13), (0, -1), (1, 1)],
               [(-2, -2), (0, 0), (0, -1), (1, 1)],
               [(0, 0), (1, 1), (0, -1), (1, 1)],
               [(0, 0), (3, 3), (0, 0), (3, 3)]],
    savgoldiff: [[(-9, -10), (-13, -14), (0, -1), (0, 0)],
                 [(-9, -10), (-13, -13), (0, -1), (0, 0)],
                 [(-2, -2), (-1, -1), (0, -1), (0, 0)],
                 [(0, -1), (0, 0), (0, 0), (1, 0)],
                 [(1, 1), (2, 2), (1, 1), (2, 2)],
                 [(1, 1), (3, 3), (1, 1), (3, 3)]],
    spectraldiff: [[(-9, -10), (-14, -15), (-1, -1), (0, 0)],
                   [(0, 0), (1, 1), (0, 0), (1, 1)],
                   [(1, 1), (1, 1), (1, 1), (1, 1)],
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
               [(0, 0), (1, 1), (0, 0), (1, 1)],
               [(0, 0), (1, 1), (0, 0), (1, 1)],
               [(0, 0), (1, 1), (0, 0), (1, 1)],
               [(1, 1), (2, 2), (1, 1), (2, 2)],
               [(1, 1), (3, 3), (1, 1), (3, 3)]],
    gaussiandiff: [[(-14, -15), (-14, -14), (0, -1), (1, 0)],
                   [(-1, -1), (1, 0), (0, 0), (1, 1)],
                   [(0, 0), (1, 1), (0, 0), (1, 1)],
                   [(0, -1), (1, 1), (0, 0), (1, 1)],
                   [(1, 1), (2, 2), (1, 1), (2, 2)],
                   [(1, 1), (3, 3), (1, 1), (3, 3)]],
    friedrichsdiff: [[(-25, -25), (-25, -25), (0, -1), (1, 0)],
                     [(-1, -1), (1, 0), (0, 0), (1, 1)],
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
    splinediff: [[(-14, -15), (-14, -15), (-1, -1), (0, 0)],
                 [(-14, -14), (-13, -14), (-1, -1), (0, 0)],
                 [(-14, -14), (-13, -13), (-1, -1), (0, 0)],
                 [(0, 0), (1, 1), (0, 0), (1, 1)],
                 [(1, 0), (2, 2), (1, 0), (2, 2)],
                 [(1, 0), (3, 3), (1, 0), (3, 3)]],
    constant_velocity: [[(-25, -25), (-25, -25), (0, -1), (0, 0)],
                        [(-6, -6), (-5, -5), (0, -1), (0, 0)],
                        [(-1, -2), (0, 0), (0, -1), (0, 0)],
                        [(-1, -1), (1, 0), (0, -1), (1, 0)],
                        [(1, 1), (2, 2), (1, 1), (2, 2)],
                        [(1, 1), (3, 3), (1, 1), (3, 3)]],
    constant_acceleration: [[(-25, -25), (-25, -25), (0, -1), (0, 0)],
                            [(-5, -6), (-4, -5), (0, -1), (0, 0)],
                            [(-5, -5), (-4, -4), (0, -1), (0, 0)],
                            [(-1, -1), (0, 0), (0, -1), (0, 0)],
                            [(1, 1), (2, 2), (1, 1), (2, 2)],
                            [(1, 1), (3, 3), (1, 1), (3, 3)]],
    constant_jerk: [[(-25, -25), (-25, -25), (0, -1), (0, 0)],
                    [(-5, -5), (-4, -5), (0, -1), (0, 0)],
                    [(-4, -5), (-3, -4), (0, -1), (0, 0)],
                    [(-1, -2), (0, 0), (0, -1), (0, 0)],
                    [(1, 0), (2, 1), (1, 0), (2, 1)],
                    [(1, 1), (3, 3), (1, 1), (3, 3)]],
    velocity: [[(-25, -25), (-18, -19), (0, -1), (1, 0)],
               [(-12, -12), (-11, -12), (-1, -1), (-1, -2)],
               [(0, 0), (1, 0), (0, 0), (1, 0)],
               [(0, -1), (1, 1), (0, 0), (1, 0)],
               [(1, 1), (2, 2), (1, 1), (2, 2)],
               [(1, 0), (3, 3), (1, 0), (3, 3)]],
    acceleration: [[(-25, -25), (-18, -18), (0, -1), (0, 0)],
                   [(-10, -10), (-9, -9), (-1, -1), (0, -1)],
                   [(-10, -10), (-9, -10), (-1, -1), (0, -1)],
                   [(0, -1), (1, 0), (0, -1), (1, 0)],
                   [(1, 1), (2, 2), (1, 1), (2, 2)],
                   [(1, 1), (3, 3), (1, 1), (3, 3)]],
    jerk: [[(-25, -25), (-18, -18), (-1, -1), (0, 0)],
           [(-9, -10), (-9, -9), (-1, -1), (0, 0)],
           [(-10, -10), (-9, -10), (-1, -1), (0, 0)],
           [(0, 0), (1, 1), (0, 0), (1, 1)],
           [(1, 1), (2, 2), (1, 1), (2, 2)],
           [(1, 1), (3, 3), (1, 1), (3, 3)]],
    iterative_velocity: [[(-9, -10), (-25, -25), (0, -1), (0, 0)],
                         [(0, 0), (0, 0), (0, 0), (1, 0)],
                         [(0, 0), (1, 0), (1, 0), (1, 0)],
                         [(1, 0), (1, 1), (1, 0), (1, 1)],
                         [(2, 1), (2, 2), (2, 1), (2, 2)],
                         [(1, 1), (3, 3), (1, 1), (3, 3)]],
    smooth_acceleration: [[(-9, -10), (-18, -18), (0, -1), (0, 0)],
                          [(-9, -9), (-10, -10), (-1, -1), (-1, -1)],
                          [(-2, -2), (-1, -1), (-1, -1), (0, -1)],
                          [(0, 0), (1, 0), (0, -1), (1, 0)],
                          [(1, 1), (2, 2), (1, 1), (2, 2)],
                          [(1, 1), (3, 3), (1, 1), (3, 3)]],
    jerk_sliding: [[(-15, -15), (-16, -16), (0, -1), (1, 0)],
                   [(-14, -14), (-14, -14), (0, -1), (0, 0)],
                   [(-14, -14), (-14, -14), (0, -1), (0, 0)],
                   [(-1, -1), (0, 0), (0, -1), (1, 0)],
                   [(0, 0), (2, 2), (0, 0), (2, 2)],
                   [(1, 1), (3, 3), (1, 1), (3, 3)]]
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
    if diff_method in [lineardiff, velocity, acceleration, jerk, smooth_acceleration]:
        try: import cvxpy
        except: warn(f"Cannot import cvxpy, skipping {diff_method} test."); return

    # sample the true function and make noisy samples, and sample true derivative
    x = f(t)
    x_noisy = x + noise
    dxdt = df(t)

    # differentiate without and with noise, accounting for new and old styles of calling functions
    x_hat, dxdt_hat = diff_method(x, dt, **params) if isinstance(params, dict) \
        else diff_method(x, dt, params) if (isinstance(params, list) and len(diff_method_and_params) < 3) \
        else diff_method(x, dt, params, options)
    x_hat_noisy, dxdt_hat_noisy = diff_method(x_noisy, dt, **params) if isinstance(params, dict) \
        else diff_method(x_noisy, dt, params) if (isinstance(params, list) and len(diff_method_and_params) < 3) \
        else diff_method(x_noisy, dt, params, options)

    # plotting code
    if request.config.getoption("--plot") and not isinstance(params, list): # Get the plot flag from pytest configuration
        fig, axes = request.config.plots[diff_method] # get the appropriate plot, set up by the store_plots fixture in conftest.py
        axes[i, 0].plot(t, f(t))
        axes[i, 0].plot(t, x, 'C0+')
        axes[i, 0].plot(t, x_hat, 'C2.', ms=4)
        axes[i, 0].plot(tt, df(tt))
        axes[i, 0].plot(t, dxdt_hat, 'C1+')
        axes[i, 0].set_ylabel(latex_name, rotation=0, labelpad=50)
        if i < len(test_funcs_and_derivs)-1: axes[i, 0].set_xticklabels([])
        else: axes[i, 0].set_xlabel('t')
        if i == 0: axes[i, 0].set_title('noiseless')
        axes[i, 1].plot(t, f(t), label=r"$x(t)$")
        axes[i, 1].plot(t, x_noisy, 'C0+', label=r"$x_n$")
        axes[i, 1].plot(t, x_hat_noisy, 'C2.', ms=4, label=r"$\hat{x}_n$")
        axes[i, 1].plot(tt, df(tt), label=r"$\frac{dx(t)}{dt}$")
        axes[i, 1].plot(t, dxdt_hat_noisy, 'C1+', label=r"$\hat{\frac{dx}{dt}}_n$")
        if i < len(test_funcs_and_derivs)-1: axes[i, 1].set_xticklabels([])
        else: axes[i, 1].set_xlabel('t')
        axes[i, 1].set_yticklabels([])
        if i == 0: axes[i, 1].set_title('with noise')

    # check x_hat and x_hat_noisy are close to x and that dxdt_hat and dxdt_hat_noisy are close to dxdt
    if request.config.getoption("--bounds"): print("\n[", end="")
    for j,(a,b) in enumerate([(x,x_hat), (dxdt,dxdt_hat), (x,x_hat_noisy), (dxdt,dxdt_hat_noisy)]):
        l2_error = np.linalg.norm(a - b)
        linf_error = np.max(np.abs(a - b))

        # bounds-printing for establishing bounds
        if request.config.getoption("--bounds"):
            #print(f"({l2_error},{linf_error})", end=", ")
            print(f"({int(np.ceil(np.log10(l2_error))) if l2_error > 0 else -25}, {int(np.ceil(np.log10(linf_error))) if linf_error > 0 else -25})", end=", ")
        # bounds checking
        else:
            log_l2_bound, log_linf_bound = error_bounds[diff_method][i][j]
            assert l2_error < 10**log_l2_bound
            assert linf_error < 10**log_linf_bound
            # methods that get super duper close can converge to different very small limits on different runs
            if 1e-18 < l2_error < 10**(log_l2_bound - 1) or 1e-18 < linf_error < 10**(log_linf_bound - 1):
                print(f"Improvement detected for method {diff_method.__name__}")
