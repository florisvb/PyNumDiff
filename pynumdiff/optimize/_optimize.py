"""Optimization"""
import scipy.optimize
import numpy as np
from itertools import product
from functools import partial
from warnings import filterwarnings
from multiprocessing import Pool

from ..utils import evaluate
from ..finite_difference import first_order
from ..smooth_finite_difference import mediandiff, meandiff, gaussiandiff, friedrichsdiff, butterdiff, splinediff
from ..linear_model import spectraldiff, polydiff, savgoldiff, lineardiff
from ..total_variation_regularization import velocity, acceleration, jerk, iterative_velocity, smooth_acceleration, jerk_sliding
from ..kalman_smooth import constant_velocity, constant_acceleration, constant_jerk


# Map from method -> (search_space, bounds_low_hi)
method_params_and_bounds = {
    spectraldiff: ({'even_extension': True,
                   'pad_to_zero_dxdt': True,
                   'high_freq_cutoff': [1e-3, 5e-2, 1e-2, 5e-2, 1e-1]},
                  {'high_freq_cutoff': (1e-5, 1-1e-5)}),
    polydiff: ({'step_size': [1, 2, 5],
                'kernel': 'friedrichs',
                'poly_order': [2, 3, 5, 7],
                'window_size': [11, 31, 51, 91, 131]},
               {'step_size': (1, 100),
                'poly_order': (1, 8),
                'window_size': (10, 1000)}),
    savgoldiff: ({'poly_order': [2, 3, 5, 7, 9, 11, 13],
                  'window_size': [3, 10, 30, 50, 90, 130, 200, 300],
                  'smoothing_win': [3, 10, 30, 50, 90, 130, 200, 300]},
                 {'poly_order': (1, 12),
                  'window_size': (3, 1000),
                  'smoothing_win': (3, 1000)}),
    #chebydiff ({order: [3, 5, 7, 9],
    #           window_size: [10, 30, 50, 90, 130],
    #           kernel: 'friedrichs'},
    #           {order: (1, 10),
    #           window_size: (10, 1000)})
    lineardiff: ({'kernel': 'gaussian',
                  'order': 3,
                  'gamma': [1e-1, 1, 10, 100],
                  'window_size': [10, 30, 50, 90, 130]},
                 {'order': (1, 5),
                  'gamma': (1e-3, 1000),
                  'window_size': (15, 1000)}),
    first_order: ({'num_iterations': [5, 10, 30, 50]},
                  {'num_iterations': (1, 1000)}),
    mediandiff: ({'window_size': [5, 15, 30, 50],
                  'num_iterations': [1, 5, 10]},
                {'window_size': (1, 1e6),
                 'num_iterations': (1, 100)}),
    butterdiff: ({'filter_order': [1, 2, 3, 4, 5, 6, 7],
                  'cutoff_freq': [0.0001, 0.001, 0.005, 0.01, 0.1, 0.5],
                  'num_iterations': [1, 5, 10]},
                 {'filter_order': (1, 10),
                  'cutoff_freq': (1e-4, 1-1e-2),
                  'num_iterations': (1, 1000)}),
    splinediff: ({'order': [3, 5],
                  's': [0.5, 0.9, 0.95, 1, 10, 100],
                  'num_iterations': [1, 5, 10]},
                 {'order': (3, 5),
                  's': (1e-2, 1e6),
                  'num_iterations': (1, 10)}),
    velocity: ({'gamma': [1e-2, 1e-1, 1, 10, 100, 1000]},
               {'gamma': (1e-4, 1e7)}),
    iterative_velocity: ({'num_iterations': [1, 5, 10],
                          'gamma': [1e-2, 1e-1, 1, 10, 100, 1000],
                          'scale': 'small'},
                         {'num_iterations': (1, 100), # gets expensive with more iterations
                          'gamma': (1e-4, 1e7)}),
    smooth_acceleration: ({'gamma': [1e-2, 1e-1, 1, 10, 100, 1000],
                           'window_size': [3, 10, 30, 50, 90, 130]},
                          {'gamma': (1e-4, 1e7),
                           'window_size': (1, 1000)}),
    constant_velocity: ({'forwardbackward': [True, False],
                         'q': [1e-8, 1e-4, 1e-1, 1e1, 1e4, 1e8],
                         'r': [1e-8, 1e-4, 1e-1, 1e1, 1e4, 1e8]},
                         {'q': (1e-10, 1e10),
                          'r': (1e-10, 1e10)})
}
for method in [meandiff, gaussiandiff, friedrichsdiff]:
    method_params_and_bounds[method] = method_params_and_bounds[mediandiff]
for method in [acceleration, jerk]:
    method_params_and_bounds[method] = method_params_and_bounds[velocity]
method_params_and_bounds[jerk_sliding] = method_params_and_bounds[smooth_acceleration]
for method in [constant_acceleration, constant_jerk]:
    method_params_and_bounds[method] = method_params_and_bounds[constant_velocity]


# This function has to be at the top level for multiprocessing
def _objective_function(point, func, x, dt, singleton_params, search_space_types, dxdt_truth, metric,
    tvgamma, padding):
    """Function minimized by scipy.optimize.minimize, needs to have the form: (point, *args) -> float
    This is mildly complicated, because "point" controls the settings of a differentiation function, but
    the method may have numerical and non-numerical parameters, and all such parameters are now passed by
    keyword arguments. So the encoded `point` has to be decoded to dict.

    :param np.array point: a numerical vector scipy chooses to try in the objective function
    All other parameters documented in `optimize`

    :return: float, cost of this objective at the point
    """
    point_params = {k:(v if search_space_types[k] == float else 
                int(np.round(v)) if search_space_types[k] == int else
                v > 0.5) for k,v in zip(search_space_types, point)} # point -> dict
    # add back in the singletons we're not searching over
    try: x_hat, dxdt_hat = func(x, dt, **point_params, **singleton_params) # estimate x and dxdt
    except np.linalg.LinAlgError: return 1000000000 # some methods can fail numerically

    # evaluate estimate
    if dxdt_truth is not None: # then minimize ||dxdt_hat - dxdt_truth||^2
        if metric == 'rmse':
            rms_rec_x, rms_x, rms_dxdt = evaluate.metrics(x, dt, x_hat, dxdt_hat, dxdt_truth=dxdt_truth, padding=padding)
            return rms_dxdt
        elif metric == 'error_correlation':
            return evaluate.error_correlation(dxdt_hat, dxdt_truth, padding=padding)
    else: # then minimize [ || integral(dxdt_hat) - x||^2 + gamma*TV(dxdt_hat) ]
        rms_rec_x, rms_x, rms_dxdt = evaluate.metrics(x, dt, x_hat, dxdt_hat, dxdt_truth=None, padding=padding)
        return rms_rec_x + tvgamma*evaluate.total_variation(dxdt_hat, padding=padding)


def optimize(func, x, dt, search_space={}, dxdt_truth=None, tvgamma=1e-2, padding='auto', metric='rmse',
    opt_method='Nelder-Mead', maxiter=10):
    """Find the optimal parameters for a given differentiation method.

    :param function func: differentiation method to optimize parameters for, e.g. linear_model.savgoldiff
    :param np.array[float]: data to differentiate
    :param float dt: step size
    :param dict search_space: function parameter settings to use as initial starting points in optimization,
                    structured as :code:`{param1:[values], param2:[values], param3:value, ...}`. The search space
                    is the Cartesian product. If left None, a default search space of initial values is used.
    :param np.array[float] dxdt_truth: actual time series of the derivative of x, if known
    :param float tvgamma: Only used if :code:`dxdt_truth` is given. Regularization value used to select for parameters
                    that yield a smooth derivative. Larger value results in a smoother derivative.
    :param int padding: number of time steps to ignore at the beginning and end of the time series in the
                    optimization. Larger value causes the optimization to emphasize the accuracy of dxdt in the
                    middle of the time series
    :param str metric: either :code:`'rmse'` or :code:`'error_correlation'`, only applies if :code:`dxdt_truth`
                    is not None, see _objective_function
    :param str opt_method: Optimization technique used by :code:`scipy.minimize`, the workhorse
    :param int maxiter: passed down to :code:`scipy.minimize`, maximum iterations

    :return: tuple[dict, float] of\n
            - **opt_params** -- best parameter settings for the differentation method
            - **opt_value** -- lowest value found for objective function
    """
    if metric not in ['rmse','error_correlation']:
        raise ValueError('`metric` should either be `rmse` or `error_correlation`.')
    if metric == 'error_correlation' and dxdt_truth is None:
        raise ValueError('`metric` can only be `error_correlation` if `dxdt_truth` is given.')

    params, bounds = method_params_and_bounds[func]
    params.update(search_space) # for things not given, use defaults

    # No need to optimize over singletons, just pass them through
    singleton_params = {k:v for k,v in params.items() if not isinstance(v, list)}

    # The search space is the Cartesian product of all dimensions where multiple options are given
    search_space_types = {k:type(v[0]) for k,v in params.items() if isinstance(v, list)} # for converting back and forth from point
    if any(v not in [float, int, bool] for v in search_space_types.values()):
        raise ValueError("Optimization over categorical strings currently not supported")
    # If excluding string type, I can just cast ints and bools to floats, and we're good to go
    cartesian_product = product(*[np.array(params[k]).astype(float) for k in search_space_types])
    
    bounds = [bounds[k] if k in bounds else # pass these to minimize(). It should respect them.
             (0, 1) if v == bool else
             None for k,v in search_space_types.items()]

    # wrap the objective and scipy.optimize.minimize because the objective and options are always the same
    _obj_fun = partial(_objective_function, func=func, x=x, dt=dt, singleton_params=singleton_params,
        search_space_types=search_space_types, dxdt_truth=dxdt_truth, metric=metric, tvgamma=tvgamma,
        padding=padding)
    _minimize = partial(scipy.optimize.minimize, _obj_fun, method=opt_method, bounds=bounds, options={'maxiter':maxiter})

    with Pool(initializer=filterwarnings, initargs=["ignore", '', UserWarning]) as pool: # The heavy lifting
        results = pool.map(_minimize, cartesian_product) # returns a bunch of OptimizeResult objects

    opt_idx = np.nanargmin([r.fun for r in results])
    opt_point = results[opt_idx].x
    # results are going to be floats, but that may not be allowed, so convert back to a dict
    opt_params = {k:(v if search_space_types[k] == float else 
                    int(np.round(v)) if search_space_types[k] == int else
                    v > 0.5) for k,v in zip(search_space_types, opt_point)}
    opt_params.update(singleton_params)

    return opt_params, results[opt_idx].fun
