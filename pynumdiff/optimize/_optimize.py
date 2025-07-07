"""Optimization"""
import scipy.optimize
import numpy as np
from itertools import product
from functools import partial
from multiprocessing import Pool

from pynumdiff.utils import evaluate

from ..linear_model import spectraldiff, polydiff


# Map from method -> (init_conds, type_low_hi)
method_params_and_bounds = {
    spectraldiff: ({'even_extension': True,
                   'pad_to_zero_dxdt': True,
                   'high_freq_cutoff': [1e-3, 5e-2, 1e-2, 5e-2, 1e-1]},
                  {'high_freq_cutoff': [1e-5, 1-1e-5]}),
    polydiff: ({'sliding': True,
                'step_size': 1,
                'kernel': 'friedrichs',
                'order': [2, 3, 5, 7],
                'window_size': [10, 30, 50, 90, 130]},
               {'order': [1, 8],
                'window_size': [10, 1000]})
}


# This function to be at the top level for multiprocessing
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
    x_hat, dxdt_hat = func(x, dt, **point_params, **singleton_params) # estimate x and dxdt

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


def optimize(func, x, dt, init_conds={}, dxdt_truth=None, tvgamma=1e-2, padding='auto', metric='rmse',
    opt_method='Nelder-Mead', opt_kwargs={'maxiter': 10}):
    """Find the optimal parameters for a given differentiation method.

    :param function func: differentiation method to optimize parameters for, e.g. linear_model.savgoldiff
    :param np.array[float]: data to differentiate
    :param float dt: step size
    :param dict init_conds: function parameter settings to use as initial starting points in optimization,
                    structured as :code:`{param1:[values], param2:[values], param3:value, ...}`. If left None,
                    a default search space of initial values is used.
    :param np.array[float] dxdt_truth: actual time series of the derivative of x, if known
    :param float tvgamma: regularization value used to select for parameters that yield a smooth derivative.
                    Larger value results in a smoother derivative
    :param int padding: number of time steps to ignore at the beginning and end of the time series in the
                    optimization. Larger value causes the optimization to emphasize the accuracy of dxdt in the
                    middle of the time series
    :param str metric: either :code:`'rmse'` or :code:`'error_correlation'`, only applies if :code:`dxdt_truth`
                    is not None, see _objective_function
    :param str opt_method: Optimization technique used by :code:`scipy.minimize`, the workhorse
    :param dict opt_kwargs: keyword arguments to pass down to :code:`scipy.minimize`

    :return: tuple[dict, float] of\n
            - **opt_params** -- best parameter settings for the differentation method
            - **opt_value** -- lowest value found for objective function
    """
    if metric not in ['rmse','error_correlation']:
        raise ValueError('`metric` should either be `rmse` or `error_correlation`.')
    if metric == 'error_correlation' and dxdt_truth is None:
        raise ValueError('`metric` can only be `error_correlation` if `dxdt_truth` is given.')

    params, bounds = method_params_and_bounds[func]
    params.update(init_conds) # for things not given, use defaults

    # No need to optimize over singletons, just pass them through
    singleton_params = {k:v for k,v in params.items() if not isinstance(v, list)}

    # The search space is the cartesian product of all dimensions where multiple options are given
    search_space_types = {k:type(v[0]) for k,v in params.items() if isinstance(v, list)} # for converting back and forth from point
    if any(v not in [float, int, bool] for v in search_space_types.values()):
        raise ValueError("Optimization over categorical strings currently not supported")
    # If excluding string type, I can just cast ints and bools to floats, and we're good to go
    search_space = product(*[np.array(params[k]).astype(float) for k in search_space_types]) # 
    
    bounds = [bounds[k] if k in bounds else # pass these to minimize(). It should respect them.
             (0, 1) if v == bool else
             None for k,v in search_space_types.items()]

    # wrap the objective and scipy.optimize.minimize because the objective and options are always the same
    _obj_fun = partial(_objective_function, func=func, x=x, dt=dt, singleton_params=singleton_params,
        search_space_types=search_space_types, dxdt_truth=dxdt_truth, metric=metric, tvgamma=tvgamma,
        padding=padding)
    _minimize = partial(scipy.optimize.minimize, _obj_fun, method=opt_method, bounds=bounds, options=opt_kwargs)

    with Pool() as pool: # The heavy lifting
        results = pool.map(_minimize, search_space) # returns a bunch of OptimizeResult objects

    opt_idx = np.nanargmin([r.fun for r in results])
    opt_point = results[opt_idx].x
    # results are going to be floats, but that may not be allowed, so convert back to a dict
    opt_params = {k:(v if search_space_types[k] == float else 
                    int(np.round(v)) if search_space_types[k] == int else
                    v > 0.5) for k,v in zip(search_space_types, opt_point)}
    opt_params.update(singleton_params)

    return opt_params, results[opt_idx].fun
