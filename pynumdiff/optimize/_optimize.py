"""Optimization"""
import scipy.optimize
import numpy as np
from itertools import product
from functools import partial
from warnings import filterwarnings, warn
from multiprocessing import Pool, Manager
from hashlib import sha1
from tqdm import tqdm

from ..utils import evaluate, utility
from ..finite_difference import finitediff, first_order, second_order, fourth_order
from ..smooth_finite_difference import kerneldiff, mediandiff, meandiff, gaussiandiff, friedrichsdiff, butterdiff
from ..polynomial_fit import polydiff, savgoldiff, splinediff
from ..basis_fit import spectraldiff, rbfdiff
from ..total_variation_regularization import tvrdiff, velocity, acceleration, jerk, iterative_velocity, smooth_acceleration, jerk_sliding
from ..kalman_smooth import rtsdiff, constant_velocity, constant_acceleration, constant_jerk, robustdiff
from ..linear_model import lineardiff

# Map from method -> (search_space, bounds_low_hi)
method_params_and_bounds = {
    kerneldiff: ({'kernel': {'mean', 'median', 'gaussian', 'friedrichs'},
             'window_size': [5, 15, 30, 50],
          'num_iterations': [1, 5, 10]},
            {'window_size': (1, 1e6),
          'num_iterations': (1, 100)}),
    meandiff: ({'window_size': [5, 15, 30, 50], # Deprecated method
             'num_iterations': [1, 5, 10]},
               {'window_size': (1, 1e6),
             'num_iterations': (1, 100)}),
    butterdiff: ({'filter_order': set(i for i in range(1,11)), # categorical to save us from doing double work by guessing between orders
                   'cutoff_freq': [0.0001, 0.001, 0.005, 0.01, 0.1, 0.5],
                'num_iterations': [1, 5, 10]},
                  {'cutoff_freq': (1e-4, 1-1e-2),
                'num_iterations': (1, 1000)}),
    finitediff: ({'num_iterations': [5, 10, 30, 50],
                           'order': {2, 4}}, # order is categorical here, because it can't be 3
                 {'num_iterations': (1, 1000)}),
    first_order: ({'num_iterations': [5, 10, 30, 50]}, # Separated because optimizing over this one is rare due to shifted answer
                  {'num_iterations': (1, 1000)}),
    polydiff: ({'step_size': [1, 2, 5],
                   'kernel': {'friedrichs', 'gaussian'}, # categorical
                   'degree': [2, 3, 5, 7],
              'window_size': [11, 31, 51, 91, 131]},
               {'step_size': (1, 100),
                   'degree': (1, 8),
              'window_size': (10, 1000)}),
    savgoldiff: ({'degree': [2, 3, 5, 7, 10],
             'window_size': [3, 10, 30, 50, 90, 130, 200, 300],
           'smoothing_win': [3, 10, 30, 50, 90, 130, 200, 300]},
                 {'degree': (1, 12),
             'window_size': (3, 1000),
           'smoothing_win': (3, 1000)}),
    splinediff: ({'degree': {3, 4, 5}, # categorical, because degree is whole number, and there aren't many choices
                       's': [0.2, 0.5, 0.75, 0.9, 1, 10],
          'num_iterations': [1, 5, 10]},
                      {'s': (1e-2, 1e4),
          'num_iterations': (1, 10)}),
    spectraldiff: ({'even_extension': {True, False}, # give categorical params in a set
                  'pad_to_zero_dxdt': {True, False},
                  'high_freq_cutoff': [1e-3, 5e-2, 1e-2, 5e-2, 1e-1]}, # give numerical params in a list to scipy.optimize over them
                 {'high_freq_cutoff': (1e-5, 1-1e-5)}),
    rbfdiff: ({'sigma': [1e-2, 1e-1, 1],
                'lmbd': [1e-3, 1e-2, 1e-1]},
              {'sigma': (1e-2, 1e3),
                'lmbd': (1e-3, 0.5)}),
    tvrdiff: ({'gamma': [1e-2, 1e-1, 1, 10, 100, 1000],
               'order': {1, 2, 3}}, # warning: order 1 hacks the loss function when tvgamma is used, tends to win but is usually suboptimal choice in terms of true RMSE
              {'gamma': (1e-4, 1e7)}),
    velocity: ({'gamma': [1e-2, 1e-1, 1, 10, 100, 1000]}, # Deprecated method
               {'gamma': (1e-4, 1e7)}),
    iterative_velocity: ({'scale': 'small', # Rare to optimize this one, because it's longer-running than convex version
                 'num_iterations': [1, 5, 10],
                          'gamma': [1e-2, 1e-1, 1, 10, 100, 1000]},
                {'num_iterations': (1, 100), # gets expensive with more iterations
                          'gamma': (1e-4, 1e7)}),
    smooth_acceleration: ({'gamma': [1e-2, 1e-1, 1, 10, 100, 1000],
                     'window_size': [3, 10, 30, 50, 90, 130]},
                          {'gamma': (1e-4, 1e7),
                     'window_size': (3, 1000)}),
    rtsdiff: ({'forwardbackward': {True, False},
                         'order': {1, 2, 3}, # for this few options, the optimization works better if this is categorical
                  'log_qr_ratio': [float(k) for k in range(-9, 10, 2)] + [12, 16]},
                 {'log_qr_ratio': (-10, 20)}), # qr_ratio is usually >>1
    constant_velocity: ({'q': [1e-8, 1e-4, 1e-1, 1e1, 1e4, 1e8], # Deprecated method
                         'r': [1e-8, 1e-4, 1e-1, 1e1, 1e4, 1e8],
           'forwardbackward': {True, False}},
                        {'q': (1e-10, 1e10),
                         'r': (1e-10, 1e10)}),
    robustdiff: ({'order': {1, 2, 3}, # warning: order 1 hacks the loss function when tvgamma is used, tends to win but is usually suboptimal choice in terms of true RMSE
                  'log_q': [1., 4, 7, 10, 13], # decimal after first entry ensure this is treated as float type
                  'log_r': [-1., 2, 5, 8, 11],
            'proc_huberM': [0., 2, 6], # 0 is l1 norm, 1.345 is Huber 95% "efficiency", 2 assumes about 5% outliers,
            'meas_huberM': [0., 2, 6]}, # 6 assumes basically no outliers per outlier_portion = (1 - norm.cdf(M))*2
                 {'log_q': (-5, 16),
                  'log_r': (-5, 16),
            'proc_huberM': (0, 6),
            'meas_huberM': (0, 6)}),
    lineardiff: ({'kernel': 'gaussian',
                   'order': 3,
                   'gamma': [1e-1, 1, 10, 100],
             'window_size': [10, 30, 50, 90, 130]},
                  {'order': (1, 5),
                   'gamma': (1e-3, 1000),
             'window_size': (15, 1000)})
} # Methods with nonunique parameter sets are aliased in the dictionary below
for method in [second_order, fourth_order]: # Deprecated, redundant methods
    method_params_and_bounds[method] = method_params_and_bounds[first_order]
for method in [mediandiff, gaussiandiff, friedrichsdiff]: # Deprecated methods
    method_params_and_bounds[method] = method_params_and_bounds[meandiff]
for method in [acceleration, jerk]: # Deprecated, redundant methods
    method_params_and_bounds[method] = method_params_and_bounds[velocity]
method_params_and_bounds[jerk_sliding] = method_params_and_bounds[smooth_acceleration]
for method in [constant_acceleration, constant_jerk]: # Deprecated, redundant methods
    method_params_and_bounds[method] = method_params_and_bounds[constant_velocity]


# This function has to be at the top level for multiprocessing but is only used by optimize.
def _objective_function(point, func, x, dt, singleton_params, categorical_params, search_space_types,
    dxdt_truth, metric, tvgamma, padding, cache, huberM):
    """Function minimized by scipy.optimize.minimize, needs to have the form: (point, *args) -> float
    This is mildly complicated, because "point" controls the settings of a differentiation function, but
    the method may have numerical and non-numerical parameters, and all such parameters are now passed by
    keyword arguments. So the encoded `point` has to be decoded to dict.

    :param np.array point: a numerical vector scipy chooses to try in the objective function
    :param dict singleton_params: maps parameter names to singleton values
    :param dict categorical_params: maps parameter names to values
    :param dict search_space_types: maps parameter names to types, for turning float vector point into dict point_params
    :param multiprocessing.manager.dict cache: available across processes to save results and work
    Other params documented in `optimize`

    :return: float, cost of this objective at the point
    """
    # Short circuit if this hyperparam combo has already been queried, ~10% savings per #160
    key = sha1((''.join(f"{v:.3e}" for v in point) + # This hash is stable across processes. Takes bytes
               ''.join(str(v) for k,v in sorted(categorical_params.items()))).encode()).digest()
    if key in cache: return cache[key]

    # Query the differentiation method at this choice of hyperparameters
    point_params = {k:(v if search_space_types[k] == float else int(np.round(v)))
                        for k,v in zip(search_space_types, point)} # point -> dict
    # add back in singletons and categorical choices the Nelder-Mead isn't searching over
    try: x_hat, dxdt_hat = func(x, dt, **point_params, **singleton_params, **categorical_params) # estimate x and dxdt
    except np.linalg.LinAlgError: cache[key] = 1e10; return 1e10 # some methods can fail numerically

    # Evaluate estimate according to a loss function
    if dxdt_truth is not None:
        if metric == 'rmse': # minimize ||dxdt_hat - dxdt_truth||_2
            rmse_dxdt = evaluate.rmse(dxdt_truth, dxdt_hat, padding=padding)
            cache[key] = rmse_dxdt; return rmse_dxdt
        elif metric == 'error_correlation':
            ec = evaluate.error_correlation(dxdt_truth, dxdt_hat, padding=padding)
            cache[key] = ec; return ec
    else: # then minimize L(Phi) = (RMSE(trapz(dxdt_hat) + c - x) || sqrt{2*Mean(Huber((trapz(dxdt_hat) + c - x)/sigma, M))}*sigma) + gamma*TV(dxdt_hat)
        # It seems like we should be able to use x_hat rather than the trapz integral of dxdt_hat + constant, but the latter is more reliable,
        # because it accounts for the accuracy of the derivative directly, not through the generating algorithm's smooth signal estimate.
        rec_x_hat = utility.integrate_dxdt_hat(dxdt_hat, dt)
        rec_x_hat += utility.estimate_integration_constant(x, rec_x_hat, M=huberM)
        # rubust_rme(,M=inf) = rmse(), so just use the simpler function if M=inf
        cost = evaluate.rmse(x, rec_x_hat, padding=padding) if huberM == float('inf') else evaluate.robust_rme(x, rec_x_hat, padding=padding, M=huberM)
        cost += tvgamma*evaluate.total_variation(dxdt_hat, padding=padding)
        cache[key] = cost; return cost


def optimize(func, x, dt, dxdt_truth=None, tvgamma=1e-2, search_space_updates={}, metric='rmse',
    padding=0, opt_method='Nelder-Mead', maxiter=10, parallel=True, huberM=6):
    """Find the optimal hyperparameters for a given differentiation method.

    :param function func: differentiation method to optimize parameters for, e.g. kalman_smooth.rtsdiff
    :param np.array[float] x: data to differentiate
    :param float dt: step size
    :param np.array[float] dxdt_truth: actual time series of the derivative of x, if known
    :param float tvgamma: Only used if :code:`dxdt_truth` is *not* given. Regularization value used to select for parameters
                    that yield a smooth derivative. Larger value results in a smoother derivative.
    :param dict search_space_updates: Each method has a default search space of parameter settings, structured as
                    :code:`{param1:[numerical, values], param2:{categorical, values}, param3:value, ...}` (defined in
                    :code:`_optimize.py`). The Cartesian product of values are used as initial starting points in optimization.
                    If left None, the default search space is used, if :code:`{param1:[different,values]}`, these are applied.
    :param str metric: either :code:`'rmse'` or :code:`'error_correlation'`, only applies if :code:`dxdt_truth` is given
    :param int padding: number of steps to ignore at the beginning and end of the data series, or :code:`'auto'` to ignore
                    2.5% at each end. Larger value causes the optimization to emphasize the accuracy in the series middle.
    :param str opt_method: Optimization technique used by :code:`scipy.minimize`, the workhorse
    :param int maxiter: passed down to :code:`scipy.minimize`, maximum iterations
    :param bool parallel: whether to use multiple processes to optimize, typically faster for single optimizations.
                    For experiments, it is often a better use of resources to parallelize at that level, meaning
                    each must run in its own process, since spawned processes are not allowed to further spawn.
    :param float huberM: For ground-truth-less situation, if :math:`M < \\infty`, use outlier-robust, Huber-based accuracy
                    metric in objective. :math:`M` is in units akin to standard deviation (see :code:`evaluate.robust_rme`),
                    so transition from quadratic to linear regime for errors lying :math:`>\\!M\\sigma` away from mean error.

    :return: - **opt_params** (dict) -- best parameter settings for the differentation method
             - **opt_value** (float) -- lowest value found for objective function
    """
    if metric not in ['rmse','error_correlation']:
        raise ValueError('`metric` should either be `rmse` or `error_correlation`.')

    default_search_space, bounds = method_params_and_bounds[func]
    search_space = {**default_search_space, **search_space_updates} # applies updates without mutating default

    # No need to optimize over singletons, just pass them through
    singleton_params = {k:v for k,v in search_space.items() if not isinstance(v, (list, set))}

    # To handle categoricals, find their combination, and then pass each set individually
    categorical_params = {k for k,v in search_space.items() if isinstance(v, set)}
    categorical_combos = [dict(zip(categorical_params, combo)) for combo in
        product(*[search_space[k] for k in categorical_params])] # ends up [{}] if there are no categorical params

    # The Nelder-Mead's search space is the dimensions where multiple numerical options are given in a list
    search_space_types = {k:type(v[0]) for k,v in search_space.items() if isinstance(v, list)} # map param name -> type, for converting to and from point
    if any(v not in [float, int] for v in search_space_types.values()):
        raise ValueError("To optimize over categorical strings or bools, put them in a tuple, not a list.")
    # Cast ints to floats, and we're good to go
    starting_points = list(product(*[np.array(search_space[k]).astype(float) for k in search_space_types]))
    # The numerical space should have bounds
    bounds = [bounds[k] if k in bounds else # pass these to minimize(). It should respect them.
            None for k,v in search_space_types.items()] # None means no bound on a dimension

    results = []
    filterwarnings("ignore", '', UserWarning) # An extra filtering call, because some worker work can actually be done in the main process
    if parallel:
        with Manager() as manager:
            cache = manager.dict() # cache answers to avoid expensive repeat queries
            with Pool(initializer=filterwarnings, initargs=["ignore", '', UserWarning]) as pool: # The heavy lifting
                for categorical_combo in categorical_combos:
                    # wrap the objective and scipy.optimize.minimize to pass kwargs and a host of other things that remain the same
                    _obj_fun = partial(_objective_function, func=func, x=x, dt=dt, singleton_params=singleton_params,
                        categorical_params=categorical_combo, search_space_types=search_space_types, dxdt_truth=dxdt_truth,
                        metric=metric, tvgamma=tvgamma, padding=padding, cache=cache, huberM=huberM)
                    _minimize = partial(scipy.optimize.minimize, _obj_fun, method=opt_method, bounds=bounds, options={'maxiter':maxiter})
                    results += pool.map(_minimize, starting_points) # returns a bunch of OptimizeResult objects
    else: # For experiments, where I want to parallelize optimization calls and am not allowed to have each spawn further processes
        cache = dict()
        for categorical_combo in categorical_combos:
            _obj_fun = partial(_objective_function, func=func, x=x, dt=dt, singleton_params=singleton_params,
                categorical_params=categorical_combo, search_space_types=search_space_types, dxdt_truth=dxdt_truth,
                metric=metric, tvgamma=tvgamma, padding=padding, cache=cache, huberM=huberM)
            _minimize = partial(scipy.optimize.minimize, _obj_fun, method=opt_method, bounds=bounds, options={'maxiter':maxiter})
            results += [_minimize(p) for p in starting_points]

    opt_idx = np.nanargmin([r.fun for r in results])
    opt_point = results[opt_idx].x
    # results are going to be floats, but that may not be allowed, so convert back to a dict
    opt_params = {k:(v if search_space_types[k] == float else 
                    int(np.round(v)) if search_space_types[k] == int else
                    v > 0.5) for k,v in zip(search_space_types, opt_point)}
    opt_params.update(singleton_params)
    opt_params.update(categorical_combos[opt_idx//len(starting_points)]) # there are |starting_points| results for each combo

    return opt_params, results[opt_idx].fun


def suggest_method(x, dt, dxdt_truth=None, cutoff_frequency=None):
    """This is meant as an easy-to-use, automatic way for users with some time on their hands to determine
    a good method and settings for their data. It calls the optimizer over (almost) all methods in the repo
    using default search spaces defined at the top of the :code:`pynumdiff/optimize/_optimize.py` file.
    This routine will take a few minutes to run, especially due to `robustdiff`.
    
    Excluded:
        - ``first_order``, because iterating causes drift
        - ``lineardiff``, ``iterative_velocity``, and ``jerk_sliding``, because they either take too long,
          can be fragile, or tend not to do best
        - all ``cvxpy``-based methods if it is not installed
        - ``velocity`` because it tends to not be best but dominates the optimization process by directly
          optimizing the second term of the metric :math:`L = \\text{RMSE} \\Big( \\text{trapz}(\\mathbf{
          \\hat{\\dot{x}}}(\\Phi)) + \\mu, \\mathbf{y} \\Big) + \\gamma \\Big({TV}\\big(\\mathbf{\\hat{
          \\dot{x}}}(\\Phi)\\big)\\Big)`

    :param np.array[float] x: data to differentiate
    :param float dt: step size, because most methods are not designed to work with variable step sizes
    :param np.array[float] dxdt_truth: if known, you can pass true derivative values; otherwise you must use
            :code: `cutoff_frequency`
    :param float cutoff_frequency: in Hz, the highest dominant frequency of interest in the signal,
            used to find parameter :math:`\\gamma` for regularization of the optimization process
            in the absence of ground truth. See https://ieeexplore.ieee.org/document/9241009.
            Estimate by (a) counting real number of peaks per second in the data, (b) looking at
            power spectrum and choosing a cutoff, or (c) making an educated guess.

    :return: tuple[callable, dict] of\n
            - **method** -- a reference to the function handle of the differentiation method that worked best
            - **opt_params** -- optimal parameter settings for the differentation method
    """
    tvgamma = None
    if dxdt_truth is None: # parameter checking
        if cutoff_frequency is None:
            raise ValueError('Either dxdt_truth or cutoff_frequency must be provided.')
        tvgamma = np.exp(-1.6*np.log(cutoff_frequency) -0.71*np.log(dt) - 5.1) # See https://ieeexplore.ieee.org/document/9241009

    methods = [meandiff, mediandiff, gaussiandiff, friedrichsdiff, butterdiff,
        polydiff, savgoldiff, splinediff, spectraldiff, rbfdiff, finitediff, rtsdiff]
    try: # optionally skip some methods
        import cvxpy
        methods += [tvrdiff, smooth_acceleration, robustdiff]
    except ImportError:
        warn("CVXPY not installed, skipping tvrdiff, smooth_acceleration, and robustdiff")

    best_value = float('inf') # core loop
    for func in tqdm(methods):
        p, v = optimize(func, x, dt, dxdt_truth=dxdt_truth, tvgamma=tvgamma, search_space_updates=(
            {'order':{2,3}} if func in [tvrdiff, robustdiff] else {})) # convex-based with order 1 hack the cost function
        if v < best_value:
            method = func
            best_value = v
            opt_params = p

    return method, opt_params
