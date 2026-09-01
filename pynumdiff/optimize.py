"""Optimization functionality"""
from itertools import product
from functools import partial
from warnings import catch_warnings, filterwarnings, warn
from multiprocessing import Pool, Manager
import scipy.optimize
import numpy as np
from tqdm import tqdm

from .utils import evaluate, utility
from .finite_difference import finitediff
from .smooth_finite_difference import kerneldiff, butterdiff
from .polynomial_fit import polydiff, savgoldiff, splinediff
from .basis_fit import spectraldiff, rbfdiff, waveletdiff
from .total_variation_regularization import tvrdiff, iterative_velocity, smooth_acceleration
from .kalman_smooth import rtsdiff, robustdiff
from .linear_model import lineardiff

# Map from method -> (search_space, bounds_low_hi)
method_params_and_bounds = {
    kerneldiff: ({'kernel': {'mean', 'median', 'gaussian', 'friedrichs'},
             'window_size': [5, 15, 31, 51],
          'num_iterations': [1, 5, 10]},
            {'window_size': (1, 1e6, 'odd'), # an even-width kernel has no center, so convolving with it
          'num_iterations': (1, 100)}),         # shifts the signal half a sample before differencing
    butterdiff: ({'filter_order': set(i for i in range(1,11)), # categorical to save us from doing double work by guessing between orders
                   'high_freq_cutoff': [0.0001, 0.001, 0.005, 0.01, 0.1, 0.5],
                'num_iterations': [1, 5, 10]},
                  {'high_freq_cutoff': (1e-4, 1-1e-2),
                'num_iterations': (1, 1000)}),
    finitediff: ({'num_iterations': [5, 10, 30, 50],
                           'order': {2, 4}}, # order is categorical here, because it can't be 3
                 {'num_iterations': (1, 1000)}),
    polydiff: ({'step_size': [1, 2, 5],
                   'kernel': {'friedrichs', 'gaussian'}, # categorical
                   'degree': [2, 3, 5, 7],
              'window_size': [11, 31, 51, 91, 131]},
               {'step_size': (1, 100),
                   'degree': (1, 8),
              'window_size': (10, 1000, 'odd')}),
    savgoldiff: ({'degree': [2, 3, 5, 7, 10],
             'window_size': [3, 11, 31, 51, 91, 131, 201, 301],
           'smoothing_win': [3, 11, 31, 51, 91, 131, 201, 301]},
                 {'degree': (1, 12),
             'window_size': (3, 1000, 'odd'), # savgol_filter accepts even windows but silently returns a
           'smoothing_win': (3, 1000, 'odd')}), # derivative evaluated half a sample late
    splinediff: ({'degree': {3, 4, 5}, # categorical, because degree is whole number, and there aren't many choices
                       's': [0.5, 1, 1.5], # multiples of the noise energy, so these hold at any N and noise level
          'num_iterations': [1, 3]}, # 5 and 10 never won across 24 sim/step/noise combinations
                      {'s': (1e-1, 1e1), # a relative floor keeps the search out of the near-interpolating regime,
          'num_iterations': (1, 5)}), # where starving the budget while iterating made scipy's knot search fail
    spectraldiff: ({'even_extension': {True, False}, # give categorical params in a set
                  'pad_to_zero_dxdt': {True, False},
                  'high_freq_cutoff': [1e-2, 5e-2, 1e-1, 5e-1]}, # give numerical params in a list to scipy.optimize over them
                 {'high_freq_cutoff': (1e-3, 1-1e-5)}), # every cutoff below ~2/N keeps only the DC term, see #209
    rbfdiff: ({'sigma': [1e-2, 1e-1, 1],
                'lmbd': [1e-3, 1e-2, 1e-1]},
              {'sigma': (1e-2, 1e3),
                'lmbd': (1e-3, 0.5)}),
    waveletdiff: ({'wavelet': {'db8', 'db12', 'sym8', 'coif1'}, # different data can favor different mother wavelets
                 'threshold': [0.5, 1, 2]}, # multiplies the Donoho-Johnstone universal threshold, 0 meaning no denoising.
                 # `level` is left at its adaptive default, min(dwt_max_level(N, wavelet), 5), which tracks both the signal and filter lengths
                {'threshold': (0.1, 10)}),
    tvrdiff: ({'gamma': [1e-2, 1e-1, 1, 10, 100, 1000],
               'order': {1, 2, 3}, # warning: order 1 hacks the loss function when tvgamma is used, tends to win but is usually suboptimal choice in terms of true RMSE
              'huberM': [2., 6]}, # the scale of sigma is mad(x), which is bigger than mad(y-x) residuals, so outliers likely come at lower M values
              {'gamma': (1e-4, 1e7),
              'huberM': (2, 6)}), # huberM too low seeks sparse solutions, which hack the tvgamma loss function
    iterative_velocity: ({'scale': 'small', # Rare to optimize this one, because it's longer-running than convex version
                 'num_iterations': [1, 5, 10],
                          'gamma': [1e-2, 1e-1, 1, 10, 100, 1000]},
                {'num_iterations': (1, 100), # gets expensive with more iterations
                          'gamma': (1e-4, 1e7)}),
    smooth_acceleration: ({'gamma': [1e-2, 1e-1, 1, 10, 100, 1000],
                     'window_size': [3, 11, 31, 51, 91, 131]},
                          {'gamma': (1e-4, 1e7),
                     'window_size': (3, 1000, 'odd')}),
    rtsdiff: ({'forwardbackward': {True, False},
                         'order': {1, 2, 3}, # for this few options, the optimization works better if this is categorical
                  'log_qr_ratio': [float(k) for k in range(-9, 10, 2)] + [12, 16]},
                 {'log_qr_ratio': (-10, 20)}), # qr_ratio is usually >>1
    robustdiff: ({'order': {1, 2, 3}, # warning: order 1 hacks the loss function when tvgamma is used, tends to win but is usually suboptimal choice in terms of true RMSE
                  'log_q': [1., 4, 7, 10, 13], # decimal after first entry ensure this is treated as float type
                  'log_r': [0.], # one seed, but allowed to drift. Holding both Huber Ms fixed, the objective is flat along characteristic curves in the
                                 # (log_q, log_r) plane, but these are sloped such that varying log_q cuts across more of them than varying log_r; extra log_r
                                 # restarts mostly duplicate each other. 0 is the natural set point for log_r, accepting noise stddev estimates as true.
            'proc_huberM': [0., 2, 6], # 0 is l1 norm, 1.345 is Huber 95% "efficiency", 2 assumes about 5% outliers,
            'meas_huberM': [0., 2, 6]}, # 6 assumes basically no outliers per outlier_portion = (1 - norm.cdf(M))*2
                 {'log_q': (-1, 14), # outside these the fit saturates: log_q below 0 or log_r above 10 both mean
                  'log_r': (-4, 10), # ignore the measurements and trust the model, and give identical answers
            'proc_huberM': (0, 6),
            'meas_huberM': (0, 6)}),
    lineardiff: ({'kernel': 'gaussian', # `step_size` barely moves accuracy so now defaults to window_size//5
                   'order': {2, 3}, # order 1 never won across 12 sim/seed sweeps
                   'gamma': [1e-2, 1e-1, 1], # in units of the data's own scale since #222
             'window_size': [41, 81, 161]},
                  {'gamma': (1e-4, 1e1),
             'window_size': (11, 1000, 'odd')})
}


# How to round float coordinates from the minimizer into possibly-discrete values
ROUND = {float: lambda v: v,
           int: lambda v: int(np.round(v)),
         'odd': lambda v: 2*int(np.round((v - 1)/2)) + 1}


# This function has to be at the top level for multiprocessing but is only used by optimize.
def _objective_function(point, func, x, dt, singleton_params, categorical_params, roundings,
    dxdt_truth, metric, tvgamma, padding, cache, huberM):
    """Function minimized by scipy.optimize.minimize, needs to have the form: (point, *args) -> float
    This is mildly complicated, because "point" controls the settings of a differentiation function, but
    the method may have numerical and non-numerical parameters, and all such parameters are now passed by
    keyword arguments. So the encoded `point` has to be decoded to dict.

    :param np.array point: a numerical vector scipy chooses to try in the objective function
    :param dict singleton_params: maps parameter names to singleton values
    :param dict categorical_params: maps parameter names to values
    :param dict roundings: maps parameter names to their key in :code:`ROUND`, for turning the float vector point into a dict
    :param multiprocessing.manager.dict cache: available across processes to save results and work
    Other params documented in `optimize`

    :return: float, cost of this objective at the point
    """
    point_params = {k:ROUND[roundings[k]](v) for k,v in zip(roundings, point)} # point -> dict, rounding for discrete parameters

    # Short circuit if this hyperparam combo (with rounding applied) has already been queried, ~10% savings per #160.
    key = tuple([f"{v:.6e}" if isinstance(v, float) else str(v) for v in point_params.values()] +
            [str(v) for _,v in sorted(categorical_params.items())])
    if key in cache: return cache[key] # send tuple(strings) to the manager process to avoid salted hashes in child processes

    # Query the differentiation method at this choice of hyperparameters
    try: x_hat, dxdt_hat = func(x, dt, **point_params, **singleton_params, **categorical_params) # take deriv, add back singletons and categorical choices
    except np.linalg.LinAlgError: cache[key] = 1e10; return 1e10 # some methods can fail numerically

    # Evaluate estimate according to a loss function
    if dxdt_truth is not None:
        if metric == 'rmse': # minimize ||dxdt_hat - dxdt_truth||_2
            rmse_dxdt = evaluate.rmse(dxdt_truth, dxdt_hat, padding=padding)
            cache[key] = rmse_dxdt; return rmse_dxdt
        if metric == 'error_correlation':
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


def optimize(func, x, dt, dxdt_truth=None, cutoff_freq=None, search_space_updates={}, metric='rmse',
    padding=0, opt_method='Nelder-Mead', maxiter=10, parallel=True, huberM=6, tvgamma=None):
    """Find the optimal hyperparameters for a given differentiation method.

    :param function func: differentiation method to optimize parameters for, e.g. kalman_smooth.rtsdiff
    :param np.array[float] x: data to differentiate
    :param float dt: step size
    :param np.array[float] dxdt_truth: actual time series of the derivative of x, if known
    :param float cutoff_freq: Only used if :code:`dxdt_truth` is *not* given. The bandlimit of underlying signal, in Hz.
                    Estimate by counting peaks per second, or by reading where the power spectrum falls off. Lower -> smoother
    :param float tvgamma: (**deprecated**, prefer :code:`cutoff_freq`) the smoothness weight itself, which is harder to
                    interpret because its meaning depends on the sampling rate as well as the signal
    :param dict search_space_updates: Each method has a default search space of parameter settings, encoded as
                    :code:`{param1:[numerical, values], param2:{categorical, values}, param3:value, ...}` (defined at the top of
                    :code:`pynumdiff/optimize.py`). The Cartesian product of dictionary values serves as initialization points,
                    where values given as singletons or in sets are plugged into separate minimization runs (since some algos cannot
                    handle discrete search), and values given in lists serve as seeds for search across continuous dimensions.
                    This parameter optionally accepts a `dictionary update <https://docs.python.org/3/library/stdtypes.html#dict.update>`_
                    to override particular or multiple parameter values.
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
    if tvgamma is not None:
        warn("`tvgamma` will be removed in a future version. Use `cutoff_freq` instead, from which it is calculated.", DeprecationWarning)
    elif cutoff_freq is not None: # See https://ieeexplore.ieee.org/document/9241009
        tvgamma = np.exp(-1.6*np.log(cutoff_freq) - 0.71*np.log(dt) - 5.1)
    elif dxdt_truth is None: raise ValueError("Either `dxdt_truth` or `cutoff_freq` must be given.")
    if metric not in ['rmse','error_correlation']: raise ValueError('`metric` should either be `rmse` or `error_correlation`.')

    default_search_space, bounds = method_params_and_bounds[func]
    search_space = {**default_search_space, **search_space_updates} # applies updates without mutating default

    # No need to optimize over singletons, just pass them through
    singleton_params = {k:v for k,v in search_space.items() if not isinstance(v, (list, set))}

    # To handle categoricals, find their combination, and then pass each set individually
    categorical_params = {k for k,v in search_space.items() if isinstance(v, set)}
    categorical_combos = [dict(zip(categorical_params, combo)) for combo in
        product(*[search_space[k] for k in categorical_params])] # ends up [{}] if there are no categorical params

    # The minimization's search space is the dimensions where numerical options are given in a list
    roundings = {k:(bounds[k][2] if len(bounds.get(k, ())) > 2 else type(v[0])) # map param name -> its key into ROUND
        for k,v in search_space.items() if isinstance(v, list)}     # taken from the seeds' type unless the bounds name one
    if len(roundings) == 0 and len(categorical_combos) == 1: # one point is not much of a space
        warn(f"Nothing to optimize: every parameter of {func.__name__} is pinned to a single value, so the objective is simply evaluated there.")

    # Cast ints to floats for optimization, and set up value boundaries
    starting_points = list(product(*[np.array(search_space[k]).astype(float) for k in roundings]))
    bounds = [bounds[k][:2] if k in bounds else # pass these to minimize(). It should respect them.
            (None, None) for k,v in roundings.items()] # tuple of Nones means no bound on a dimension

    # Bind everything that stays the same across jobs, leaving `minimize`'s `fun` and `x0` args positional so one `partial` can serve them all.
    _minimize = partial(scipy.optimize.minimize, method=opt_method, bounds=bounds, options={'maxiter':maxiter})
    obj_kwargs = {'func':func, 'x':x, 'dt':dt, 'singleton_params':singleton_params, 'roundings':roundings,
        'dxdt_truth':dxdt_truth, 'metric':metric, 'tvgamma':tvgamma, 'padding':padding, 'huberM':huberM}

    with catch_warnings(action="ignore", category=UserWarning): # some worker work is done in main process; scoped so caller's filters restored after. See #206.
        obj_kwargs['cache'] = Manager().dict() if parallel else {} # a Manager's dict can be shared across processes; avoid repeat queries
        # Line up every (objective, starting point) pair in combo-major order, so the whole sweep can go out at once
        jobs = [(partial(_objective_function, categorical_params=categorical_combo, **obj_kwargs), point)
                for categorical_combo in categorical_combos for point in starting_points]

        if len(roundings) == 0: # no space for minimizer to walk, so just evaluate each starting point where it stands
            results = [scipy.optimize.OptimizeResult(x=point, fun=obj(point)) for obj, point in jobs] # list of opt results to match expected type
        elif parallel:
            with Pool(initializer=filterwarnings, initargs=["ignore", '', UserWarning]) as pool:
                results = pool.starmap(_minimize, jobs, chunksize=1) # the heavy lifting
        else: # For experiments, where I want to parallelize optimization calls and am not allowed to have each spawn further processes
            results = [_minimize(obj, point) for obj, point in jobs]

    opt_idx = np.nanargmin([r.fun for r in results])
    opt_point = results[opt_idx].x
    # results are going to be floats, but that may not be allowed, so convert back to a dict
    opt_params = {k:ROUND[roundings[k]](v) for k,v in zip(roundings, opt_point)} # same rounding the objective used, so we return what ran
    opt_params.update(singleton_params) # add back in the non-searched params
    opt_params.update(categorical_combos[opt_idx//len(starting_points)]) # there are |starting_points| results for each combo

    return opt_params, results[opt_idx].fun


def suggest_method(x, dt, dxdt_truth=None, cutoff_freq=None):
    """This is meant as an easy-to-use, automatic way for users with some time on their hands to determine
    a good method and settings for their data. It calls the optimizer over (almost) all methods in the repo
    using default search spaces defined in :code:`method_params_and_bounds` at the top of :code:`pynumdiff/optimize.py`.
    This routine will take a few minutes to run.
    
    Excluded:
        - ``iterative_velocity``, because it's mostly academic
        - all ``cvxpy``-based methods if it is not installed
        - first-order ``tvrdiff`` and ``robustdiff`` because they hack the optimization function by directly
          optimizing the second term of the metric :math:`L = \\text{RMSE} \\Big( \\text{trapz}(\\mathbf{
          \\hat{\\dot{x}}}(\\Phi)) + \\mu, \\mathbf{y} \\Big) + \\gamma \\Big({TV}\\big(\\mathbf{\\hat{
          \\dot{x}}}(\\Phi)\\big)\\Big)`

    :param np.array[float] x: data to differentiate
    :param float dt: step size, because most methods are not designed to work with variable step sizes
    :param np.array[float] dxdt_truth: if known, you can pass true derivative values; otherwise you must use
            :code: `cutoff_freq`
    :param float cutoff_freq: in Hz, the highest dominant frequency of interest in the signal,
            used to find parameter :math:`\\gamma` for regularization of the optimization process
            in the absence of ground truth. See https://ieeexplore.ieee.org/document/9241009.
            Estimate by (a) counting real number of peaks per second in the data, (b) looking at
            power spectrum and choosing a cutoff, or (c) making an educated guess.

    :return: tuple[callable, dict] of\n
            - **method** -- a reference to the function handle of the differentiation method that worked best
            - **opt_params** -- optimal parameter settings for the differentation method
    """
    methods = [kerneldiff, butterdiff, polydiff, savgoldiff, splinediff, spectraldiff, rbfdiff, waveletdiff, finitediff, rtsdiff]
    try: # optionally skip some methods
        import cvxpy
        methods += [tvrdiff, smooth_acceleration, robustdiff, lineardiff]
    except ImportError:
        warn("CVXPY not installed, skipping tvrdiff, smooth_acceleration, robustdiff, and lineardiff")

    best_value = float('inf') # core loop
    for func in tqdm(methods):
        p, v = optimize(func, x, dt, dxdt_truth=dxdt_truth, cutoff_freq=cutoff_freq, search_space_updates=(
            {'order':{2,3}} if func in [tvrdiff, robustdiff] else {})) # convex-based with order 1 hack the cost function
        if v < best_value:
            method = func
            best_value = v
            opt_params = p

    return method, opt_params
