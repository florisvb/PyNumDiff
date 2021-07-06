"""
functions for optimizing linear models
"""
import pynumdiff.linear_model
from pynumdiff.optimize.__optimize__ import __optimize__


def spectraldiff(x, dt, params=None, options={'even_extension': True, 'pad_to_zero_dxdt': True}, dxdt_truth=None,
                 tvgamma=1e-2, padding='auto', optimization_method='Nelder-Mead',
                 optimization_options={'maxiter': 10}, metric='rmse'):
    """
    Optimize the parameters for pynumdiff.linear_model.spectraldiff
    See pynumdiff.optimize.__optimize__ and pynumdiff.linear_model.spectraldiff for detailed documentation.
    """
    # initial condition
    if params is None:
        params = [[1e-3], [5e-2], [1e-2], [5e-2], [1e-1]]

    # param types and bounds
    params_types = [float, ]
    params_low = [1e-5, ]
    params_high = [1-1e-5, ]

    # optimize
    func = pynumdiff.linear_model.spectraldiff
    args = [func, x, dt, params_types, params_low, params_high, options, dxdt_truth, tvgamma, padding, metric]
    opt_params, opt_val = __optimize__(params, args, optimization_method=optimization_method,
                                       optimization_options=optimization_options)

    return opt_params, opt_val


def polydiff(x, dt, params=None, options={'sliding': True, 'step_size': 1, 'kernel_name': 'friedrichs'},
             dxdt_truth=None, tvgamma=1e-2, padding='auto', optimization_method='Nelder-Mead',
             optimization_options={'maxiter': 10}, metric='rmse'):
    """
    Optimize the parameters for pynumdiff.linear_model.polydiff
    See pynumdiff.optimize.__optimize__ and pynumdiff.linear_model.polydiff for detailed documentation.
    """
    # initial condition
    if params is None:
        orders = [2, 3, 5, 7]
        if options['sliding']:
            window_sizes = [10, 30, 50, 90, 130]
            params = []
            for order in orders:
                for window_size in window_sizes:
                    params.append([order, window_size])
        else:
            params = []
            for order in orders:
                params.append([order])

    # param types and bounds
    if options['sliding']:
        params_types = [int, int]
        params_low = [1, 10]
        params_high = [8, 1e3]
    else:
        params_types = [int, ]
        params_low = [1, ]
        params_high = [8, ]

    # optimize
    func = pynumdiff.linear_model.polydiff
    args = [func, x, dt, params_types, params_low, params_high, options, dxdt_truth, tvgamma, padding, metric]
    opt_params, opt_val = __optimize__(params, args, optimization_method=optimization_method,
                                       optimization_options=optimization_options)

    return opt_params, opt_val


def savgoldiff(x, dt, params=None, options={}, dxdt_truth=None, tvgamma=1e-2, padding='auto',
               optimization_method='Nelder-Mead', optimization_options={'maxiter': 10}, metric='rmse'):
    """
    Optimize the parameters for pynumdiff.linear_model.savgoldiff
    See pynumdiff.optimize.__optimize__ and pynumdiff.linear_model.savgoldiff for detailed documentation.
    """
    # initial condition
    if params is None:
        orders = [2, 3, 5, 7, 9, 11, 13]
        window_sizes = [3, 10, 30, 50, 90, 130, 200, 300]
        smoothing_wins = [3, 10, 30, 50, 90, 130, 200, 300]
        params = []
        for order in orders:
            for window_size in window_sizes:
                for smoothing_win in smoothing_wins:
                    params.append([order, window_size, smoothing_win])

    # param types and bounds
    params_types = [int, int, int]
    params_low = [1, 3, 3]
    params_high = [12, 1e3, 1e3]

    # optimize
    func = pynumdiff.linear_model.savgoldiff
    args = [func, x, dt, params_types, params_low, params_high, options, dxdt_truth, tvgamma, padding, metric]
    opt_params, opt_val = __optimize__(params, args, optimization_method=optimization_method,
                                       optimization_options=optimization_options)

    return opt_params, opt_val


def chebydiff(x, dt, params=None, options={'sliding': True, 'step_size': 1, 'kernel_name': 'friedrichs'},
              dxdt_truth=None, tvgamma=1e-2, padding='auto', optimization_method='Nelder-Mead',
              optimization_options={'maxiter': 10}, metric='rmse'):
    """
    Optimize the parameters for pynumdiff.linear_model.chebydiff
    See pynumdiff.optimize.__optimize__ and pynumdiff.linear_model.chebydiff for detailed documentation.
    """
    # initial condition
    if params is None:
        orders = [3, 5, 7, 9]
        if options['sliding']:
            window_sizes = [10, 30, 50, 90, 130]
            params = []
            for order in orders:
                for window_size in window_sizes:
                    params.append([order, window_size])
        else:
            params = []
            for order in orders:
                params.append([order])

    # param types and bounds
    if options['sliding']:
        params_types = [int, int]
        params_low = [1, 10]
        params_high = [10, 1e3]
    else:
        params_types = [int, ]
        params_low = [1, ]
        params_high = [10, ]

    # optimize
    func = pynumdiff.linear_model.chebydiff
    args = [func, x, dt, params_types, params_low, params_high, options, dxdt_truth, tvgamma, padding, metric]
    opt_params, opt_val = __optimize__(params, args, optimization_method=optimization_method,
                                       optimization_options=optimization_options)

    return opt_params, opt_val


def lineardiff(x, dt, params=None, options={'sliding': True, 'step_size': 10, 'kernel_name': 'gaussian'},
               dxdt_truth=None, tvgamma=1e-2, padding='auto', optimization_method='Nelder-Mead',
               optimization_options={'maxiter': 10}, metric='rmse'):
    """
    Optimize the parameters for pynumdiff.linear_model.lineardiff
    See pynumdiff.optimize.__optimize__ and pynumdiff.linear_model.lineardiff for detailed documentation.
    """
    # initial condition
    if params is None:
        Ns = [3]
        gammas = [1e-1, 1, 10, 100]
        if options['sliding']:
            window_sizes = [10, 30, 50, 90, 130]
            params = []
            for N in Ns:
                for gamma in gammas:
                    for window_size in window_sizes:
                        params.append([N, gamma, window_size])
        else:
            params = []
            for N in Ns:
                for gamma in gammas:
                    params.append([N, gamma])

    # param types and bounds
    if options['sliding']:
        params_types = [int, float, int]
        params_low = [3, 1e-3, 15]
        params_high = [3, 1e3, 1e3]
    else:
        params_types = [int, float, ]
        params_low = [3, 1e-3, ]
        params_high = [3, 1e3, ]

    # optimize
    func = pynumdiff.linear_model.lineardiff
    args = [func, x, dt, params_types, params_low, params_high, options, dxdt_truth, tvgamma, padding, metric]
    opt_params, opt_val = __optimize__(params, args, optimization_method=optimization_method,
                                       optimization_options=optimization_options)

    return opt_params, opt_val
