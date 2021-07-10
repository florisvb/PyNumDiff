
"""
Optimization functions
"""
from multiprocessing import Pool
import scipy.optimize
import numpy as np

from pynumdiff.utils import utility
from pynumdiff.utils import evaluate


def __correct_params__(params, params_types, params_low, params_high):
    """
    :param params:
    :param params_types:
    :param params_low:
    :param params_high:
    :return:
    """
    _params = []
    for p, param in enumerate(params):
        param = params_types[p](param)
        param = np.max([param, params_low[p]])
        param = np.min([param, params_high[p]])
        param = params_types[p](param)
        _params.append(param)
    return _params


def __objective_function__(params, *args):
    """
    :param params:
    :param args:
    :return:
    """
    func, x, dt, params_types, params_low, params_high, options, dxdt_truth, tvgamma, padding, metric = args[0]

    # keep param in bounds and correct type
    params = __correct_params__(params, params_types, params_low, params_high)

    # estimate x and dxdt
    x_hat, dxdt_hat = func(x, dt, params, options)

    # evaluate estimate
    # pylint: disable=no-else-return
    if dxdt_truth is not None:  # then minimize ||dxdt_hat - dxdt_truth||2
        if metric == 'rmse':
            rms_rec_x, rms_x, rms_dxdt = evaluate.metrics(x, dt, x_hat, dxdt_hat, x_truth=None,
                                                          dxdt_truth=dxdt_truth, padding=padding)
            # print('rms_rec_x: ', rms_rec_x, 'tv x hat: ', utility.total_variation(x_hat))
            return rms_dxdt
        elif metric == 'error_correlation':
            error_correlation = evaluate.error_correlation(dxdt_hat, dxdt_truth, padding=padding)
            return error_correlation
        else:
            raise ValueError('metric should either be rmse or error_correlation!')
    else:  # then minimize [ || integral(dxdt_hat) - x||2 + gamma*TV(dxdt_hat) ]
        # print('Optimizing with [ || integral(dxdt_hat) - x||2 + gamma*TV(dxdt_hat) ]')
        rms_rec_x, rms_x, rms_dxdt = evaluate.metrics(x, dt, x_hat, dxdt_hat, x_truth=None,
                                                      dxdt_truth=None, padding=padding)

        return rms_rec_x + tvgamma*utility.total_variation(dxdt_hat[padding:-padding])


def __go__(input_args):
    """
    :param input_args:
    :return:
    """
    paramset, args, optimization_method, optimization_options, params_types, params_low, params_high = input_args
    try:
        result = scipy.optimize.minimize(__objective_function__, paramset, args=args, method=optimization_method,
                                         options=optimization_options)
        p = __correct_params__(result.x, params_types, params_low, params_high)
    except:
        return __correct_params__(paramset, params_types, params_low, params_high), 1000000000

    return p, result.fun


def __optimize__(params, args, optimization_method='Nelder-Mead', optimization_options={'maxiter': 20}):
    """
    Find the optimal parameters for a given differentiation method. This function gets called by optimize.METHOD_FAMILY.METHOD.
    For example, optimize.smooth_finite_difference.butterdiff will call this function to determine the optimal parameters for butterdiff.
    This function is a wrapper that parallelizes the __go__ function, which is a wrapper for __objective_function__.

    :param params: Initial guess for params, list of guesses, or None
    :type params: list, list of lists, or None

    :param args: list of the following: function, x, dt, params_types, params_low, params_high, options, dxdt_truth, tvgamma, padding, metric
                    - function      : function to optimize parameters for, i.e. optimize.smooth_finite_difference.butterdiff
                    - x             : (np.array of floats, 1xN) time series to differentiate
                    - dt            : (float) time step
                    - params_types  : (list of types) types for each parameter, i.e. [int, int], or [int, float, float]
                    - params_low    : (list) list of lowest allowable values for each parameter
                    - params_high   : (list) list of highest allowable values for each parameter
                    - options       : (dict)  options, see for example finite_difference.first_order
                    - dxdt_truth    : (like x) actual time series of the derivative of x, if known
                    - tvgamma       : (float)  regularization value used to select for parameters that yield a smooth derivative. Larger value results in a smoother derivative
                    - padding       : (int)    number of time steps to ignore at the beginning and end of the time series in the optimization. Larger value causes the optimization to emphasize the accuracy of dxdt in the middle of the time series
                    - metric        : (string) either 'rmse' or 'error_correlation', only applies if dxdt_truth is not None, see __objective_function__

    :type args: list -> (function reference, np.array, float, list, list, list, dict, np.array, float, int, string)

    :return: a tuple containing:
            - optimal parameters
            - optimal values of objective function
    :rtype: tuple -> (list, float)
    """
    func, x, dt, params_types, params_low, params_high, options, dxdt_truth, tvgamma, padding, metric = args
    if padding == 'auto':
        padding = max(int(0.025*len(x)), 1)
        args[-2] = padding

    # use_cpus = int(0.6*multiprocessing.cpu_count())

    # minimize with multiple initial conditions
    if not isinstance(params[0], list):
        params = [[p] for p in params]

    all_input_values = []
    for paramset in params:
        input_args = [paramset, args, optimization_method, optimization_options,
                      params_types, params_low, params_high]
        all_input_values.append(input_args)

    pool = Pool()
    result = pool.map(__go__, all_input_values)
    pool.close()
    pool.join()

    opt_params = []
    opt_vals = []
    for r in result:
        opt_params.append(r[0])
        opt_vals.append(r[1])

    opt_vals = np.array(opt_vals)
    opt_vals[np.where(np.isnan(opt_vals))] = 1000000000  # np.inf # avoid nans
    idx = np.argmin(opt_vals)
    opt_params = opt_params[idx]

    return list(opt_params), opt_vals[idx]
