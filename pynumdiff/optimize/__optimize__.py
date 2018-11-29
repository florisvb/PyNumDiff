import scipy.optimize
import numpy as np

from pynumdiff.utils import utility as utility
from pynumdiff.utils import evaluate as evaluate
import pynumdiff.smooth_finite_difference

####################################################################################################################################################
# Documentation
####################################################################################################################################################

def docstring(x, dt, params=None, options={'iterate': False}, dxdt_truth=None, tvgamma=1e-2, padding=10):
    '''
    All optimization calls have the same interface. 

    Parameters
    ----------
    x       : (np.array of floats, 1xN) time series to differentiate
    dt      : (float) time step
    params  : (list, list of lists, or None)  Initial guess for params, see finite_difference.first_order
                                              If list: params is the initial condition
                                              If list of lists: params is a list of multiple initial conditions
                                              If None: default list of initial conditions is used (see code)
    options : (dict)  options, see finite_difference.first_order

    dxdt_truth : (like x) actual time series of the derivative of x, if known
    tvgamma     : (float)  regularization value used to select for parameters that yield a smooth derivative. 
                           larger value results in a smoother derivative
    padding     : (int)    number of time steps to ignore at the beginning and end of the time series in the optimization
                           larger value causes the optimization to emphasize the accuracy of dxdt in the middle of the time series

    Returns
    -------
    params : (list)  optimal parameters
    value  : (float) optimal value of objective function
    '''
    return None

####################################################################################################################################################
# Documentation
####################################################################################################################################################

def __objective_function__(params, *args):
    function, x, dt, params_types, params_low, params_high, options, dxdt_truth, tvgamma, padding = args[0]

    # keep param in bounds and correct type
    _params = []
    for p, param in enumerate(params):
        param = np.max([param, params_low[p]])
        param = np.min([param, params_high[p]])
        _params.append( params_types[p](param) ) 
    params = _params

    # estimate x and dxdt
    est_x, est_dxdt = function(x, dt, params, options)

    # evaluate estimate
    if dxdt_truth is not None: # then minimize ||est_dxdt - dxdt_truth||2
        rms_rec_x, rms_x, rms_dxdt = evaluate.metrics(x, dt, est_x, est_dxdt, actual_x=None, dxdt_truth=dxdt_truth, padding=padding)
        return rms_dxdt
    else: # then minimize [ || integral(est_dxdt) - noisy_x||2 + gamma*TV(est_dxdt) ]
        rms_rec_x, rms_x, rms_dxdt = evaluate.metrics(x, dt, est_x, est_dxdt, actual_x=None, dxdt_truth=None, padding=padding)
        return rms_rec_x + tvgamma*utility.total_variation(est_dxdt)

def __optimize__(params, args, method='Nelder-Mead'):
    # minimize with multiple initial conditions
    if type(params[0]) is list:
        opt_params = []
        opt_vals = []
        for paramset in params:
            result = scipy.optimize.minimize(__objective_function__, paramset, args=args, method=method, options={'maxiter': 10})
            opt_params.append(list(result.x.astype(int)))
            opt_vals.append(result.fun)
        idx = np.argmin(opt_vals)
        return opt_params[idx], opt_vals[idx]

    # minimize from single initial condition
    else:
        result = scipy.optimize.minimize(__objective_function__, params, args=args, method=method, options={'maxiter': 10})
        return list(result.x.astype(int)), result.fun
