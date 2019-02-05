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

def __correct_params__(params, params_types, params_low, params_high):
    _params = []
    for p, param in enumerate(params):
        param = np.max([param, params_low[p]])
        param = np.min([param, params_high[p]])
        _params.append( params_types[p](param) ) 
    return _params

def __objective_function__(params, *args):
    function, x, dt, params_types, params_low, params_high, options, dxdt_truth, tvgamma, padding = args[0]

    # keep param in bounds and correct type
    params = __correct_params__(params, params_types, params_low, params_high)

    # estimate x and dxdt
    x_hat, dxdt_hat = function(x, dt, params, options)

    #print(x_hat.shape, dxdt_hat.shape, params)

    # evaluate estimate
    if dxdt_truth is not None: # then minimize ||dxdt_hat - dxdt_truth||2
        rms_rec_x, rms_x, rms_dxdt = evaluate.metrics(x, dt, x_hat, dxdt_hat, x_truth=None, dxdt_truth=dxdt_truth, padding=padding)
        #print('rms_rec_x: ', rms_rec_x, 'tv x hat: ', utility.total_variation(x_hat))
        return rms_dxdt
    else: # then minimize [ || integral(dxdt_hat) - x||2 + gamma*TV(dxdt_hat) ]
        rms_rec_x, rms_x, rms_dxdt = evaluate.metrics(x, dt, x_hat, dxdt_hat, x_truth=None, dxdt_truth=None, padding=padding)

        #acc
        #try regularizing total sum of abs(acc) or jerk?

        return rms_rec_x + tvgamma*utility.total_variation(dxdt_hat)

def __optimize__(params, args, optimization_method='Nelder-Mead', optimization_options={'maxiter': 20}):
    function, x, dt, params_types, params_low, params_high, options, dxdt_truth, tvgamma, padding = args

    # minimize with multiple initial conditions
    if type(params[0]) is list:
        opt_params = []
        opt_vals = []
        for paramset in params:
            result = scipy.optimize.minimize(__objective_function__, paramset, args=args, method=optimization_method, options=optimization_options)
            p = __correct_params__(result.x, params_types, params_low, params_high)
            #print(p, result.fun)
            opt_params.append(p)
            opt_vals.append(result.fun)
        opt_vals = np.array(opt_vals)
        opt_vals[np.where(np.isnan(opt_vals))] = np.inf # avoid nans
        idx = np.argmin(opt_vals)
        opt_params = opt_params[idx]
        return list(opt_params), opt_vals[idx]

    # minimize from single initial condition
    else:
        result = scipy.optimize.minimize(__objective_function__, params, args=args, method=optimization_method, options=optimization_options)
        opt_params = __correct_params__(result.x, params_types, params_low, params_high)
        return list(opt_params), result.fun
