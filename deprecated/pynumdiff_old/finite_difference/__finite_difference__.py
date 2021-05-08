import numpy as np 

# included code
from pynumdiff.utils import utility

####################################################################################################################################################
# Finite Difference Methods
####################################################################################################################################################

def first_order(x, dt, params=[], options={}):
    '''
    First order centered finite difference.
    
    Inputs
    ------
    x       : (np.array of floats, 1xN) time series to differentiate
    dt      : (float) time step
    
    Parameters
    ----------
    params  : (list)  [iterations] : (int, optional)  number of iterations (ignored if 'iterate' not in options)
    options : (dict)  {'iterate'}  : (bool, optional) iterate the finite difference method (smooths the estimates)

    Outputs
    -------
    x_hat    : estimated (smoothed) x
    dxdt_hat : estimated derivative of x

    '''
    if 'iterate' in options:
        return __iterate_first_order__(x, dt, params)

    # Calculate the finite difference
    dxdt_hat = np.diff(x)/dt
    # Pad the data
    dxdt_hat = np.hstack((dxdt_hat[0], dxdt_hat, dxdt_hat[-1]))
    # Re-finite dxdt_hat using linear interpolation
    dxdt_hat = np.mean((dxdt_hat[0:-1], dxdt_hat[1:]), axis=0)

    return x, dxdt_hat

def second_order(x, dt, params=[], options={}):
    '''
    Second order centered finite difference.
    
    Inputs
    ------
    x       : (np.array of floats, 1xN) time series to differentiate
    dt      : (float) time step

    Parameters
    ----------
    params  : (list)  [] : ignored
    options : (dict)  {} : ignored

    Outputs
    -------
    x_hat    : estimated (smoothed) x
    dxdt_hat : estimated derivative of x

    '''
    dxdt_hat = (x[2:] - x[0:-2])/(2*dt)
    first_dxdt_hat = (-3*x[0] + 4*x[1] - x[2]) / (2*dt)
    last_dxdt_hat = (3*x[-1] - 4*x[-2] + x[-3]) / (2*dt)
    dxdt_hat = np.hstack((first_dxdt_hat, dxdt_hat, last_dxdt_hat))
    return x, dxdt_hat

def __x_hat_using_finite_difference__(x, dt):
    x_hat, dxdt_hat = first_order(x, dt)
    x_hat = utility.integrate_dxdt_hat(dxdt_hat, dt)
    x0 = utility.estimate_initial_condition(x, x_hat)
    return x_hat + x0

def __iterate_first_order__(x, dt, params):
    '''
    Iterate first order centered finite difference.
    
    Inputs
    ------
    x       : (np.array of floats, 1xN) time series to differentiate
    dt      : (float) time step
    
    Parameters
    ----------
    params  : (list)  [iterations] : (int) number of iterations

    Outputs
    -------
    x_hat    : estimated (smoothed) x
    dxdt_hat : estimated derivative of x

    '''
    if type(params) is list:
        iterations = params[0]
    else:
        iterations = params

    # set up weights
    w = np.arange(0,len(x),1)
    w = w/np.max(w)

    # forward backward passes
    for i in range(iterations):
        xf = __x_hat_using_finite_difference__(x, dt)
        xb = __x_hat_using_finite_difference__(x[::-1], dt)
        x = xf*w + xb[::-1]*(1-w)

    x_hat, dxdt_hat = first_order(x, dt)

    return x_hat, dxdt_hat
