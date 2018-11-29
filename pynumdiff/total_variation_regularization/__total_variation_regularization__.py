import numpy as np 
import time
import copy
import math

try:
    import cvxpy
except:
    warnings.warn('Could not import cvxpy. Install cvxpy (http://www.cvxpy.org/install/index.html) to use total variation regularized derivatives. \
                   Recommended solver: MOSEK, free academic license available: https://www.mosek.com/products/academic-licenses/')

# Generalized total variation regularized derivatives
def __total_variation_regularized_derivative__(x, dt, N, gamma, solver='MOSEK'):
    '''
    Use convex optimization (cvxpy) to solve for the Nth total variation regularized derivative.
    Default solver is MOSEK: https://www.mosek.com/

    Inputs
    ------
    x       : (np.array of floats, 1xN) time series to differentiate
    dt      : (float) time step

    Parameters
    ----------
    N       : (int) 1, 2, or 3, the Nth derivative to regularize
    gamma   : (float) regularization parameter
    solver  : (string) Solver to use. Solver options include: 'MOSEK' and 'CVXOPT', 
                                      in testing, 'MOSEK' was the most robust.

    Outputs
    -------
    x_hat    : estimated (smoothed) x
    dxdt_hat : estimated derivative of x

    '''
    # Define the variables for the highest order derivative and the integration constants
    var = cvxpy.Variable( len(x)+N )

    # Recursively integrate the highest order derivative to get back to the position
    derivatives = [var[N:]]
    for i in range(N):
        d = cvxpy.cumsum(derivatives[-1]) + var[i]
        derivatives.append(d)
    
    # Compare the recursively integration position to the noisy position
    sum_squared_error = cvxpy.sum_squares(derivatives[-1] - x)

    # Total variation regularization on the highest order derivative
    r = cvxpy.sum( gamma*cvxpy.tv(derivatives[0]) )
    
    # Set up and solve the optimization problem
    obj = cvxpy.Minimize(sum_squared_error + r)
    prob = cvxpy.Problem(obj)
    prob.solve(solver=solver)

    # Recursively calculate the value of each derivative 
    final_derivative = var.value[N:]
    derivative_values = [final_derivative]
    for i in range(N):
        d = np.cumsum(derivative_values[-1]) + var.value[i]
        derivative_values.append(d)
    for i in range(len(derivative_values)):
        derivative_values[i] = derivative_values[i]/(dt**(N-i))
    
    # Extract the velocity and smoothed position
    dxdt_hat = derivative_values[-2]
    x_hat = derivative_values[-1]
        
    return x_hat, dxdt_hat



def velocity(x, dt, params, options={'solver': 'MOSEK'}):
    '''
    Use convex optimization (cvxpy) to solve for the velocity total variation regularized derivative.
    Default solver is MOSEK: https://www.mosek.com/
    
    Inputs
    ------
    x       : (np.array of floats, 1xN) time series to differentiate
    dt      : (float) time step

    Parameters
    ----------
    params  : (list) [gamma], where gamma (float) is the regularization parameter
                     or if 'iterate' in options: [gamma, num_iterations]
    options : (dict) {'solver': SOLVER} SOLVER options include: 'MOSEK' and 'CVXOPT', 
                                        in testing, 'MOSEK' was the most robust.

    Outputs
    -------
    x_hat    : estimated (smoothed) x
    dxdt_hat : estimated derivative of x

    '''
    if type(params) is list:
        gamma = params[0]
    else:
        gamma = params

    return __total_variation_regularized_derivative__(x, dt, 1, gamma, solver=options['solver'])

def acceleration(x, dt, params, options={'solver': 'MOSEK'}):
    '''
    Use convex optimization (cvxpy) to solve for the acceleration total variation regularized derivative.
    Default solver is MOSEK: https://www.mosek.com/
    
    Inputs
    ------
    x       : (np.array of floats, 1xN) time series to differentiate
    dt      : (float) time step

    Parameters
    ----------
    params  : (list) [gamma], where gamma (float) is the regularization parameter
                     or if 'iterate' in options: [gamma, num_iterations]
    options : (dict) {'solver': SOLVER} SOLVER options include: 'MOSEK' and 'CVXOPT', 
                                        in testing, 'MOSEK' was the most robust.

    Outputs
    -------
    x_hat    : estimated (smoothed) x
    dxdt_hat : estimated derivative of x

    '''
    if type(params) is list:
        gamma = params[0]
    else:
        gamma = params

    return __total_variation_regularized_derivative__(x, dt, 2, gamma, solver=options['solver'])

def jerk(x, dt, params, options={'solver': 'MOSEK'}):
    '''
    Use convex optimization (cvxpy) to solve for the jerk total variation regularized derivative.
    Default solver is MOSEK: https://www.mosek.com/
    
    Inputs
    ------
    x       : (np.array of floats, 1xN) time series to differentiate
    dt      : (float) time step

    Parameters
    ----------
    params  : (list) [gamma], where gamma (float) is the regularization parameter
                     or if 'iterate' in options: [gamma, num_iterations]
    options : (dict) {'solver': SOLVER} SOLVER options include: 'MOSEK' and 'CVXOPT', 
                                        in testing, 'MOSEK' was the most robust.

    Outputs
    -------
    x_hat    : estimated (smoothed) x
    dxdt_hat : estimated derivative of x

    '''
    if type(params) is list:
        gamma = params[0]
    else:
        gamma = params

    return __total_variation_regularized_derivative__(x, dt, 3, gamma, solver=options['solver'])


