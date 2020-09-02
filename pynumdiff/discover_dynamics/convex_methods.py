import copy
import numpy as np
import sympy as sp
import scipy.sparse
import matplotlib.pyplot as plt
import cvxpy

def __vectorize__(M):
    '''
    Converts M (a n by k matrix) into a colum of n*k height,
    where each column of M is stacked on top of the other.

    Parameters
    ----------

    M : 2D numpy Matrix or Array [n, k]

    Returns
    -------

    vec : 1D numpy matrix [n*k, 1]

    '''
    vec = M[:,0]
    for i in range(1, M.shape[1]):
        vec = np.vstack((vec, M[:,i]))
    return vec

def __devectorize__(x, shape):
    '''
    Invert the function vectorize.

    Parameters
    ----------

    x : 1D numpy Matrix or Array [n*k, 1]

    shape : desired shape [n, k]

    Returns
    -------

    A_rec : reshaped matrix

    '''
    A_rec = x.reshape([shape[0],shape[1]])
    return A_rec

def __construct_constraints__(A_known, epsilon=None):
    '''
    Set upper and lower bounds according to nan structure of A.

    Parameters
    ----------

    A_known : 2D numpy matrix [n, k]
              Elements that are known should be floats. 
              Elements that are not known (and need to be estimated) should be numpy.nan's

    epsilon : float or 2D numpy matrix with same shape as A_known

    Returns
    -------

    constrains_bool :   1D numpy array of bools [1, n*k]
                        1: constrained entry
                        0: unconstrained entry

    constraints_low :   1D numpy array of floats [n*k, 1]
                        nonzero: constrained value (equal to value in A_known - epsilon)
                        zero: unconstrained value

    constraints_high :  1D numpy array of floats [n*k, 1]
                        nonzero: constrained value (equal to value in A_known + epsilon)
                        zero: unconstrained value

    '''
    if epsilon is None:
        epsilon = 1e-10*np.ones_like(A_known)
    
    X_ = A_known.T
    x = __vectorize__(X_)

    C = np.zeros_like(A_known)
    d = np.zeros([len(x), 1])

    n = 0
    for r in range(A_known.shape[0]):
        for c in range(A_known.shape[1]):
            if np.isnan(A_known[r,c]):
                C[r,c] = 0
                d[n] = 0
            else:
                C[r,c] = 1
                d[n] = A_known[r,c]
            n += 1

    constraints_bool = np.ravel(C)

    if type(epsilon) is np.matrix and epsilon.shape == A_known.shape:
        epsilon = __vectorize__(epsilon)

    constraints_low = d - epsilon
    constraints_high = d + epsilon

    return constraints_bool, constraints_low, constraints_high

def solve_for_A_given_X_Y_A_Known(A_known, X, Y, BUe, 
                                  gamma=0,
                                  whiten=False,
                                  epsilon=None,
                                  rows_of_interest='all',
                                  solver='MOSEK'):
    if epsilon is None:
        epsilon = 1e-10*np.ones_like(A_known)
    
    X = X + np.ones_like(X)*(np.random.random()-0.5)*1e-8 # need some noise otherwise get solver errors
    Y = Y + np.ones_like(Y)*(np.random.random()-0.5)*1e-8 # need some noise otherwise get solver errors

    k = X.shape[0] # number of states

    if rows_of_interest == 'all':
        rows_of_interest = np.arange(0,A_known.shape[0])

    # Transpose the system
    A_ = X.T
    Y_ = Y[rows_of_interest,:].T

    # Vectorize the system
    b = __vectorize__(Y_)
    eye = np.matrix( np.eye(len(rows_of_interest)) )
    A_k = scipy.sparse.kron( eye, A_)


    # define right number of variables
    x = cvxpy.Variable( (np.product(A_known[rows_of_interest,:].shape), 1) ) 

    # the weights here are for future flexibility, they are all just one right now
    weights = np.ones_like(X)
    weights_k = __vectorize__(weights[rows_of_interest,:])

    # We want Ax = b, so find the error. Could use a variety of norms, we use the Huber because it is fast and stable, 
    # but sort of like the one-norm
    error = cvxpy.square(A_k*x - b).T*weights_k / float(X.shape[1]) #* 100 

    # D is just used to add all the components of x together
    n_vars = np.product(A_known[rows_of_interest,:].shape)
    D = np.matrix(np.eye(n_vars))

    # Set up the objective function we want to solve
    penalty = gamma*cvxpy.norm(D*x, 1)/float(n_vars)
    obj = cvxpy.Minimize(error + penalty) 

    # constraints - these come from the known values of A
    constraints_bool, constraints_low, constraints_high = __construct_constraints__(
                                                          A_known[rows_of_interest,:],
                                                          epsilon[rows_of_interest,:])
    constraints = []
    for i, xi in enumerate(x):
        if constraints_bool[i] == 1:
            constraints.append(xi <= constraints_high[i])
            constraints.append(xi >= constraints_low[i])

    # define and solve the problem
    prob = cvxpy.Problem(obj, constraints)
    prob.solve(solver=solver)

    # reconstruct A
    A_rec_tmp = __devectorize__(np.matrix(x.value), A_known[rows_of_interest,:].shape)
    A_rec = A_known
    A_rec[rows_of_interest,:] = A_rec_tmp

    return A_rec, prob.value

def evaluate(A_rec, X, BUe, actual=None, row_check=0, plot=False):

    #state_rec = X[:,0]
    #for i in range(X.shape[1]-1):
    #    new_state = A_est*state_rec[:,-1] + BUe[:,i]
    #    state_rec = np.hstack((state_rec, new_state))
        
    state_rec = X[:,0]
    for i in range(X.shape[1]-1):
        new_state = A_rec*state_rec[:,-1] + BUe[:,i]
        state_rec = np.hstack((state_rec, new_state))

    if plot:        
        plt.plot(np.ravel(state_rec[row_check,:]))
        plt.plot(np.ravel(actual), '--')
    
    return np.abs(np.mean(X[row_check,1:] - state_rec[row_check,1:]))

def stack_states_and_controls(states, controls):
    stacked_states = np.vstack((states, controls))
    stacked_controls = np.vstack((np.zeros_like(states), controls))
    return stacked_states, stacked_controls

def unstack_states_and_controls(stacked_states, n_states):
    states = stacked_states[0:n_states, :]
    controls = stacked_states[n_states, :]
    return states, controls

def stack_A_B(A, B):
    AB = np.hstack((A, B))
    rows, cols = AB.shape
    AB = np.vstack((AB, np.zeros([cols-rows, cols])))
    return AB

def unstack_A_B(AB, n_states):
    A = AB[0:n_states, 0:n_states]
    B = AB[0:n_states, n_states:]
    return A, B


def run_convex(xhat_dmdc, controls_dmdc, A_known, B_known, rows_of_interest):
    n_states = xhat_dmdc.shape[0]
    n_controls = controls_dmdc.shape[0]
    
    X, BU = stack_states_and_controls(xhat_dmdc, controls_dmdc)
    X0 = X[:,0:-1]
    X1 = X[:,1:]
    BU = BU[:,1:]
    
    AB_known = stack_A_B(A_known, B_known)

    AB, prob_val = solve_for_A_given_X_Y_A_Known(AB_known, X0, X1, BU, 
                                                                   rows_of_interest=rows_of_interest)

    print(AB_known.shape, X0.shape, BU.shape)
    
    A, B = unstack_A_B(AB, n_states)
    
    return A, B