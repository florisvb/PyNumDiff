import numpy as np
from warnings import warn

####################
# Helper functions #
####################
def _kalman_forward_filter(xhat0, P0, y, A, C, Q, R, u=None, B=None):
    """Run the forward pass of a Kalman filter
    :param np.array xhat0: initial estimate of state (often 0) one step before the first measurement
    :param np.array P0: initial guess of state covariance (often identity matrix) before the first measurement
    :param np.array y: series of measurements, stacked along axis 0. In this case just the raw noisy data (fully observable)
    :param np.array A: discrete-time state transition matrix
    :param np.array C: measurement matrix
    :param np.array Q: discrete-time process noise covariance
    :param np.array R: measurement noise covariance
    :param np.array u: optional control input
    :param np.array B: optional control matrix
    :return: tuple[np.array, np.array, np.array, np.array] of\n
        - **xhat_pre** -- a priori estimates of xhat, with axis=0 the batch dimension, so xhat[n] gets the nth step
        - **xhat_post** -- a posteriori estimates of xhat
        - **P_pre** -- a priori estimates of P
        - **P_post** -- a posteriori estimates of P
    """
    if B is None: B = np.zeros((A.shape[0], 1))
    if u is None: u = np.zeros(B.shape[1])
    xhat = xhat0
    P = P0

    # _pre variables are a priori predictions based on only past information
    # _post variables are a posteriori combinations of all information available at the current step
    xhat_pre = []; xhat_post = []; P_pre = []; P_post = []
    
    for n in range(y.shape[0]):
        xhat_ = A @ xhat + B @ u # ending underscores denote an a priori prediction
        P_ = A @ P @ A.T + Q
        xhat_pre.append(xhat_) # the [n]th index holds _{n|n-1} info
        P_pre.append(P_)

        xhat = xhat_ # handle missing data
        P = P_
        if not np.isnan(y[n]):
            K = P_ @ C.T @ np.linalg.inv(C @ P_ @ C.T + R)
            xhat += K @ (y[n] - C @ xhat_)
            P -= K @ C @ P_
        xhat_post.append(xhat) # the [n]th index holds _{n|n} info
        P_post.append(P)

    return np.stack(xhat_pre, axis=0), np.stack(xhat_post, axis=0), np.stack(P_pre, axis=0), np.stack(P_post, axis=0)


def _kalman_backward_smooth(A, xhat_pre, xhat_post, P_pre, P_post):
    """Do the additional Rauch-Tung-Striebel smoothing step
    :param A: discrete-time state transition matrix
    :param xhat_pre: a priori estimates of xhat
    :param xhat_post: a posteriori estimates of xhat
    :param P_pre: a priori estimates of P
    :param P_post: a posteriori estimates of P
    :return: tuple[np.array, np.array] of\n
        - **xhat_smooth** -- RTS smoothed xhat
        - **P_smooth** -- RTS smoothed P
    """
    xhat_smooth = [xhat_post[-1]]
    P_smooth = [P_post[-1]]
    for n in range(xhat_pre.shape[0]-2, -1, -1):
        C_RTS = P_post[n] @ A.T @ np.linalg.inv(P_pre[n+1]) # the [n+1]th index holds _{n+1|n} info 
        xhat_smooth.append(xhat_post[n] + C_RTS @ (xhat_smooth[-1] - xhat_pre[n+1])) # The original authors use C, not to
        P_smooth.append(P_post[n] + C_RTS @ (P_smooth[-1] - P_pre[n+1]) @ C_RTS.T) # be confused with the measurement matrix

    return np.stack(xhat_smooth[::-1], axis=0), np.stack(P_smooth[::-1], axis=0) # reverse lists


def _RTS_smooth(xhat0, P0, y, A, C, Q, R, u=None, B=None):
    """forward-backward Kalman/Rauch-Tung-Striebel smoother. For params see the helper functions.
    """
    xhat_pre, xhat_post, P_pre, P_post = _kalman_forward_filter(xhat0, P0, y, A, C, Q, R) # noisy x are the "y" in Kalman-land
    xhat_smooth, _ = _kalman_backward_smooth(A, xhat_pre, xhat_post, P_pre, P_post) # not doing anything with P_smooth
    return xhat_smooth


#########################################
# Constant 1st, 2nd, and 3rd derivative #
#########################################
def _constant_derivative(x, P0, A, C, R, Q, forwardbackward):
    """Helper for `constant_{velocity,acceleration,jerk}` functions, because there was a lot of
    repeated code.
    """
    xhat0 = np.zeros(A.shape[0]); xhat0[0] = x[0] # See #110 for why this choice of xhat0
    xhat_smooth = _RTS_smooth(xhat0, P0, x, A, C, Q, R) # noisy x are the "y" in Kalman-land  
    x_hat_forward = xhat_smooth[:, 0] # first dimension is time, so slice first element at all times
    dxdt_hat_forward = xhat_smooth[:, 1]

    if not forwardbackward: # bound out here if not doing the same in reverse and then combining
        return x_hat_forward, dxdt_hat_forward

    xhat0[0] = x[-1] # starting from the other end of the signal
    Ainv = np.linalg.inv(A) # dynamics are inverted
    xhat_smooth = _RTS_smooth(xhat0, P0, x[::-1], Ainv, C, Q, R) # noisy x are the "y" in Kalman-land    
    x_hat_backward = xhat_smooth[:, 0][::-1] # the result is backwards still, so reverse it
    dxdt_hat_backward = xhat_smooth[:, 1][::-1]

    w = np.linspace(0, 1, len(x))
    x_hat = x_hat_forward*w + x_hat_backward*(1-w)
    dxdt_hat = dxdt_hat_forward*w + dxdt_hat_backward*(1-w)

    return x_hat, dxdt_hat


def constant_velocity(x, dt, params=None, options=None, r=None, q=None, forwardbackward=True):
    """Run a forward-backward constant velocity RTS Kalman smoother to estimate the derivative.

    :param np.array[float] x: data series to differentiate
    :param float dt: step size
    :param list[float] params: (**deprecated**, prefer :code:`r` and :code:`q`)
    :param options: (**deprecated**, prefer :code:`forwardbackward`)
        a dictionary consisting of {'forwardbackward': (bool)}
    param float r: variance of the signal noise
    param float q: variance of the constant velocity model
    :param bool forwardbackward: indicates whether to run smoother forwards and backwards
        (usually achieves better estimate at end points)

    :return: tuple[np.array, np.array] of\n
        - **x_hat** -- estimated (smoothed) x
        - **dxdt_hat** -- estimated derivative of x
    """
    if params != None: # boilerplate backwards compatibility code
        warn("`params` and `options` parameters will be removed in a future version. Use `r`, " +
            "`q`, and `forwardbackward` instead.", DeprecationWarning)
        r, q = params
        if options != None:
            if 'forwardbackward' in options: forwardbackward = options['forwardbackward']
    elif r == None or q == None:
        raise ValueError("`q` and `r` must be given.")

    A = np.array([[1, dt], [0, 1]]) # states are x, x'. This basically does an Euler update.
    C = np.array([[1, 0]]) # we measure only y = noisy x
    R = np.array([[r]])
    Q = np.array([[1e-16, 0], [0, q]]) # uncertainty is around the velocity
    P0 = np.array(100*np.eye(2)) # See #110 for why this choice of P0

    return _constant_derivative(x, P0, A, C, R, Q, forwardbackward)


def constant_acceleration(x, dt, params=None, options=None, r=None, q=None, forwardbackward=True):
    """Run a forward-backward constant acceleration RTS Kalman smoother to estimate the derivative.

    :param np.array[float] x: data series to differentiate
    :param float dt: step size
    :param list[float] params: (**deprecated**, prefer :code:`r` and :code:`q`)
    :param options: (**deprecated**, prefer :code:`forwardbackward`)
        a dictionary consisting of {'forwardbackward': (bool)}
    param float r: variance of the signal noise
    param float q: variance of the constant acceleration model
    :param bool forwardbackward: indicates whether to run smoother forwards and backwards
        (usually achieves better estimate at end points)

    :return: tuple[np.array, np.array] of\n
        - **x_hat** -- estimated (smoothed) x
        - **dxdt_hat** -- estimated derivative of x
    """
    if params != None: # boilerplate backwards compatibility code
        warn("`params` and `options` parameters will be removed in a future version. Use `r`, " +
            "`q`, and `forwardbackward` instead.", DeprecationWarning)
        r, q = params
        if options != None:
            if 'forwardbackward' in options: forwardbackward = options['forwardbackward']
    elif r == None or q == None:
        raise ValueError("`q` and `r` must be given.")

    A = np.array([[1, dt, (dt**2)/2], # states are x, x', x"
                  [0, 1, dt],
                  [0, 0,  1]])
    C = np.array([[1, 0, 0]]) # we measure only y = noisy x
    R = np.array([[r]])
    Q = np.array([[1e-16, 0, 0],
                  [0, 1e-16, 0],
                  [0,     0, q]]) # uncertainty is around the acceleration
    P0 = np.array(100*np.eye(3)) # See #110 for why this choice of P0

    return _constant_derivative(x, P0, A, C, R, Q, forwardbackward)


def constant_jerk(x, dt, params=None, options=None, r=None, q=None, forwardbackward=True):
    """Run a forward-backward constant jerk RTS Kalman smoother to estimate the derivative.

    :param np.array[float] x: data series to differentiate
    :param float dt: step size
    :param list[float] params: (**deprecated**, prefer :code:`r` and :code:`q`)
    :param options: (**deprecated**, prefer :code:`forwardbackward`)
        a dictionary consisting of {'forwardbackward': (bool)}
    param float r: variance of the signal noise
    param float q: variance of the constant jerk model
    :param bool forwardbackward: indicates whether to run smoother forwards and backwards
        (usually achieves better estimate at end points)

    :return: tuple[np.array, np.array] of\n
        - **x_hat** -- estimated (smoothed) x
        - **dxdt_hat** -- estimated derivative of x
    """
    if params != None: # boilerplate backwards compatibility code
        warn("`params` and `options` parameters will be removed in a future version. Use `r`, " +
            "`q`, and `forwardbackward` instead.", DeprecationWarning)
        r, q = params
        if options != None:
            if 'forwardbackward' in options: forwardbackward = options['forwardbackward']
    elif r == None or q == None:
        raise ValueError("`q` and `r` must be given.")

    A = np.array([[1, dt, (dt**2)/2, (dt**3)/6], # states are x, x', x", x'"
                  [0, 1, dt, (dt**2)/2],
                  [0, 0,  1, dt],
                  [0, 0,  0, 1]])
    C = np.array([[1, 0, 0, 0]]) # we measure only y = noisy x
    R = np.array([[r]])
    Q = np.array([[1e-16,  0, 0,     0],
                   [0, 1e-16, 0,     0],
                   [0,     0, 1e-16, 0],
                   [0,     0, 0,     q]]) # uncertainty is around the jerk
    P0 = np.array(100*np.eye(4)) # See #110 for why this choice of P0

    return _constant_derivative(x, P0, A, C, R, Q, forwardbackward)


def known_dynamics(x, params, u=None, options=None, xhat0=None, P0=None, A=None,
    B=0, C=None, Q=None, R=None, smooth=True):
    """Run a forward RTS Kalman smoother given known dynamics.

    :param np.array[float] x: data series of noisy measurements
    :param list params: (**deprecated**, prefer :code:`xhat0`, :code:`P0`, :code:`A`,
        :code:`B`, :code:`C`, :code:`R`, and :code:`Q`), a list in the order here (note flip of Q and R)
    :param np.array[float] u: series of control inputs
    :param options: (**deprecated**, prefer :code:`smooth`)
        a dictionary consisting of {'smooth': (bool)}
    :param np.array xhat0: inital condition, length N = number of states
    :param np.array P0: initial covariance matrix of NxN
    :param np.array A: dynamics matrix, NxN
    :param np.array B: control input matrix, NxM, M = number of measurements
    :param np.array C: measurement dynamics, MxN
    :param np.array Q: covariance matrix for the model, NxN
    :param np.array R: covariance matrix for the measurements, MxM
    :parma bool smooth: whether to run the RTS smoother step

    :return: np.array **x_hat** -- estimated (smoothed) x
    """ # Why not also returning derivative here?
    if params != None:
        warn("`params` and `options` parameters will be removed in a future version. Use `xhat0`, " +
            "`P0`, `A`, `B`, `C`, `Q`, `R`, and `smooth` instead.", DeprecationWarning)
        xhat0, P0, A, B, C, R, Q = params
        if options != None:
            if 'smooth' in options: smooth = options['smooth']
    elif None in [xhat0, P0, A, C, R, Q]:
        raise ValueError("`xhat0`, `P0`, `A`, `C`, `Q`, and `R` must be given.")

    xhat_pre, xhat_post, P_pre, P_post = _kalman_forward_filter(xhat0, P0, x, A, C, Q, R, u, B) # noisy x are the "y" in Kalman-land
    if not smooth:
        return xhat_post

    xhat_smooth, _ = _kalman_backward_smooth(A, xhat_pre, xhat_post, P_pre, P_post)
    return xhat_smooth # We're not calculating a derivative here. Why not?
