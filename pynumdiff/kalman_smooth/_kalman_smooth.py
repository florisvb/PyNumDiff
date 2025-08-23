import numpy as np
from warnings import warn
from scipy.linalg import expm


def kalman_filter(y, _t, xhat0, P0, A, Q, C, R, B=None, u=None, save_P=True):
    """Run the forward pass of a Kalman filter, with regular or irregular step size.

    :param np.array y: series of measurements, stacked along axis 0.
    :param float or array[float] _t: This function supports variable step size. This parameter is either the constant
        dt if given as a single float, or data locations if given as an array of same length as :code:`x`.
    :param np.array xhat0: initial estimate of state (often 0) one step before the first measurement
    :param np.array P0: initial guess of state covariance (often identity matrix) before the first measurement
    :param np.array A: state transition matrix, in discrete time if constant dt, in continuous time if variable dt
    :param np.array Q: noise covariance matrix, in discrete time if constant dt, in continuous time if variable dt
    :param np.array C: measurement matrix
    :param np.array R: measurement noise covariance
    :param np.array B: optional control matrix, in discrete time if constant dt, in continuous time if variable dt
    :param np.array u: optional control input
    :param bool save_P: whether to save history of error covariance and a priori state estimates, used with rts
        smoothing but nonstandard to compute for ordinary filtering

    :return: tuple[np.array, np.array, np.array, np.array] of\n
        - **xhat_pre** -- a priori estimates of xhat, with axis=0 the batch dimension, so xhat[n] gets the nth step
        - **xhat_post** -- a posteriori estimates of xhat
        - **P_pre** -- a priori estimates of P
        - **P_post** -- a posteriori estimates of P
        if :code:`save_P` is :code:`True` else only **xhat_post** to save memory
    """
    # _pre variables are a priori predictions based on only past information
    # _post variables are a posteriori combinations of all information available at the current step
    xhat_post = []
    if save_P: xhat_pre = []; P_pre = []; P_post = []
    xhat = xhat0
    P = P0

    equispaced = isinstance(_t, (float, int)) # I'd rather not call isinstance in the loop
    control = isinstance(B, np.ndarray) and isinstance(B, np.ndarray) # whether there is a control input
    for n in range(y.shape[0]):
        if not equispaced:
            M = expm(np.block([[A, Q],[numpy.zeros(A.shape), -A.T]]) * (_t[n] - _t[n-1])) # form discrete-time matrices TODO doesn't work at n=0
            An = M[:order+1,:order+1] # upper left block
            Qn = M[:order+1,order+1:] @ An.T # upper right block
            if control:
                M = expm(np.block([[A, B], [np.zeros((A.shape[0], 2*A.shape[1]))]]))
                Bn = M[:order+1,order+1:] # upper right block 
        else: # matrices should already be discrete time
            An, Qn, Bn = A, Q, B

        xhat_ = An @ xhat + Bn @ u if control else An @ xhat # ending underscores denote an a priori prediction
        P_ = An @ P @ An.T + Qn # the dense matrix multiplies here are the most expensive step
        xhat = xhat_.copy() # copies, lest modifications to these variables change the a priori estimates. See #122
        P = P_.copy()
        if not np.isnan(y[n]): # handle missing data
            K = P_ @ C.T @ np.linalg.inv(C @ P_ @ C.T + R)
            xhat += K @ (y[n] - C @ xhat_)
            P -= K @ C @ P_
        # the [n]th index of pre variables holds _{n|n-1} info; the [n]th index of post variables holds _{n|n} info
        xhat_post.append(xhat)
        if save_P: xhat_pre.append(xhat_); P_pre.append(P_); P_post.append(P)

    xhat_post = np.stack(xhat_post, axis=0)
    return xhat_post if not save_P else (np.stack(xhat_pre, axis=0), xhat_post,
                                         np.stack(P_pre, axis=0), np.stack(P_post, axis=0))


def rts_smooth(_t, A, xhat_pre, xhat_post, P_pre, P_post, compute_P_smooth=False):
    """Perform Rauch-Tung-Striebel smoothing, using information from forward Kalman filtering.

    :param float or array[float] _t: This function supports variable step size. This parameter is either the constant
        dt if given as a single float, or data locations if given as an array of same length as :code:`x`.
    :param np.array A: state transition matrix, in discrete time if constant dt, in continuous time if variable dt
    :param list[np.array] xhat_pre: a priori estimates of xhat from a kalman_filter forward pass
    :param list[np.array] xhat_post: a posteriori estimates of xhat from a kalman_filter forward pass
    :param list[np.array] P_pre: a priori estimates of P from a kalman_filter forward pass
    :param list[np.array] P_post: a posteriori estimates of P from a kalman_filter forward pass
    :param bool compute_P_smooth: it's extra work if you don't need it
    
    :return: tuple[np.array, np.array] of\n
        - **xhat_smooth** -- RTS smoothed xhat
        - **P_smooth** -- RTS smoothed P
        if :code:`compute_P_smooth` is :code:`True` else only **xhat_smooth**
    """
    xhat_smooth = np.empty(xhat_post.shape); xhat_smooth[-1] = xhat_post[-1] # preallocate arrays
    if compute_P_smooth: P_smooth = np.empty(P_post.shape); P_smooth[-1] = P_post[-1]
    
    equispaced = isinstance(_t, (int, float)) # I'd rather not call isinstance in a loop when it's avoidable
    for n in range(xhat_pre.shape[0]-2, -1, -1):
        An = A if equispaced else expm(A * (_t[n+1] - _t[n])) # state transition from n to n+1, per the paper
        C_RTS = P_post[n] @ An.T @ np.linalg.inv(P_pre[n+1]) # the [n+1]th index holds _{n+1|n} info 
        xhat_smooth[n] = xhat_post[n] + C_RTS @ (xhat_smooth[n+1] - xhat_pre[n+1]) # The original authors use C, not to be confused
        if compute_P_smooth: P_smooth[n] = P_post[n] + C_RTS @ (P_smooth[n+1] - P_pre[n+1]) @ C_RTS.T # with the measurement matrix

    return xhat_smooth if not compute_P_smooth else (xhat_smooth, P_smooth)


def rtsdiff(x, _t, order, qr_ratio, forwardbackward):
    """Perform Rauch-Tung-Striebel smoothing with a naive constant derivative model. Other constant derivative
    methods in this module call this function.

    :param np.array[float] x: data series to differentiate
    :param float or array[float] _t: This function supports variable step size. This parameter is either the constant
        dt if given as a single float, or data locations if given as an array of same length as :code:`x`.
    :param int order: which derivative to stabilize in the constant-derivative model
    :param qr_ratio: the process noise level of the divided by the measurement noise level, because,
        per `our analysis <https://github.com/florisvb/PyNumDiff/issues/139>`_, the mean result is
        dependent on the relative rather than absolute size of :math:`q` and :math:`r`.
    :param bool forwardbackward: indicates whether to run smoother forwards and backwards
        (usually achieves better estimate at end points)

    :return: tuple[np.array, np.array] of\n
        - **x_hat** -- estimated (smoothed) x
        - **dxdt_hat** -- estimated derivative of x
    """
    if isinstance(_t, (np.ndarray, list)) and len(x) != len(_t):
        raise ValueError("If `_t` is given as array-like, must have same length as `x`.")

    q = 10**int(np.log10(qr_ratio)/2) # even-ish split of the powers across 0
    r = q/qr_ratio

    A = np.diag(np.ones(order), 1) # continuous-time A just has 1s on the first diagonal (where 0th is main diagonal)
    Q = np.zeros(A.shape); Q[-1,-1] = q # continuous-time uncertainty around the last derivative
    C = np.zeros((1, order+1)); C[0,0] = 1 # we measure only y = noisy x
    R = np.array([[r]]) # 1 observed state, so this is 1x1
    P0 = 100*np.eye(order+1) # See #110 for why this choice of P0
    xhat0 = np.zeros(A.shape[0]); xhat0[0] = x[0] # The first estimate is the first seen state. See #110

    if isinstance(_t, (float, int)):
        M = expm(np.block([[A, Q],[numpy.zeros(A.shape), -A.T]]) * _t) # form discrete-time versions
        A = M[:order+1,:order+1]
        Q = M[:order+1,order+1:] @ A.T

    xhat_pre, xhat_post, P_pre, P_post = kalman_filter(x, _t, xhat0, P0, A, Q, C, R) # noisy x are the "y" in Kalman-land
    xhat_smooth = rts_smooth(_t, A, xhat_pre, xhat_post, P_pre, P_post) # not doing anything with P_smooth  
    x_hat_forward = xhat_smooth[:, 0] # first dimension is time, so slice first element at all times
    dxdt_hat_forward = xhat_smooth[:, 1]

    if not forwardbackward: # bounce out here if not doing the same in reverse and then combining
        return x_hat_forward, dxdt_hat_forward

    # for backward case, if discrete time invert the matrix, and for continuous time reverse _t, because then dts will go negative
    xhat0[0] = x[-1] # starting from the other end of the signal

    ### TODO fix between here and next hashes
    if isinstance(_t, (float, int)):
        M = expm(np.block([[A, Q],[numpy.zeros(A.shape), -A.T]]) * -_t) # form discrete-time versions
        A = M[:order+1,:order+1] # inverse of A
        Q = M[:order+1,order+1:] @ A.T # 

    Ainv = np.linalg.inv(A_c) # dynamics are inverted, same as expm() with negative dt
    _t_rev = _t[::-1]  _t
    
    xhat_pre, xhat_post, P_pre, P_post = _kalman_forward_filter(x[::-1], _t[::-1], xhat0, P0, A, Q, C, R) if \
        isinstance(_t, (np.ndarray, list)) else _kalman_forward_filter(x[::-1], _t, xhat0, P0, np.linalg.inv(A), Q, C, R)
    xhat_smooth = _kalman_backward_smooth(_t_rev, A_c, xhat_pre, xhat_post, P_pre, P_post)
    ###

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
    :param float r: variance of the signal noise
    :param float q: variance of the constant velocity model
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

    return rtsdiff(x, dt, 1, q/r, forwardbackward)


def constant_acceleration(x, dt, params=None, options=None, r=None, q=None, forwardbackward=True):
    """Run a forward-backward constant acceleration RTS Kalman smoother to estimate the derivative.

    :param np.array[float] x: data series to differentiate
    :param float dt: step size
    :param list[float] params: (**deprecated**, prefer :code:`r` and :code:`q`)
    :param options: (**deprecated**, prefer :code:`forwardbackward`)
        a dictionary consisting of {'forwardbackward': (bool)}
    :param float r: variance of the signal noise
    :param float q: variance of the constant acceleration model
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

    return rtsdiff(x, dt, 2, q/r, forwardbackward)


def constant_jerk(x, dt, params=None, options=None, r=None, q=None, forwardbackward=True):
    """Run a forward-backward constant jerk RTS Kalman smoother to estimate the derivative.

    :param np.array[float] x: data series to differentiate
    :param float dt: step size
    :param list[float] params: (**deprecated**, prefer :code:`r` and :code:`q`)
    :param options: (**deprecated**, prefer :code:`forwardbackward`)
        a dictionary consisting of {'forwardbackward': (bool)}
    :param float r: variance of the signal noise
    :param float q: variance of the constant jerk model
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

    return rtsdiff(x, dt, 3, q/r, forwardbackward)
