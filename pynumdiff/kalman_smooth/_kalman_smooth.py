import numpy as np
from warnings import warn
from scipy.linalg import expm, sqrtm

try: import cvxpy
except ImportError: pass


def kalman_filter(y, _t, xhat0, P0, A, Q, C, R, B=None, u=None, save_P=True):
    """Run the forward pass of a Kalman filter, with regular or irregular step size.

    :param np.array y: series of measurements, stacked along axis 0.
    :param float or array[float] _t: This function supports variable step size. This parameter is either the constant
        step size if given as a single float, or sample locations if given as an array of same length as :code:`x`.
    :param np.array xhat0: a priori guess of initial state at the time of the first measurement
    :param np.array P0: a priori guess of state covariance at the time of first measurement (often identity matrix)
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
    N = y.shape[0]
    m = xhat0.shape[0] # dimension of the state
    xhat_post = np.empty((N,m))
    if save_P:
        xhat_pre = np.empty((N,m)) # _pre = a priori predictions based on only past information
        P_pre = np.empty((N,m,m)) # _post = a posteriori combinations of all information available at a step
        P_post = np.empty((N,m,m))
    # determine some things ahead of the loop
    equispaced = np.isscalar(_t)
    control = isinstance(B, np.ndarray) and isinstance(B, np.ndarray) # whether there is a control input
    if equispaced:
        An, Qn, Bn = A, Q, B # in this case only need to assign once
    else:
        M = np.block([[A, Q],[np.zeros(A.shape), -A.T]]) # If variable dt, we'll exponentiate this a bunch
        if control: Mc = np.block([[A, B],[np.zeros((A.shape[0], 2*A.shape[1]))]])
    
    for n in range(N):
        if n == 0: # first iteration is a special case, involving less work
            xhat_ = xhat0
            P_ = P0
        else:
            if not equispaced:
                dt = _t[n] - _t[n-1]
                eM = expm(M * dt) # form discrete-time matrices TODO doesn't work at n=0
                An = eM[:m,:m] # upper left block
                Qn = eM[:m,m:] @ An.T # upper right block
                if dt < 0: Qn = np.abs(Qn) # eigenvalues go negative if reverse time, but noise shouldn't shrink
                if control:
                    eM = expm(Mc * dt)
                    Bn = eM[:m,m:] # upper right block 
            xhat_ = An @ xhat + Bn @ u if control else An @ xhat # ending underscores denote an a priori prediction
            P_ = An @ P @ An.T + Qn # the dense matrix multiplies here are the most expensive step
        
        xhat = xhat_.copy() # copies, lest modifications to these variables change the a priori estimates. See #122
        P = P_.copy()
        if not np.isnan(y[n]): # handle missing data
            K = P_ @ C.T @ np.linalg.inv(C @ P_ @ C.T + R)
            xhat += K @ (y[n] - C @ xhat_)
            P -= K @ C @ P_
        # the [n]th index of pre variables holds _{n|n-1} info; the [n]th index of post variables holds _{n|n} info
        xhat_post[n] = xhat
        if save_P: xhat_pre[n] = xhat_; P_pre[n] = P_; P_post[n] = P

    return xhat_post if not save_P else (xhat_pre, xhat_post, P_pre, P_post)


def rts_smooth(_t, A, xhat_pre, xhat_post, P_pre, P_post, compute_P_smooth=True):
    """Perform Rauch-Tung-Striebel smoothing, using information from forward Kalman filtering.

    :param float or array[float] _t: This function supports variable step size. This parameter is either the constant
        step size if given as a single float, or sample locations if given as an array of same length as the state histories.
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
    
    equispaced = np.isscalar(_t) # I'd rather not call isinstance in a loop when it's avoidable
    if equispaced: An = A # in this case only assign once, outside the loop
    for n in range(xhat_pre.shape[0]-2, -1, -1):
        if not equispaced: An = expm(A * (_t[n+1] - _t[n])) # state transition from n to n+1, per the paper
        C_RTS = P_post[n] @ An.T @ np.linalg.inv(P_pre[n+1]) # the [n+1]th index holds _{n+1|n} info 
        xhat_smooth[n] = xhat_post[n] + C_RTS @ (xhat_smooth[n+1] - xhat_pre[n+1]) # The original authors use C, not to be confused
        if compute_P_smooth: P_smooth[n] = P_post[n] + C_RTS @ (P_smooth[n+1] - P_pre[n+1]) @ C_RTS.T # with the measurement matrix

    return xhat_smooth if not compute_P_smooth else (xhat_smooth, P_smooth)


def rtsdiff(x, _t, order, qr_ratio, forwardbackward):
    """Perform Rauch-Tung-Striebel smoothing with a naive constant derivative model. Makes use of :code:`kalman_filter`
    and :code:`rts_smooth`, which are made public. Other constant derivative methods in this module call this function.

    :param np.array[float] x: data series to differentiate
    :param float or array[float] _t: This function supports variable step size. This parameter is either the constant
        step size if given as a single float, or data locations if given as an array of same length as :code:`x`.
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
    if not np.isscalar(_t) and len(x) != len(_t):
        raise ValueError("If `_t` is given as array-like, must have same length as `x`.")
    x = np.asarray(x) # to flexibly allow array-like inputs

    q = 10**int(np.log10(qr_ratio)/2) # even-ish split of the powers across 0
    r = q/qr_ratio

    A = np.diag(np.ones(order), 1) # continuous-time A just has 1s on the first diagonal (where 0th is main diagonal)
    Q = np.zeros(A.shape); Q[-1,-1] = q # continuous-time uncertainty around the last derivative
    C = np.zeros((1, order+1)); C[0,0] = 1 # we measure only y = noisy x
    R = np.array([[r]]) # 1 observed state, so this is 1x1
    P0 = 100*np.eye(order+1) # See #110 for why this choice of P0
    xhat0 = np.zeros(A.shape[0]); xhat0[0] = x[0] # The first estimate is the first seen state. See #110

    if np.isscalar(_t):
        eM = expm(np.block([[A, Q],[np.zeros(A.shape), -A.T]]) * _t) # form discrete-time versions
        A = eM[:order+1,:order+1]
        Q = eM[:order+1,order+1:] @ A.T

    xhat_pre, xhat_post, P_pre, P_post = kalman_filter(x, _t, xhat0, P0, A, Q, C, R) # noisy x are the "y" in Kalman-land
    xhat_smooth = rts_smooth(_t, A, xhat_pre, xhat_post, P_pre, P_post, compute_P_smooth=False)  
    x_hat_forward = xhat_smooth[:, 0] # first dimension is time, so slice first element at all times
    dxdt_hat_forward = xhat_smooth[:, 1]

    if not forwardbackward: # bounce out here if not doing the same in reverse and then combining
        return x_hat_forward, dxdt_hat_forward

    xhat0[0] = x[-1] # starting from the other end of the signal

    if np.isscalar(_t): A = np.linalg.inv(A) # discrete time dynamics are just the inverse
    else: _t = _t[::-1] # in continuous time, reverse the time vector so dts go negative
    
    xhat_pre, xhat_post, P_pre, P_post = kalman_filter(x[::-1], _t, xhat0, P0, A, Q, C, R)
    xhat_smooth = rts_smooth(_t, A, xhat_pre, xhat_post, P_pre, P_post, compute_P_smooth=False)
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


def robustdiff(x, dt, order, qr_ratio, huberM):
    """Perform robust differentiation using L1-norm optimization instead of L2-norm RTS smoothing.
    
    This function solves the L1-normed MAP optimization problem for outlier-resistant differentiation:
    :math:`\\min_{\\{x_n\\}} \\sum_{n=0}^{N-1} V(R^(-1/2)(y_n - C x_n)) + sum_{n=1}^{N-1} J(Q^(-1/2)(x_n - A x_{n-1}))`
    
    The L1 norm provides better robustness to outliers compared to the L2 norm used in standard
    Kalman smoothing. Uses CVXPY for convex optimization.

    :param np.array[float] x: data series to differentiate
    :param float dt: step size
    :param int order: which derivative to stabilize in the constant-derivative model (1=velocity, 2=acceleration, 3=jerk)
    :param float qr_ratio: the process noise level of the divided by the measurement noise level, because the result is
        dependent on the relative rather than absolute size of :math:`q` and :math:`r`.
    :param float huberM: radius where quadratic of Huber loss function turns linear. M=0 reduces to the :math:`\\ell_1` norm.

    :return: tuple[np.array, np.array] of\n
        - **x_hat** -- estimated (smoothed) x
        - **dxdt_hat** -- estimated derivative of x
    """
    #q = 10**int(np.log10(qr_ratio)) # even-ish split of the powers across 0
    #r = q/qr_ratio
    q = 1e4
    r = q/qr_ratio

    A_c = np.diag(np.ones(order), 1) # continuous-time A just has 1s on the first diagonal (where 0th is main diagonal)
    Q_c = np.zeros(A_c.shape); Q_c[-1,-1] = q # continuous-time uncertainty around the last derivative
    C = np.zeros((1, order+1)); C[0,0] = 1 # we measure only y = noisy x
    R = np.array([[r]]) # 1 observed state, so this is 1x1
    
    # convert to discrete time using matrix exponential
    eM = expm(np.block([[A_c, Q_c], [np.zeros(A_c.shape), -A_c.T]]) * dt) # Note this could handle variable dt, similar to rtsdiff
    A_d = eM[:order+1, :order+1]
    Q_d = eM[:order+1, order+1:] @ A_d.T
    if np.linalg.cond(Q_d) > 1e12: Q_d += np.eye(order + 1)*1e-12 # for numerical stability with convex solver. Doesn't change answers appreciably (or at all).

    print(f"order={order}, qr_ratio={qr_ratio:.3e}, huberM={huberM}, cond(Q)={np.linalg.cond(Q_d):.3e}")
    x_states = convex_smooth(x, A_d, Q_d, R, C, huberM=huberM) # outsource solution of the convex optimization problem
    return x_states[:, 0], x_states[:, 1]


def convex_smooth(y, A, Q, R, C, huberM=0):
    """Solve the optimization problem for robust smoothing using CVXPY. Note this currently assumes constant dt
    but could be extended to handle variable step sizes by finding discrete-time A and Q for requisite gaps.

    :param np.array[float] y: measurements
    :param np.array A: discrete-time state transition matrix
    :param np.array Q: discrete-time process noise covariance matrix
    :param np.array R: measurement noise covariance matrix
    :param np.array C: measurement matrix
    :param float huberM: radius where quadratic of Huber loss function turns linear. M=0 reduces to the :math:`\\ell_1` norm.
    
    :return: np.array -- state estimates (N x state_dim)
    """
    N = len(y)
    x_states = cvxpy.Variable((N, A.shape[0])) # each row is [position, velocity, acceleration, ...] at step n

    R_sqrt_inv = np.linalg.inv(sqrtm(R))
    Q_sqrt_inv = np.linalg.inv(sqrtm(Q))
    objective = cvxpy.sum([cvxpy.norm(R_sqrt_inv @ (y[n] - C @ x_states[n]), 1) if huberM < 1e-3
                     else cvxpy.sum(cvxpy.huber(R_sqrt_inv @ (y[n] - C @ x_states[n]), huberM)) for n in range(N)]) # Measurement terms: sum of |R^(-1/2)(y_n - C x_n)|_1
    objective += cvxpy.sum([cvxpy.norm(Q_sqrt_inv @ (x_states[n] - A @ x_states[n-1]), 1) if huberM < 1e-3
                      else cvxpy.sum(cvxpy.huber(Q_sqrt_inv @ (x_states[n] - A @ x_states[n-1]), huberM)) for n in range(1, N)]) # Process terms: sum of |Q^(-1/2)(x_n - A x_{n-1})|_1
    
    problem = cvxpy.Problem(cvxpy.Minimize(objective))
    try:
        problem.solve(solver=cvxpy.CLARABEL)
    except cvxpy.error.SolverError:
        print(f"CLARABEL failed. Retrying with SCS.")
        problem.solve(solver=cvxpy.SCS) # SCS is a lot slower but pretty bulletproof even with big condition numbers

    if x_states.value is None: # There is occasional solver failure with huber as opposed to 1-norm
        warn("Convex solvers failed with status {problem.status}. Returning NaNs.")
        return np.full((N, A.shape[0]), np.nan)
    
    return x_states.value
