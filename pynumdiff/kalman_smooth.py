"""This module implements constant-derivative model-based smoothers based on Kalman filtering and its generalization."""
from warnings import warn
import numpy as np
from scipy.linalg import expm, sqrtm
try: import cvxpy
except ImportError: pass

from pynumdiff.utils.utility import huber_const, wrap_angle, ensure_iterable


def kalman_filter(y, xhat0, P0, A, Q, C, R, B=None, u=None, save_P=True, 
                  circular_vars=None, circular_units='rad'):
    """Run the forward pass of a Kalman filter. Expects discrete-time matrices; use :func:`scipy.linalg.expm`
    in the caller to convert from continuous time if needed.

    :param np.array y: series of measurements, stacked in the direction of axis 0.
    :param np.array xhat0: a priori guess of initial state at the time of the first measurement
    :param np.array P0: a priori guess of state covariance at the time of first measurement (often identity matrix)
    :param np.array A: discrete-time state transition matrix. If 2D (:math:`m \\times m`), the same matrix is used for all steps;
        if 3D (:math:`N-1 \\times m \\times m`), :code:`A[n-1]` is used for the transition from step :math:`n-1` to :math:`n`.
    :param np.array Q: discrete-time process noise covariance. Same 2D or 3D shape convention as :code:`A`.
    :param np.array C: measurement matrix
    :param np.array R: measurement noise covariance
    :param np.array B: optional discrete-time control matrix, optionally stacked like :code:`A`.
    :param np.array u: optional control inputs, stacked in the direction of axis 0
    :param bool save_P: whether to save history of error covariance and a priori state estimates, used with rts
        smoothing but nonstandard to compute for ordinary filtering
    :param bool or None circular_vars: bool indicating whether the measurement y is a circular (angular) variable 
        that is wrapped. This will use a circular innovation calculation for the Kalman filter. The smoothed result
        will be returned in an unwrapped form. 
    :param string circular_units: 'rad' or 'deg' to specify whether wrapping is in degrees or radians. 

    :return: - **xhat_pre** (np.array) -- a priori estimates of xhat, with axis=0 the batch dimension, so xhat[n] gets the nth step
             - **xhat_post** (np.array) -- a posteriori estimates of xhat
             - **P_pre** (np.array) -- a priori estimates of P
             - **P_post** (np.array) -- a posteriori estimates of P
             if :code:`save_P` is :code:`True` else only **xhat_post** to save memory
    """
    if A.ndim != Q.ndim: raise ValueError("number of dimensions of A and Q must agree, either single matrix or stacked")

    N = y.shape[0]
    m = xhat0.shape[0] # dimension of the state
    xhat_post = np.empty((N,m))
    if save_P:
        xhat_pre = np.empty_like(xhat_post) # _pre = a priori predictions based on only past information
        P_pre = np.empty((N,m,m)) # _post = a posteriori combinations of all information available at a step
        P_post = np.empty_like(P_pre)
    
    control = isinstance(B, np.ndarray) and isinstance(u, np.ndarray) # whether there is a control input
    if A.ndim == 2: An, Qn, Bn = A, Q, B # single matrices, assign once outside the loop

    for n in range(N):
        if n == 0: # first iteration is a special case, involving less work
            xhat_ = xhat0
            P_ = P0
        else:
            if A.ndim == 3: An, Qn = A[n-1], Q[n-1]; Bn = B[n-1] if control else B # use the matrices corresponding to this step
            xhat_ = An @ xhat + Bn @ u[n] if control else An @ xhat # ending underscores denote a priori predictions
            P_ = An @ P @ An.T + Qn # the dense matrix multiplies here are the most expensive step

        xhat = xhat_.copy() # copies, lest modifications to these variables change the a priori estimates. See #122
        P = P_.copy()
        if not np.isnan(y[n]): # handle missing data
            K = P_ @ C.T @ np.linalg.inv(C @ P_ @ C.T + R)
            innovation = y[n] - C @ xhat_
            if circular_vars is not None and circular_vars is not False:
                innovation[0] = wrap_angle(innovation[0], circular_units)
            xhat += K @ innovation
            P -= K @ C @ P_
        # the [n]th index of pre variables holds _{n|n-1} info; the [n]th index of post variables holds _{n|n} info
        xhat_post[n] = xhat
        if save_P: xhat_pre[n] = xhat_; P_pre[n] = P_; P_post[n] = P

    return xhat_post if not save_P else (xhat_pre, xhat_post, P_pre, P_post)


def rts_smooth(A, xhat_pre, xhat_post, P_pre, P_post, compute_P_smooth=True):
    """Perform Rauch-Tung-Striebel smoothing, using information from forward Kalman filtering.

    :param np.array A: discrete-time state transition matrix. If 2D (:math:`m \\times m`), the same matrix is used for all steps;
        if 3D (:math:`N-1 \\times m \\times m`), :code:`A[n-1]` is used for the transition from step :math:`n-1` to :math:`n`.
    :param np.array xhat_pre: a priori estimates of xhat from a kalman_filter forward pass
    :param np.array xhat_post: a posteriori estimates of xhat from a kalman_filter forward pass
    :param np.array P_pre: a priori estimates of P from a kalman_filter forward pass
    :param np.array P_post: a posteriori estimates of P from a kalman_filter forward pass
    :param bool compute_P_smooth: it's extra work if you don't need it
    
    :return: - **xhat_smooth** (np.array) -- RTS smoothed xhat
             - **P_smooth** (np.array) -- RTS smoothed P estimates
             if :code:`compute_P_smooth` is :code:`True` else only **xhat_smooth**
    """
    xhat_smooth = np.empty(xhat_post.shape); xhat_smooth[-1] = xhat_post[-1] # preallocate arrays
    if compute_P_smooth: P_smooth = np.empty(P_post.shape); P_smooth[-1] = P_post[-1]

    if A.ndim == 2: An = A # single matrix, assign once outside the loop
    for n in range(xhat_pre.shape[0]-2, -1, -1):
        if A.ndim == 3: An = A[n] # state transition matrix from index n to n+1
        C_RTS = P_post[n] @ An.T @ np.linalg.inv(P_pre[n+1]) # the [n+1]th index holds _{n+1|n} info
        xhat_smooth[n] = xhat_post[n] + C_RTS @ (xhat_smooth[n+1] - xhat_pre[n+1]) # The original authors use C, not to be confused
        if compute_P_smooth: P_smooth[n] = P_post[n] + C_RTS @ (P_smooth[n+1] - P_pre[n+1]) @ C_RTS.T # with the measurement matrix

    return xhat_smooth if not compute_P_smooth else (xhat_smooth, P_smooth)


def rtsdiff(x, dt_or_t, order, log_qr_ratio, forwardbackward, axis=0, 
            circular_vars=None, circular_units='rad'):
    """Perform Rauch-Tung-Striebel smoothing with a naive constant derivative model. Makes use of :code:`kalman_filter`
    and :code:`rts_smooth`, which are made public. :code:`constant_X` methods in this module call this function.

    :param np.array[float] x: data series to differentiate. May contain NaN values (missing data); NaNs are excluded
        from fitting and imputed by dynamical model evolution. May be multidimensional; see :code:`axis`.
    :param float or array[float] dt_or_t: This function supports variable step size. This parameter is either the constant
        :math:`\\Delta t` if given as a single float, or data locations if given as an array of same length as :code:`x`.
    :param int order: which derivative to stabilize in the constant-derivative model
    :param log_qr_ratio: the log of the process noise level divided by the measurement noise level, because,
        per `our analysis <https://github.com/florisvb/PyNumDiff/issues/139>`_, the mean result is
        dependent on the relative rather than absolute size of :math:`q` and :math:`r`.
    :param bool forwardbackward: indicates whether to run smoother forwards and backwards
        (usually achieves better estimate at end points)
    :param int axis: data dimension along which differentiation is performed
    :param list[bool] circular_vars: set list element to bool for any axes of x that are a circular (angular) variable 
        that is wrapped. This will use a circular innovation calculation for the Kalman filter. The smoothed result
        will be returned in an unwrapped form. 
    :param string circular_units: 'rad' or 'deg' to specify whether wrapping is in degrees or radians. 

    :return: - **x_hat** (np.array) -- estimated (smoothed) x, same shape as input :code:`x`
             - **dxdt_hat** (np.array) -- estimated derivative of x, same shape as input :code:`x`
    """
    N = x.shape[axis]
    if not np.isscalar(dt_or_t) and N != len(dt_or_t):
        raise ValueError("If `dt_or_t` is given as array-like, must have same length as x along `axis`.")
    
    # turn circular_vars into something with the same shape as the number of differentiated axes in x
    if len(x.shape) > 1:
        n = int(np.prod(x.shape[:axis] + x.shape[axis+1:]))
        circular_vars = ensure_iterable(circular_vars, n)
    else:
        circular_vars = ensure_iterable(circular_vars, 1)

    q = 10**int(log_qr_ratio/2) # even-ish split of the powers across 0
    r = q/(10**log_qr_ratio)
    A_c = np.diag(np.ones(order), 1) # continuous-time A just has 1s on the first diagonal (where 0th is main diagonal)
    Q_c = np.zeros(A_c.shape); Q_c[-1,-1] = q # continuous-time uncertainty around the last derivative
    C = np.zeros((1, order+1)); C[0,0] = 1 # we measure only y = noisy x
    R = np.array([[r]]); P0 = 100*np.eye(order+1) # See #110 for why this choice of P0
    M = np.block([[A_c, Q_c],[np.zeros(A_c.shape), -A_c.T]])

    # Pre-compute discrete-time transition matrices once (shared across all dimensions)
    if np.isscalar(dt_or_t):
        eM = expm(M * dt_or_t)
        A_d = eM[:order+1, :order+1]
        Q_d = eM[:order+1, order+1:] @ A_d.T
        if forwardbackward: A_d_bwd = np.linalg.inv(A_d)
    else:
        A_d = np.empty((N-1, order+1, order+1))
        Q_d = np.empty_like(A_d)
        for n,dt in enumerate(np.diff(dt_or_t)):
            eM = expm(M * dt)
            A_d[n] = eM[:order+1, :order+1]
            Q_d[n] = eM[:order+1, order+1:] @ A_d[n].T
        if forwardbackward: A_d_bwd = np.linalg.inv(A_d[::-1]) # properly broadcasts, taking inv of each stacked 2D array

    x_hat = np.empty_like(x); dxdt_hat = np.empty_like(x)
    if forwardbackward: w = np.linspace(0, 1, N) # weights used to combine forward and backward results

    for i, vec_idx in enumerate(np.ndindex(x.shape[:axis] + x.shape[axis+1:])): # works properly for 1D case too
        s = vec_idx[:axis] + (slice(None),) + vec_idx[axis:] # for indexing the vector we wish to differentiate
        xhat0 = np.zeros(order+1); xhat0[0] = x[s][0] if not np.isnan(x[s][0]) else 0 # The first estimate is the first seen state. See #110

        xhat_pre, xhat_post, P_pre, P_post = kalman_filter(x[s], xhat0, P0, A_d, Q_d, C, R, 
                                                           circular_vars=circular_vars[i], circular_units=circular_units)
        xhat_smooth = rts_smooth(A_d, xhat_pre, xhat_post, P_pre, P_post, compute_P_smooth=False)
        x_hat[s] = xhat_smooth[:,0] # first dimension is time, so slice first and second states at all times
        dxdt_hat[s] = xhat_smooth[:,1]

        if forwardbackward:
            xhat0[0] = x[s][-1] if not np.isnan(x[s][-1]) else 0
            xhat_pre, xhat_post, P_pre, P_post = kalman_filter(x[s][::-1], xhat0, P0, A_d_bwd,
                Q_d if Q_d.ndim == 2 else Q_d[::-1], C, R, 
                circular_vars=circular_vars[i], circular_units=circular_units) # Use same Q matrices as before, because noise should still grow in reverse time
            xhat_smooth = rts_smooth(A_d_bwd, xhat_pre, xhat_post, P_pre, P_post, compute_P_smooth=False)

            x_hat[s] = x_hat[s] * w + xhat_smooth[:, 0][::-1] * (1-w)
            dxdt_hat[s] = dxdt_hat[s] * w + xhat_smooth[:, 1][::-1] * (1-w)

    return x_hat, dxdt_hat


def constant_velocity(x, dt, params=None, options=None, r=None, q=None, forwardbackward=True):
    """Run a forward-backward constant velocity RTS Kalman smoother to estimate the derivative.\n
    **Deprecated**, prefer :code:`rtsdiff` with order 1 instead.

    :param np.array[float] x: data series to differentiate
    :param float dt: step size
    :param list[float] params: (**deprecated**, prefer :code:`r` and :code:`q`)
    :param options: (**deprecated**, prefer :code:`forwardbackward`)
        a dictionary consisting of {'forwardbackward': (bool)}
    :param float r: variance of the signal noise
    :param float q: variance of the constant velocity model
    :param bool forwardbackward: indicates whether to run smoother forwards and backwards
        (usually achieves better estimate at end points)

    :return: - **x_hat** (np.array) -- estimated (smoothed) x
             - **dxdt_hat** (np.array) -- estimated derivative of x
    """
    if params is not None: # boilerplate backwards compatibility code
        warn("`params` and `options` parameters will be removed in a future version. Use `r`, " +
            "`q`, and `forwardbackward` instead.", DeprecationWarning)
        r, q = params
        if options is not None:
            if 'forwardbackward' in options: forwardbackward = options['forwardbackward']
    elif r is None or q is None:
        raise ValueError("`q` and `r` must be given.")

    warn("`constant_velocity` is deprecated. Call `rtsdiff` with order 1 instead.", DeprecationWarning)
    return rtsdiff(x, dt, 1, np.log10(q/r), forwardbackward)


def constant_acceleration(x, dt, params=None, options=None, r=None, q=None, forwardbackward=True):
    """Run a forward-backward constant acceleration RTS Kalman smoother to estimate the derivative.\n
    **Deprecated**, prefer :code:`rtsdiff` with order 2 instead.

    :param np.array[float] x: data series to differentiate
    :param float dt: step size
    :param list[float] params: (**deprecated**, prefer :code:`r` and :code:`q`)
    :param options: (**deprecated**, prefer :code:`forwardbackward`)
        a dictionary consisting of {'forwardbackward': (bool)}
    :param float r: variance of the signal noise
    :param float q: variance of the constant acceleration model
    :param bool forwardbackward: indicates whether to run smoother forwards and backwards
        (usually achieves better estimate at end points)

    :return: - **x_hat** (np.array) -- estimated (smoothed) x
             - **dxdt_hat** (np.array) -- estimated derivative of x
    """
    if params is not None: # boilerplate backwards compatibility code
        warn("`params` and `options` parameters will be removed in a future version. Use `r`, " +
            "`q`, and `forwardbackward` instead.", DeprecationWarning)
        r, q = params
        if options is not None:
            if 'forwardbackward' in options: forwardbackward = options['forwardbackward']
    elif r is None or q is None:
        raise ValueError("`q` and `r` must be given.")

    warn("`constant_acceleration` is deprecated. Call `rtsdiff` with order 2 instead.", DeprecationWarning)
    return rtsdiff(x, dt, 2, np.log10(q/r), forwardbackward)


def constant_jerk(x, dt, params=None, options=None, r=None, q=None, forwardbackward=True):
    """Run a forward-backward constant jerk RTS Kalman smoother to estimate the derivative.\n
    **Deprecated**, prefer :code:`rtsdiff` with order 3 instead.

    :param np.array[float] x: data series to differentiate
    :param float dt: step size
    :param list[float] params: (**deprecated**, prefer :code:`r` and :code:`q`)
    :param options: (**deprecated**, prefer :code:`forwardbackward`)
        a dictionary consisting of {'forwardbackward': (bool)}
    :param float r: variance of the signal noise
    :param float q: variance of the constant jerk model
    :param bool forwardbackward: indicates whether to run smoother forwards and backwards
        (usually achieves better estimate at end points)

    :return: - **x_hat** (np.array) -- estimated (smoothed) x
             - **dxdt_hat** (np.array) -- estimated derivative of x
    """
    if params is not None: # boilerplate backwards compatibility code
        warn("`params` and `options` parameters will be removed in a future version. Use `r`, " +
            "`q`, and `forwardbackward` instead.", DeprecationWarning)
        r, q = params
        if options is not None:
            if 'forwardbackward' in options: forwardbackward = options['forwardbackward']
    elif r is None or q is None:
        raise ValueError("`q` and `r` must be given.")

    warn("`constant_jerk` is deprecated. Call `rtsdiff` with order 3 instead.", DeprecationWarning)
    return rtsdiff(x, dt, 3, np.log10(q/r), forwardbackward)


def robustdiff(x, dt_or_t, order, log_q, log_r, proc_huberM=6, meas_huberM=0, axis=0):
    """Perform outlier-robust differentiation by solving the Maximum A Priori optimization problem:
    :math:`\\text{argmin}_{\\{x_n\\}} \\sum_{n=0}^{N-1} V(R^{-1/2}(y_n - C x_n)) + \\sum_{n=1}^{N-1} J(Q_{n-1}^{-1/2}(x_n - A_{n-1} x_{n-1}))`,
    where :math:`A,Q,C,R` come from an assumed constant derivative model and :math:`V,J` are the :math:`\\ell_1` norm or Huber
    loss rather than the :math:`\\ell_2` norm optimized by RTS smoothing. This problem is convex, so this method calls
    :code:`convex_smooth`, which in turn forms a sparse CVXPY problem and invokes CLARABEL.

    Note that for Huber losses, :code:`M` is the radius where the Huber loss function turns from quadratic to linear. Because
    all loss function inputs are normalized by noise level, :math:`q^{1/2}` or :math:`r^{1/2}`, :code:`M` is in units of inlier
    standard deviation. In other words, this choice affects which portion of inliers might be treated as outliers. For example,
    assuming Gaussian inliers, the portion beyond :math:`M\\sigma` is :code:`outlier_portion = 2*(1 - scipy.stats.norm.cdf(M))`.
    The inverse of this is :code:`M = scipy.stats.norm.ppf(1 - outlier_portion/2)`. As :math:`M \\to \\infty`, Huber becomes the
    1/2-sum-of-squares case, :math:`\\frac{1}{2}\\|\\cdot\\|_2^2`, because the normalization constant of the Huber loss (See
    :math:`c_2` in `section 6 of this paper <https://jmlr.org/papers/volume14/aravkin13a/aravkin13a.pdf>`_, missing a
    :math:`\\sqrt{\\cdot}` term there, see p2700) approaches 1 as :math:`M` increases. Similarly, as :code:`M` approaches 0,
    Huber reduces to the :math:`\\ell_1` norm case, because the normalization constant approaches :math:`\\frac{\\sqrt{2}}{M}`,
    cancelling the :math:`M` multiplying :math:`|\\cdot|` in the Huber function, and leaving behind :math:`\\sqrt{2}`, the
    proper :math:`\\ell_1` normalization.

    Note that :code:`log_q` and :code:`proc_huberM` are coupled, as are :code:`log_r` and :code:`meas_huberM`, via the relation
    :math:`\\text{Huber}(q^{-1/2}v, M) = q^{-1}\\text{Huber}(v, Mq^{1/2})`, but these are still independent enough that for
    the purposes of optimization we cannot collapse them. Nor can :code:`log_q` and :code:`log_r` be combined into
    :code:`log_qr_ratio` as in RTS smoothing without the addition of a new absolute scale parameter, becausee :math:`q` and
    :math:`r` interact with the distinct Huber :math:`M` parameters.

    :param np.array[float] x: data series to differentiate. May be multidimensional; see :code:`axis`.
    :param float or array[float] dt_or_t: This function supports variable step size. This parameter is either the constant
        :math:`\\Delta t` if given as a single float, or data locations if given as an array of same length as :code:`x`.
    :param int order: which derivative to stabilize in the constant-derivative model (1=velocity, 2=acceleration, 3=jerk)
    :param float log_q: base 10 logarithm of process noise variance, so :code:`q = 10**log_q`
    :param float log_r: base 10 logarithm of measurement noise variance, so :code:`r = 10**log_r`
    :param float proc_huberM: quadratic-to-linear transition point for process loss
    :param float meas_huberM: quadratic-to-linear transition point for measurement loss
    :param int axis: data dimension along which differentiation is performed

    :return: - **x_hat** (np.array) -- estimated (smoothed) x, same shape as input :code:`x`
             - **dxdt_hat** (np.array) -- estimated derivative of x, same shape as input :code:`x`
    """
    N = x.shape[axis]
    if not np.isscalar(dt_or_t) and N != len(dt_or_t):
        raise ValueError("If `dt_or_t` is given as array-like, must have same length as `x` along `axis`.")

    A_c = np.diag(np.ones(order), 1) # continuous-time A just has 1s on the first diagonal (where 0th is main diagonal)
    Q_c = np.zeros(A_c.shape); Q_c[-1,-1] = 10**log_q # continuous-time uncertainty around the last derivative
    C = np.zeros((1, order+1)); C[0,0] = 1 # we measure only y = noisy x
    R = np.array([[10**log_r]]) # 1 observed state, so this is 1x1
    M = np.block([[A_c, Q_c], [np.zeros(A_c.shape), -A_c.T]])  # exponentiate per step

    if np.isscalar(dt_or_t): # convert to discrete time using matrix exponential
        eM = expm(M * dt_or_t)
        A_d = eM[:order+1, :order+1]
        Q_d = eM[:order+1, order+1:] @ A_d.T
        if np.linalg.cond(Q_d) > 1e12: Q_d += np.eye(order+1)*1e-12 # for numerical stability with convex solver. Doesn't change answers appreciably (or at all).
    else: # support variable step size for this function
        A_d = np.empty((N-1, order+1, order+1))
        Q_d = np.empty_like(A_d)
        for n,dt in enumerate(np.diff(dt_or_t)):
            eM = expm(M * dt)
            A_d[n] = eM[:order+1, :order+1] # extract discrete time A matrix
            Q_d[n] = eM[:order+1, order+1:] @ A_d[n].T # extract discrete time Q matrix
            if np.linalg.cond(Q_d[n]) > 1e12: Q_d[n] += np.eye(order+1)*1e-12

    x_hat = np.empty_like(x); dxdt_hat = np.empty_like(x)

    for vec_idx in np.ndindex(x.shape[:axis] + x.shape[axis+1:]): # works properly for 1D case too
        s = vec_idx[:axis] + (slice(None),) + vec_idx[axis:]
        x_states = convex_smooth(x[s], A_d, Q_d, C, R, proc_huberM=proc_huberM, meas_huberM=meas_huberM) # outsource solution of the convex optimization problem
        x_hat[s] = x_states[:,0]; dxdt_hat[s] = x_states[:,1]

    return x_hat, dxdt_hat


def convex_smooth(y, A, Q, C, R, B=None, u=None, proc_huberM=6, meas_huberM=0):
    """Solve the optimization problem for robust smoothing using CVXPY.

    :param np.array[float] y: measurements
    :param np.array A: discrete-time state transition matrix. If 2D (:math:`m \\times m`), the same matrix is used for all steps;
        if 3D (:math:`N-1 \\times m \\times m`), :code:`A[n-1]` is used for the transition from step :math:`n-1` to :math:`n`.
    :param np.array Q: discrete-time process noise covariance. Same 2D or 3D shape convention as :code:`A`.
    :param np.array C: measurement matrix
    :param np.array R: measurement noise covariance matrix
    :param np.array B: optional discrete-time control matrix, optionally stacked like :code:`A`.
    :param np.array u: optional control inputs, stacked in the direction of axis 0
    :param float proc_huberM: Huber loss parameter. :math:`M=0` reduces to :math:`\\sqrt{2}\\ell_1`.
    :param float meas_huberM: Huber loss parameter. :math:`M=\\infty` reduces to :math:`\\frac{1}{2}\\ell_2^2`.
    
    :return: (np.array) -- state estimates (state_dim x N)
    """
    N = len(y)
    state_dim = A.shape[-1]
    x_states = cvxpy.Variable((state_dim, N)) # each column is [position, velocity, acceleration, ...] at step n
    control = isinstance(B, np.ndarray) and isinstance(u, np.ndarray) # whether there is a control input

    if A.ndim == 3: # It is extremely important to runtime that CVXPY expressions be in vectorized form
        Ax = cvxpy.einsum('nij,jn->in', A, x_states[:, :-1]) # multipy each A matrix by the corresponding x_states at that time step
        Q_inv_sqrts = np.array([np.linalg.inv(sqrtm(Q[n])) for n in range(N-1)]) # precompute Q^(-1/2) for each time step
        Bu = 0 if not control else cvxpy.einsum('nij,nj->in', B, u[1:]) if B.ndim == 3 else B @ u[1:].T
        proc_resids = cvxpy.einsum('nij,jn->in', Q_inv_sqrts, x_states[:,1:] - Ax - Bu)
    else: # all Q^(-1/2)(x_n - (A x_{n-1} + B u_n))
        proc_resids = np.linalg.inv(sqrtm(Q)) @ (x_states[:,1:] - A @ x_states[:,:-1] - (0 if not control else B @ u[1:].T))
    
    obs = ~np.isnan(y) # boolean mask of non-NaN observations
    meas_resids = np.linalg.inv(sqrtm(R)) @ (y[obs].reshape(C.shape[0],-1) - C @ x_states[:,obs]) # all R^(-1/2)(y_n - C x_n)

    # Process terms: sum of J(proc_resids)
    objective = 0.5*cvxpy.sum_squares(proc_resids) if proc_huberM == float('inf') \
                else np.sqrt(2)*cvxpy.sum(cvxpy.abs(proc_resids)) if proc_huberM < 1e-3 \
                else huber_const(proc_huberM)*cvxpy.sum(0.5*cvxpy.huber(proc_resids, proc_huberM)) # 1/2 l2^2, l1, or Huber
    # Measurement terms: sum of V(meas_resids)
    objective += 0.5*cvxpy.sum_squares(meas_resids) if meas_huberM == float('inf') \
                else np.sqrt(2)*cvxpy.sum(cvxpy.abs(meas_resids)) if meas_huberM < 1e-3 \
                else huber_const(meas_huberM)*cvxpy.sum(0.5*cvxpy.huber(meas_resids, meas_huberM))
    # CVXPY quirks: norm(, 1) != sum(abs()) for matrices. And huber() is defined as twice the magnitude of the canonical
    # function https://www.cvxpy.org/api_reference/cvxpy.atoms.elementwise.html#huber, so correct with a factor of 0.5.

    problem = cvxpy.Problem(cvxpy.Minimize(objective))
    try: problem.solve(solver=cvxpy.CLARABEL, canon_backend=cvxpy.SCIPY_CANON_BACKEND)
    except cvxpy.error.SolverError: pass # Could try another solver here, like SCS, but slows things down

    if x_states.value is None: return np.full((N, state_dim), np.nan) # There can be solver failure, even without error
    return x_states.value.T
