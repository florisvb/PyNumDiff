"""
This module implements Kalman filters
"""
import copy
import numpy as np

from pynumdiff.linear_model import savgoldiff

####################
# Helper functions #
####################


def __kalman_forward_update__(xhat_fm, P_fm, y, u, A, B, C, R, Q):
    """
    :param xhat_fm:
    :param P_fm:
    :param y:
    :param u:
    :param A:
    :param B:
    :param C:
    :param R:
    :param Q:
    :return:
    """
    I = np.matrix(np.eye(A.shape[0]))
    gammaW = np.matrix(np.eye(A.shape[0]))

    K_f = P_fm*C.T*(C*P_fm*C.T + R).I

    xhat_fp = xhat_fm + K_f*(y - C*xhat_fm)

    P_fp = (I - K_f*C)*P_fm

    xhat_fm = A*xhat_fp + B*u

    P_fm = A*P_fp*A.T + gammaW*Q*gammaW.T

    return xhat_fp, xhat_fm, P_fp, P_fm


def __kalman_forward_filter__(xhat_fm, P_fm, y, u, A, B, C, R, Q):
    """
    :param xhat_fm:
    :param P_fm:
    :param y:
    :param u:
    :param A:
    :param B:
    :param C:
    :param R:
    :param Q:
    :return:
    """
    assert isinstance(xhat_fm, np.matrix)
    assert isinstance(P_fm, np.matrix)
    assert isinstance(A, np.matrix)
    assert isinstance(B, np.matrix)
    assert isinstance(C, np.matrix)
    assert isinstance(R, np.matrix)
    assert isinstance(Q, np.matrix)
    assert isinstance(y, np.matrix)
    if u is None:
        u = np.matrix(np.zeros([B.shape[1], y.shape[1]]))
    assert isinstance(u, np.matrix)

    xhat_fp = None
    P_fp = []
    P_fm = [P_fm]

    for i in range(y.shape[1]):
        _xhat_fp, _xhat_fm, _P_fp, _P_fm = __kalman_forward_update__(xhat_fm[:, -1], P_fm[-1], y[:, i], u[:, i],
                                                                     A, B, C, R, Q)
        if xhat_fp is None:
            xhat_fp = _xhat_fp
        else:
            xhat_fp = np.hstack((xhat_fp, _xhat_fp))
        xhat_fm = np.hstack((xhat_fm, _xhat_fm))

        P_fp.append(_P_fp)
        P_fm.append(_P_fm)

    return xhat_fp, xhat_fm, P_fp, P_fm


def __kalman_backward_smooth__(xhat_fp, xhat_fm, P_fp, P_fm, A):
    """
    :param xhat_fp:
    :param xhat_fm:
    :param P_fp:
    :param P_fm:
    :param A:
    :return:
    """
    N = xhat_fp.shape[1]

    xhat_smooth = copy.copy(xhat_fp)
    P_smooth = copy.copy(P_fp)
    for t in range(N-2, -1, -1):
        L = P_fp[t]*A.T*P_fm[t].I
        xhat_smooth[:, t] = xhat_fp[:, t] + L*(xhat_smooth[:, t+1] - xhat_fm[:, t+1])
        P_smooth[t] = P_fp[t] - L*(P_smooth[t+1] - P_fm[t+1])

    return xhat_smooth, P_smooth


#####################
# Constant Velocity #
#####################


def __constant_velocity__(x, dt, params, options=None):
    """
    Run a forward-backward constant acceleration RTS Kalman smoother to estimate the derivative.

    :param x: (np.array of floats, 1xN) time series to differentiate
    :param dt: (float) time step size
    :param params: (list)  [r, : (float) covariance of the x noise
                            q] : (float) covariance of the constant velocity model
    :param options: (dict) {'backward'} : (bool) run smoother backwards in time
    :return:
    """
    if options is None:
        options = {'backward': False}

    r, q = params

    A = np.matrix([[1, dt], [0, 1]])
    B = np.matrix([[0], [0]])
    C = np.matrix([[1, 0]])
    R = np.matrix([[r]])
    Q = np.matrix([[1e-16, 0], [0, q]])
    x0 = np.matrix([[x[0]], [0]])
    P0 = np.matrix(100*np.eye(2))
    y = np.matrix(x)
    u = None

    if options['backward']:
        A = A.I
        y = y[:, ::-1]

    xhat_fp, xhat_fm, P_fp, P_fm = __kalman_forward_filter__(x0, P0, y, u, A, B, C, R, Q)
    xhat_smooth, _ = __kalman_backward_smooth__(xhat_fp, xhat_fm, P_fp, P_fm, A)

    x_hat = np.ravel(xhat_smooth[0, :])
    dxdt_hat = np.ravel(xhat_smooth[1, :])

    if not options['backward']:
        return x_hat, dxdt_hat

    return x_hat[::-1], dxdt_hat[::-1]


def constant_velocity(x, dt, params, options=None):
    """
    Run a forward-backward constant velocity RTS Kalman smoother to estimate the derivative.

    :param x: array of time series to differentiate
    :type x: np.array (float)

    :param dt: time step size
    :type dt: float

    :param params: a list of two elements:

                    - r: covariance of the x noise
                    - q: covariance of the constant velocity model

    :type params: list (float)


    :param options: a dictionary indicating whether to run smoother forwards and backwards
                    (usually achieves better estimate at end points)
    :type params: dict {'forwardbackward': boolean}, optional

    :return: a tuple consisting of:

            - x_hat: estimated (smoothed) x
            - dxdt_hat: estimated derivative of x


    :rtype: tuple -> (np.array, np.array)
    """
    if options is None:
        options = {'forwardbackward': True}

    if options['forwardbackward']:
        x_hat_f, smooth_dxdt_hat_f = __constant_velocity__(x, dt, params, options={'backward': False})
        x_hat_b, smooth_dxdt_hat_b = __constant_velocity__(x, dt, params, options={'backward': True})

        w = np.arange(0, len(x_hat_f), 1)
        w = w/np.max(w)

        x_hat = x_hat_f*w + x_hat_b*(1-w)
        smooth_dxdt_hat = smooth_dxdt_hat_f*w + smooth_dxdt_hat_b*(1-w)

        smooth_dxdt_hat_corrected = np.mean((smooth_dxdt_hat, smooth_dxdt_hat_f), axis=0)

        return x_hat, smooth_dxdt_hat_corrected

    return __constant_velocity__(x, dt, params, options={'backward': False})


#########################
# Constant Acceleration #
#########################


def __constant_acceleration__(x, dt, params, options=None):
    """
    Run a forward-backward constant acceleration RTS Kalman smoother to estimate the derivative.

    :param x: array of time series to differentiate
    :type x: np.array (float)

    :param dt: time step size
    :type dt: float

    :param params: a list of two elements:

                    - r: covariance of the x noise
                    - q: covariance of the constant velocity model

    :type params: list (float)


    :param options: a dictionary indicating whether to run smoother backwards in time
    :type params: dict {'backward': boolean}, optional

    :return: a tuple consisting of:

            - x_hat: estimated (smoothed) x
            - dxdt_hat: estimated derivative of x

    :rtype: tuple -> (np.array, np.array)
    """

    if options is None:
        options = {'backward': False}

    r, q = params
    A = np.matrix([[1, dt, 0],
                   [0, 1, dt],
                   [0, 0,  1]])
    B = np.matrix([[0], [0], [0]])
    C = np.matrix([[1, 0, 0]])
    R = np.matrix([[r]])
    Q = np.matrix([[1e-16, 0, 0],
                   [0, 1e-16, 0],
                   [0,     0, q]])
    x0 = np.matrix([[x[0]], [0], [0]])
    P0 = np.matrix(10*np.eye(3))
    y = np.matrix(x)
    u = None

    if options['backward']:
        A = A.I
        y = y[:, ::-1]

    xhat_fp, xhat_fm, P_fp, P_fm = __kalman_forward_filter__(x0, P0, y, u, A, B, C, R, Q)
    xhat_smooth, _ = __kalman_backward_smooth__(xhat_fp, xhat_fm, P_fp, P_fm, A)

    x_hat = np.ravel(xhat_smooth[0, :])
    dxdt_hat = np.ravel(xhat_smooth[1, :])

    if not options['backward']:
        return x_hat, dxdt_hat

    return x_hat[::-1], dxdt_hat[::-1]


def constant_acceleration(x, dt, params, options=None):
    """
    Run a forward-backward constant acceleration RTS Kalman smoother to estimate the derivative.

    :param x: array of time series to differentiate
    :type x: np.array (float)

    :param dt: time step size
    :type dt: float

    :param params: a list of two elements:

                    - r: covariance of the x noise
                    - q: covariance of the constant velocity model

    :type params: list (float)


    :param options: a dictionary indicating whether to run smoother forwards and backwards
                    (usually achieves better estimate at end points)
    :type params: dict {'forwardbackward': boolean}, optional

    :return: a tuple consisting of:

            - x_hat: estimated (smoothed) x
            - dxdt_hat: estimated derivative of x


    :rtype: tuple -> (np.array, np.array)
    """

    if options is None:
        options = {'forwardbackward': True}

    if options['forwardbackward']:
        x_hat_f, smooth_dxdt_hat_f = __constant_acceleration__(x, dt, params, options={'backward': False})
        x_hat_b, smooth_dxdt_hat_b = __constant_acceleration__(x, dt, params, options={'backward': True})

        w = np.arange(0, len(x_hat_f), 1)
        w = w/np.max(w)

        x_hat = x_hat_f*w + x_hat_b*(1-w)
        smooth_dxdt_hat = smooth_dxdt_hat_f*w + smooth_dxdt_hat_b*(1-w)

        smooth_dxdt_hat_corrected = np.mean((smooth_dxdt_hat, smooth_dxdt_hat_f), axis=0)

        return x_hat, smooth_dxdt_hat_corrected

    return __constant_acceleration__(x, dt, params, options={'backward': False})


#################
# Constant Jerk #
#################


def __constant_jerk__(x, dt, params, options=None):
    """
    Run a forward-backward constant jerk RTS Kalman smoother to estimate the derivative.

    :param x: array of time series to differentiate
    :type x: np.array (float)

    :param dt: time step size
    :type dt: float

    :param params: a list of two elements:

                    - r: covariance of the x noise
                    - q: covariance of the constant velocity model

    :type params: list (float)


    :param options: a dictionary indicating whether to run smoother backwards in time
    :type params: dict {'backward': boolean}, optional

    :return: a tuple consisting of:

            - x_hat: estimated (smoothed) x
            - dxdt_hat: estimated derivative of x

    :rtype: tuple -> (np.array, np.array)
    """

    if options is None:
        options = {'backward': False}

    r, q = params
    A = np.matrix([[1, dt, 0, 0],
                   [0, 1, dt, 0],
                   [0, 0,  1, dt],
                   [0, 0,  0, 1]])
    B = np.matrix([[0], [0], [0], [0]])
    C = np.matrix([[1, 0, 0, 0]])
    R = np.matrix([[r]])
    Q = np.matrix([[1e-16, 0, 0,     0],
                   [0, 1e-16, 0,     0],
                   [0,     0, 1e-16, 0],
                   [0,     0, 0,     q]])
    x0 = np.matrix([[x[0]], [0], [0], [0]])
    P0 = np.matrix(10*np.eye(4))
    y = np.matrix(x)
    u = None

    if options['backward']:
        A = A.I
        y = y[:, ::-1]

    xhat_fp, xhat_fm, P_fp, P_fm = __kalman_forward_filter__(x0, P0, y, u, A, B, C, R, Q)
    xhat_smooth, _ = __kalman_backward_smooth__(xhat_fp, xhat_fm, P_fp, P_fm, A)

    x_hat = np.ravel(xhat_smooth[0,:])
    dxdt_hat = np.ravel(xhat_smooth[1,:])

    if not options['backward']:
        return x_hat, dxdt_hat

    return x_hat[::-1], dxdt_hat[::-1]


def constant_jerk(x, dt, params, options=None):
    """
    Run a forward-backward constant jerk RTS Kalman smoother to estimate the derivative.

    :param x: array of time series to differentiate
    :type x: np.array (float)

    :param dt: time step size
    :type dt: float

    :param params: a list of two elements:

                    - r: covariance of the x noise
                    - q: covariance of the constant velocity model

    :type params: list (float)


    :param options: a dictionary indicating whether to run smoother forwards and backwards
                    (usually achieves better estimate at end points)
    :type params: dict {'forwardbackward': boolean}, optional

    :return: a tuple consisting of:

            - x_hat: estimated (smoothed) x
            - dxdt_hat: estimated derivative of x


    :rtype: tuple -> (np.array, np.array)
    """

    if options is None:
        options = {'forwardbackward': True}

    if options['forwardbackward']:
        x_hat_f, smooth_dxdt_hat_f = __constant_jerk__(x, dt, params, options={'backward': False})
        x_hat_b, smooth_dxdt_hat_b = __constant_jerk__(x, dt, params, options={'backward': True})

        w = np.arange(0, len(x_hat_f), 1)
        w = w/np.max(w)

        x_hat = x_hat_f*w + x_hat_b*(1-w)
        smooth_dxdt_hat = smooth_dxdt_hat_f*w + smooth_dxdt_hat_b*(1-w)

        smooth_dxdt_hat_corrected = np.mean((smooth_dxdt_hat, smooth_dxdt_hat_f), axis=0)

        return x_hat, smooth_dxdt_hat_corrected

    return __constant_jerk__(x, dt, params, options={'backward': False})


def known_dynamics(x, params, u=None, options=None):
    """
    Run a forward-backward constant acceleration RTS Kalman smoother to estimate the derivative.

    :param x: matrix of time series of (noisy) measurements
    :type x: np.matrix (float)

    :param params: a list of:
                    - x0: inital condition, matrix of Nx1, N = number of states
                    - P0: initial covariance matrix of NxN
                    - A: dynamics matrix, NxN
                    - B: control input matrix, NxM, M = number of measurements
                    - C: measurement dynamics, MxN
                    - R: covariance matrix for the measurements, MxM
                    - Q: covariance matrix for the model, NxN
    :type params: list (matrix)

    :param u: matrix of time series of control inputs
    :type u: np.matrix (float)

    :param options: a dictionary indicating whether to run smoother
    :type params: dict {'smooth': boolean}, optional

    :return: matrix:
            - xhat_smooth: smoothed estimates of the full state x

    :rtype: tuple -> (np.array, np.array)
    """
    if options is None:
        options = {'smooth': True}

    x0, P0, A, B, C, R, Q = params
    y = np.matrix(x)

    xhat_fp, xhat_fm, P_fp, P_fm = __kalman_forward_filter__(x0, P0, y, u, A, B, C, R, Q)
    xhat_smooth, _ = __kalman_backward_smooth__(xhat_fp, xhat_fm, P_fp, P_fm, A)

    if not options['smooth']:
        return xhat_fp

    return xhat_smooth



###################################################################################################
# Constant Acceleration with Savitzky-Golay pre-estimate (not worth the parameter tuning trouble) #
###################################################################################################


def __savgol_const_accel__(x, sg_dxdt_hat, dt, params, options=None):
    """
    Run a forward-backward constant acceleration RTS Kalman smoother to estimate the derivative.

    :param x: array of time series to differentiate
    :type x: np.array (float)

    :param sg_dxdt_hat: ? ? ?
    :type sg_dxdt_hat: np.array (float)

    :param dt: time step size
    :type dt: float

    :param params: a list of two elements:

                    - r: covariance of the x noise
                    - q: covariance of the constant velocity model

    :type params: list (float)


    :param options: a dictionary indicating whether to run smoother backwards in time
    :type params: dict {'backward': boolean}, optional

    :return: a tuple consisting of:

            - x_hat: estimated (smoothed) x
            - dxdt_hat: estimated derivative of x

    :rtype: tuple -> (np.array, np.array)
    """

    if options is None:
        options = {'backward': False}

    r1, r2, q = params
    A = np.matrix([[1, dt, 0],
                   [0, 1, dt],
                   [0, 0,  1]])
    B = np.matrix([[0], [0], [0]])
    C = np.matrix([[1, 0, 0],
                   [0, 1, 0]])
    R = np.matrix([[r1, 0],
                   [0, r2]])
    Q = np.matrix([[1e-16, 0, 0],
                   [0, 1e-16, 0],
                   [0,     0, q]])
    x0 = np.matrix([[x[0]], [sg_dxdt_hat[0]], [0]])
    P0 = np.matrix(10*np.eye(3))
    y = np.matrix(np.vstack((x, sg_dxdt_hat)))
    u = None

    if options['backward']:
        A = A.I
        y = y[:, ::-1]

    xhat_fp, xhat_fm, P_fp, P_fm = __kalman_forward_filter__(x0, P0, y, u, A, B, C, R, Q)
    xhat_smooth, _ = __kalman_backward_smooth__(xhat_fp, xhat_fm, P_fp, P_fm, A)

    x_hat = np.ravel(xhat_smooth[0, :])
    dxdt_hat = np.ravel(xhat_smooth[1, :])

    if not options['backward']:
        return x_hat, dxdt_hat

    return x_hat[::-1], dxdt_hat[::-1]


def savgol_const_accel(x, dt, params, options=None):
    """
    Run a forward-backward constant acceleration RTS Kalman smoother to estimate the derivative.

    :param x: array of time series to differentiate
    :type x: np.array (float)

    :param dt: time step size
    :type dt: float

    :param params: a list of two elements:

                    - r: covariance of the x noise
                    - q: covariance of the constant velocity model

    :type params: list (float)


    :param options: a dictionary indicating whether to run smoother forwards and backwards
                    (usually achieves better estimate at end points)
    :type params: dict {'forwardbackward': boolean}, optional

    :return: a tuple consisting of:

            - x_hat: estimated (smoothed) x
            - dxdt_hat: estimated derivative of x


    :rtype: tuple -> (np.array, np.array)
    """
    if options is None:
        options = {'forwardbackward': True}

    N, window_size, r1, r2, q = params

    _, sg_dxdt_hat = savgoldiff(x, dt, [N, window_size])

    if options['forwardbackward']:
        x_hat_f, smooth_dxdt_hat_f = __savgol_const_accel__(x, sg_dxdt_hat, dt, [r1, r2, q],
                                                            options={'backward': False})
        x_hat_b, smooth_dxdt_hat_b = __savgol_const_accel__(x, sg_dxdt_hat, dt, [r1, r2, q],
                                                            options={'backward': True})

        w = np.arange(0, len(x_hat_f), 1)
        w = w/np.max(w)

        x_hat = x_hat_f*w + x_hat_b*(1-w)
        smooth_dxdt_hat = smooth_dxdt_hat_f*w + smooth_dxdt_hat_b*(1-w)

        smooth_dxdt_hat_corrected = np.mean((smooth_dxdt_hat, smooth_dxdt_hat_f), axis=0)

        return x_hat, smooth_dxdt_hat_corrected

    return __constant_acceleration__(x, dt, params, options={'backward': False})
