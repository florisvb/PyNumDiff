"""
Metrics and evaluations?
"""
import numpy as _np
import matplotlib.pyplot as _plt
import scipy.stats as _scipy_stats

# local imports
from pynumdiff.utils import utility as _utility
_finite_difference = _utility.finite_difference


# pylint: disable-msg=too-many-locals, too-many-arguments
def plot(x, dt, x_hat, dxdt_hat, x_truth, dxdt_truth, xlim=None, ax_x=None, ax_dxdt=None,
         show_error=True, markersize=5):
    """
    Make comparison plots of 'x (blue) vs x_truth (black) vs x_hat (red)' and
    'dxdt_truth (black) vs dxdt_hat (red)'

    :param x: array of noisy time series
    :type x: np.array (float)

    :param dt: a float number representing the time step size
    :type dt: float

    :param x_hat: array of smoothed estimation of x
    :type x_hat: np.array (float)

    :param dxdt_hat: array of estimated derivative
    :type dxdt_hat: np.array (float)

    :param x_truth: array of noise-free time series
    :type x_truth: np.array (float)

    :param dxdt_truth: array of true derivative
    :type dxdt_truth: np.array (float)

    :param xlim: a list specifying range of x
    :type xlim: list (2 integers), optional

    :param ax_x: axis of the first plot
    :type ax_x: :class:`matplotlib.axes`, optional

    :param ax_dxdt: axis of the second plot
    :type ax_dxdt: :class:`matplotlib.axes`, optional

    :param show_error: whether to show the rmse
    :type show_error: boolean, optional

    :param markersize: marker size of noisy observations
    :type markersize: int, optional

    :return: Display two plots
    :rtype: None
    """
    t = _np.arange(0, dt*len(x), dt)
    if ax_x is None and ax_dxdt is None:
        fig = _plt.figure(figsize=(20, 6))
        ax_x = fig.add_subplot(121)
        ax_dxdt = fig.add_subplot(122)

    if xlim is None:
        xlim = [t[0], t[-1]]

    if ax_x is not None:
        if x_hat is not None:
            ax_x.plot(t, x_hat, color='red')
        ax_x.plot(t, x_truth, '--', color='black')
        ax_x.plot(t, x, '.', color='blue', zorder=-100, markersize=markersize)
        ax_x.set_ylabel('Position', fontsize=20)
        ax_x.set_xlabel('Time', fontsize=20)
        ax_x.set_xlim(xlim[0], xlim[-1])
        ax_x.tick_params(axis='x', labelsize=15)
        ax_x.tick_params(axis='y', labelsize=15)
        ax_x.set_rasterization_zorder(0)

    if ax_dxdt is not None:
        ax_dxdt.plot(t, dxdt_hat, color='red')
        ax_dxdt.plot(t, dxdt_truth, '--', color='black', linewidth=3)
        ax_dxdt.set_ylabel('Velocity', fontsize=20)
        ax_dxdt.set_xlabel('Time', fontsize=20)
        ax_dxdt.set_xlim(xlim[0], xlim[-1])
        ax_dxdt.tick_params(axis='x', labelsize=15)
        ax_dxdt.tick_params(axis='y', labelsize=15)
        ax_dxdt.set_rasterization_zorder(0)

    if show_error:
        _, _, rms_dxdt = metrics(x, dt, x_hat, dxdt_hat, x_truth, dxdt_truth)
        print('RMS error in velocity: ', rms_dxdt)


def __rms_error__(a, e):
    """
    Calculate rms error

    :param a: the first array
    :param e: the second array
    :return: a float number representing the rms error
    """
    if _np.max(_np.abs(a-e)) > 1e16:
        return 1e16
    s_error = _np.ravel((a - e))**2
    ms_error = _np.mean(s_error)
    rms_error = _np.sqrt(ms_error)
    return rms_error


def metrics(x, dt, x_hat, dxdt_hat, x_truth=None, dxdt_truth=None, padding=None):
    """
    Evaluate x_hat based on various metrics, depending on whether dxdt_truth and x_truth are known or not.

    :param x: time series that was differentiated
    :type x: np.array

    :param dt: time step in seconds
    :type dt: float

    :param x_hat: estimated (smoothed) x
    :type x_hat: np.array

    :param dxdt_hat: estimated xdot
    :type dxdt_hat: np.array

    :param x_truth: true value of x, if known, optional
    :type x_truth: np.array like x or None

    :param dxdt_truth: true value of dxdt, if known, optional
    :type dxdt_truth: np.array like x or None

    :param padding: number of snapshots on either side of the array to ignore when calculating the metric. If autor or None, defaults to 2.5% of the size of x
    :type padding: int, None, or auto

    :return: a tuple containing the following:
            - rms_rec_x: RMS error between the integral of dxdt_hat and x
            - rms_x: RMS error between x_hat and x_truth, returns None if x_truth is None
            - rms_dxdt: RMS error between dxdt_hat and dxdt_truth, returns None if dxdt_hat is None
    :rtype: tuple -> (float, float, float)

    """
    if padding is None or padding == 'auto':
        padding = int(0.025*len(x))
        padding = max(padding, 1)
    if _np.isnan(x_hat).any():
        return _np.nan, _np.nan, _np.nan

    # RMS dxdt
    if dxdt_truth is not None:
        rms_dxdt = __rms_error__(dxdt_hat[padding:-padding], dxdt_truth[padding:-padding])
    else:
        rms_dxdt = None

    # RMS x
    if x_truth is not None:
        rms_x = __rms_error__(x_hat[padding:-padding], x_truth[padding:-padding])
    else:
        rms_x = None

    # RMS reconstructed x
    rec_x = _utility.integrate_dxdt_hat(dxdt_hat, dt)
    x0 = _utility.estimate_initial_condition(x, rec_x)
    rec_x = rec_x + x0
    rms_rec_x = __rms_error__(rec_x[padding:-padding], x[padding:-padding])

    return rms_rec_x, rms_x, rms_dxdt


def error_correlation(dxdt_hat, dxdt_truth, padding=None):
    """
    Calculate the error correlation (pearsons correlation coefficient) between the estimated dxdt and true dxdt

    :param dxdt_hat: estimated xdot
    :type dxdt_hat: np.array

    :param dxdt_truth: true value of dxdt, if known, optional
    :type dxdt_truth: np.array like x or None

    :param padding: number of snapshots on either side of the array to ignore when calculating the metric. If autor or None, defaults to 2.5% of the size of x
    :type padding: int, None, or auto

    :return: r-squared correlation coefficient
    :rtype: float

    """
    if padding is None or padding == 'auto':
        padding = int(0.025*len(dxdt_hat))
        padding = max(padding, 1)
    errors = (dxdt_hat[padding:-padding] - dxdt_truth[padding:-padding])
    r = _scipy_stats.linregress(dxdt_truth[padding:-padding] -
                                _np.mean(dxdt_truth[padding:-padding]), errors)
    return r.rvalue**2


def rmse(dxdt_hat, dxdt_truth, padding=None):
    """
    Calculate the Root Mean Squared Error between the estimated dxdt and true dxdt

    :param dxdt_hat: estimated xdot
    :type dxdt_hat: np.array

    :param dxdt_truth: true value of dxdt, if known, optional
    :type dxdt_truth: np.array like x or None

    :param padding: number of snapshots on either side of the array to ignore when calculating the metric. If autor or None, defaults to 2.5% of the size of x
    :type padding: int, None, or auto

    :return: Root Mean Squared Error
    :rtype: float
    """
    if padding is None or padding == 'auto':
        padding = int(0.025*len(dxdt_hat))
        padding = max(padding, 1)
    RMSE = _np.sqrt(_np.mean((dxdt_hat[padding:-padding] - dxdt_truth[padding:-padding])**2))
    return RMSE
