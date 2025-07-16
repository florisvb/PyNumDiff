"""Some tools to help evaluate and plot performance, used in optimization and in jupyter notebooks"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from pynumdiff.utils import utility


# pylint: disable-msg=too-many-locals, too-many-arguments
def plot(x, dt, x_hat, dxdt_hat, x_truth, dxdt_truth, xlim=None, ax_x=None, ax_dxdt=None,
         show_error=True, markersize=5):
    """Make comparison plots of 'x (blue) vs x_truth (black) vs x_hat (red)' and 'dxdt_truth
    (black) vs dxdt_hat (red)'

    :param np.array[float] x: array of noisy data
    :param float dt: a float number representing the step size
    :param np.array[float] x_hat: array of smoothed estimation of x
    :param np.array[float] dxdt_hat: array of estimated derivative
    :param np.array[float] x_truth: array of noise-free time series
    :param np.array[float] dxdt_truth: array of true derivative
    :param list[int] xlim: a list specifying range of x
    :param matplotlib.axes ax_x: axis of the first plot
    :param matplotlib.axes ax_dxdt: axis of the second plot
    :param bool show_error: whether to show the rmse
    :param int markersize: marker size of noisy observations

    :return: Display two plots
    """
    t = np.arange(0, dt*len(x), dt)
    if ax_x is None and ax_dxdt is None:
        fig = plt.figure(figsize=(20, 6))
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


def metrics(x, dt, x_hat, dxdt_hat, x_truth=None, dxdt_truth=None, padding=0):
    """Evaluate x_hat based on various metrics, depending on whether dxdt_truth and x_truth are known or not.

    :param np.array[float] x: data that was differentiated
    :param float dt: step size
    :param np.array[float] x_hat: estimated (smoothed) x
    :param np.array[float] dxdt_hat: estimated xdot
    :param np.array[float] x_truth: true value of x, if known
    :param np.array[float] dxdt_truth: true value of dxdt, if known, optional
    :param int padding: number of snapshots on either side of the array to ignore when calculating
        the metric. If :code:`'auto'`, defaults to 2.5% of the size of x

    :return: tuple[float, float, float] containing\n
            - **rms_rec_x** -- RMS error between the integral of dxdt_hat and x
            - **rms_x** -- RMS error between x_hat and x_truth, returns None if x_truth is None
            - **rms_dxdt** -- RMS error between dxdt_hat and dxdt_truth, returns None if dxdt_hat is None
    """
    if np.isnan(x_hat).any():
        return np.nan, np.nan, np.nan
    if padding == 'auto':
        padding = int(0.025*len(x))
        padding = max(padding, 1)
    s = slice(padding, len(x)-padding) # slice out data we want to measure

    # RMS of dxdt and x_hat
    root = np.sqrt(s.stop - s.start)
    rms_dxdt = np.linalg.norm(dxdt_hat[s] - dxdt_truth[s]) / root if dxdt_truth is not None else None
    rms_x = np.linalg.norm(x_hat[s] - x_truth[s]) / root if x_truth is not None else None

    # RMS reconstructed x from integrating dxdt vs given raw x, available even in the absence of ground truth
    rec_x = utility.integrate_dxdt_hat(dxdt_hat, dt)
    x0 = utility.estimate_integration_constant(x, rec_x)
    rec_x = rec_x + x0
    rms_rec_x = np.linalg.norm(rec_x[s] - x[s]) / root

    return rms_rec_x, rms_x, rms_dxdt


def error_correlation(dxdt_hat, dxdt_truth, padding=0):
    """Calculate the error correlation (pearsons correlation coefficient) between the estimated
    dxdt and true dxdt

    :param np.array[float] dxdt_hat: estimated xdot
    :param np.array[float] dxdt_truth: true value of dxdt, if known, optional
    :param int padding: number of snapshots on either side of the array to ignore when calculating
        the metric. If :code:`'auto'`, defaults to 2.5% of the size of x

    :return: (float) -- r-squared correlation coefficient
    """
    if padding == 'auto':
        padding = int(0.025*len(dxdt_hat))
        padding = max(padding, 1)
    s = slice(padding, len(dxdt_hat)-padding) # slice out data we want to measure
    errors = (dxdt_hat[s] - dxdt_truth[s])
    r = stats.linregress(dxdt_truth[s] - np.mean(dxdt_truth[s]), errors)
    return r.rvalue**2


def total_variation(x, padding=0):
    """Calculate the total variation of an array. Used by optimizer.

    :param np.array[float] x: data
    :param int padding: number of snapshots on either side of the array to ignore when calculating
        the metric. If :code:`'auto'`, defaults to 2.5% of the size of x

    :return: (float) -- total variation
    """
    if np.isnan(x).any():
        return np.nan
    if padding == 'auto':
        padding = int(0.025*len(x))
        padding = max(padding, 1)
    x = x[padding:len(x)-padding]
    
    return np.linalg.norm(x[1:]-x[:-1], 1)/len(x) # normalized version of what cvxpy.tv does
