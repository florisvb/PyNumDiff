"""Some tools to help evaluate and plot performance, used in optimization and in jupyter notebooks"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from pynumdiff.utils import utility

# pylint: disable-msg=too-many-locals, too-many-arguments
def plot(x, dt, x_hat, dxdt_hat, x_truth, dxdt_truth, xlim=None, show_error=True, markersize=5):
    """Make comparison plots of 'x (blue) vs x_truth (black) vs x_hat (red)' and 'dxdt_truth
    (black) vs dxdt_hat (red)'

    :param np.array[float] x: array of noisy data
    :param float dt: a float number representing the step size
    :param np.array[float] x_hat: array of smoothed estimation of x
    :param np.array[float] dxdt_hat: array of estimated derivative
    :param np.array[float] x_truth: array of noise-free time series
    :param np.array[float] dxdt_truth: array of true derivative
    :param list[int] xlim: a list specifying range of x
    :param bool show_error: whether to show the rmse
    :param int markersize: marker size of noisy observations

    :return: Display two plots
    """
    t = np.arange(0, dt*len(x), dt)
    if xlim is None:
        xlim = [t[0], t[-1]]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    axes[0].plot(t, x_truth, '--', color='black', linewidth=3, label=r"true $x$")
    axes[0].plot(t, x, '.', color='blue', zorder=-100, markersize=markersize, label=r"noisy data")
    axes[0].plot(t, x_hat, color='red', label=r"estimated $\hat{x}$")
    axes[0].set_ylabel('Position', fontsize=18)
    axes[0].set_xlabel('Time', fontsize=18)
    axes[0].set_xlim(xlim[0], xlim[-1])
    axes[0].tick_params(axis='x', labelsize=15)
    axes[0].tick_params(axis='y', labelsize=15)
    axes[0].legend(loc='lower right', fontsize=12)
    axes[0].set_rasterization_zorder(0)

    axes[1].plot(t, dxdt_truth, '--', color='black', linewidth=3, label=r"true  $\frac{dx}{dt}$")
    axes[1].plot(t, dxdt_hat, color='red', label=r"est. $\hat{\frac{dx}{dt}}$")
    axes[1].set_ylabel('Velocity', fontsize=18)
    axes[1].set_xlabel('Time', fontsize=18)
    axes[1].set_xlim(xlim[0], xlim[-1])
    axes[1].tick_params(axis='x', labelsize=15)
    axes[1].tick_params(axis='y', labelsize=15)
    axes[1].legend(loc='lower right', fontsize=12)
    axes[1].set_rasterization_zorder(0)

    fig.tight_layout()

    if show_error:
        _, _, rms_dxdt = rmse(x, dt, x_hat, dxdt_hat, x_truth, dxdt_truth)
        R_sqr = error_correlation(dxdt_hat, dxdt_truth)
        print('RMS error in velocity: ', rms_dxdt)
        print('Error correlation: ', R_sqr)


def plot_comparison(dt, dxdt_truth, dxdt_hat1, title1, dxdt_hat2, title2, dxdt_hat3, title3):
    """This is intended to show method performances with different choices of parameter"""
    t = np.arange(0, dt*len(dxdt_truth), dt)
    fig, axes = plt.subplots(1, 3, figsize=(22,6))

    for i,(dxdt_hat,title) in enumerate(zip([dxdt_hat1, dxdt_hat2, dxdt_hat3], [title1, title2, title3])):
        axes[i].plot(t, dxdt_truth, '--', color='black', linewidth=3, label=r"true  $\frac{dx}{dt}$")
        axes[i].plot(t, dxdt_hat, color='red', label=r"est. $\hat{\frac{dx}{dt}}$")
        if i==0: axes[i].set_ylabel('Velocity', fontsize=18)
        axes[i].set_xlabel('Time', fontsize=18)
        axes[i].tick_params(axis='x', labelsize=15)
        axes[i].tick_params(axis='y', labelsize=15)
        axes[i].set_title(title, fontsize=18)
        if i==2: axes[i].legend(loc='lower right', fontsize=12)

    fig.tight_layout()


def rmse(x, dt, x_hat, dxdt_hat, x_truth=None, dxdt_truth=None, padding=0):
    """Evaluate x_hat based on RMSE, calculating different ones depending on whether :code:`dxdt_truth`
    and :code:`x_truth` are known.

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
