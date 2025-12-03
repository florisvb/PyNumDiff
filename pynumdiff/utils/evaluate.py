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

    :return: (tuple) -- figure and axes
    """
    t = np.arange(len(x))*dt
    if xlim is None:
        xlim = [t[0], t[-1]]

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)

    axes[0].plot(t, x_truth, '--', color='black', linewidth=3, label=r"true $x$")
    axes[0].plot(t, x, '.', color='blue', zorder=-100, markersize=markersize, label=r"noisy data")
    axes[0].plot(t, x_hat, color='red', label=r"estimated $\hat{x}$")
    axes[0].set_ylabel('Position', fontsize=18)
    axes[0].set_xlabel('Time', fontsize=18)
    axes[0].set_xlim(xlim[0], xlim[-1])
    axes[0].tick_params(axis='x', labelsize=15)
    axes[0].tick_params(axis='y', labelsize=15)
    axes[0].legend(loc='lower right', fontsize=12)

    axes[1].plot(t, dxdt_truth, '--', color='black', linewidth=3, label=r"true  $\frac{dx}{dt}$")
    axes[1].plot(t, dxdt_hat, color='red', label=r"est. $\hat{\frac{dx}{dt}}$")
    axes[1].set_ylabel('Velocity', fontsize=18)
    axes[1].set_xlabel('Time', fontsize=18)
    axes[1].set_xlim(xlim[0], xlim[-1])
    axes[1].tick_params(axis='x', labelsize=15)
    axes[1].tick_params(axis='y', labelsize=15)
    axes[1].legend(loc='lower right', fontsize=12)

    if show_error:
        rmse_dxdt = rmse(dxdt_truth, dxdt_hat)
        R_sqr = error_correlation(dxdt_truth, dxdt_hat)
        axes[1].text(0.05, 0.95, f"RMSE = {rmse_dxdt:.2f}\n$R^2$ = {R_sqr:.2g}",
                     transform=axes[1].transAxes, fontsize=15, verticalalignment='top')

    return fig, axes


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
        rmse_dxdt = rmse(dxdt_truth, dxdt_hat)
        R_sqr = error_correlation(dxdt_truth, dxdt_hat)
        axes[i].text(0.05, 0.95, f"RMSE = {rmse_dxdt:.2f}\n$R^2$ = {R_sqr:.2g}",
                     transform=axes[i].transAxes, fontsize=15, verticalalignment='top')

    fig.tight_layout()


def robust_rme(u, v, padding=0, M=6):
    """Robustified/Huberized Root Mean Error metric, used to determine fit between smooth estimate and data.
    Equals np.linalg.norm(u[s] - v[s]) / np.sqrt(N) if M=float('inf'), and dang close for M=6 or even 2.

    :param np.array[float] u: e.g. noisy data
    :param np.array[float] v: e.g. estimated smoothed signal, reconstructed from derivative
    :param int padding: number of snapshots on either side of the array to ignore when calculating
        the metric. If :code:`'auto'`, defaults to 2.5% of the size of inputs
    :param float M: Huber loss parameter in units of ~1.4*mean absolute deviation of population of residual
        errors, intended to approximate standard deviation robustly.

    :return: (float) -- Robust root mean error between u and v
    """
    if padding == 'auto': padding = max(1, int(0.025*len(u)))
    s = slice(padding, len(u)-padding) # slice out data we want to measure

    sigma = stats.median_abs_deviation(u[s] - v[s], scale='normal') # M is in units of this robust scatter metric
    return np.sqrt(2*np.mean(utility.huber(u[s] - v[s], M*sigma)))


def rmse(u, v, padding=0):
    """True RMSE between vectors

    :param np.array[float] u: e.g. known true derivative 
    :param np.array[float] v: e.g. estimated derivative 
    :param int padding: number of snapshots on either side of the array to ignore when calculating
        the metric. If :code:`'auto'`, defaults to 2.5% of the size of inputs

    :return: **true_rmse** (float) -- RMS error between dxdt_hat and dxdt_truth, returns None if dxdt_hat is None
    """
    if padding == 'auto': padding = max(1, int(0.025*len(u)))
    s = slice(padding, len(u)-padding) # slice out data we want to measure
    N = s.stop - s.start

    return np.linalg.norm(u[s] - v[s]) / np.sqrt(N)


def error_correlation(u, v, padding=0):
    """Calculate the error correlation (pearsons correlation coefficient) between the first vector and the
    difference between the two vectors.

    :param np.array[float] u: e.g. true value of dxdt, if known, optional
    :param np.array[float] v: e.g. estimated dxdt_hat
    :param int padding: number of snapshots on either side of the array to ignore when calculating
        the metric. If :code:`'auto'`, defaults to 2.5% of the size of inputs

    :return: (float) -- r-squared correlation coefficient
    """
    if padding == 'auto': padding = max(1, int(0.025*len(u)))
    s = slice(padding, len(u)-padding) # slice out data we want to measure

    return stats.linregress(u[s], v[s] - u[s]).rvalue**2


def total_variation(x, padding=0):
    """Calculate the total variation of an array. Used by optimizer.

    :param np.array[float] x: data
    :param int padding: number of snapshots on either side of the array to ignore when calculating
        the metric. If :code:`'auto'`, defaults to 2.5% of the size of x

    :return: (float) -- total variation
    """
    if padding == 'auto': padding = max(1, int(0.025*len(x)))
    x = x[padding:len(x)-padding]

    return np.linalg.norm(x[1:]-x[:-1], 1)/len(x) # normalized version of what cvxpy.tv does
