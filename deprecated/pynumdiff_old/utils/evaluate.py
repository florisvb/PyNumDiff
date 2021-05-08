import numpy as _np 
import matplotlib.pyplot as _plt 
import scipy.stats as _scipy_stats

# local imports
from pynumdiff.utils import utility as _utility
_finite_difference = _utility.finite_difference

def plot(x, dt, x_hat, dxdt_hat, x_truth, dxdt_truth, xlim=None, ax_x=None, ax_dxdt=None, show_error=True, markersize=5):
    t = _np.arange(0, dt*len(x), dt)
    if ax_x is None and ax_dxdt is None:
        fig = _plt.figure(figsize=(20,6))
        ax_x = fig.add_subplot(121)
        ax_dxdt = fig.add_subplot(122)

    if xlim is None:
        xlim = [t[0], t[-1]]

    if ax_x is not None:
        if x_hat is not None:
            ax_x.plot(t, x_hat, color='red')
        ax_x.plot(t, x_truth, '--', color='black')
        ax_x.plot(t, x, '.', color='blue', zorder=-100, markersize=markersize)
        ax_x.set_ylabel('Position')
        ax_x.set_xlabel('Time')
        ax_x.set_xlim(xlim[0], xlim[-1])

    if ax_dxdt is not None:
        ax_dxdt.plot(t, dxdt_hat, color='red')
        ax_dxdt.plot(t, dxdt_truth, '--', color='black')
        ax_dxdt.set_ylabel('Velocity')
        ax_dxdt.set_xlabel('Time')
        ax_dxdt.set_xlim(xlim[0], xlim[-1])

    if show_error:
        rms_rec_x, rms_x, rms_dxdt = metrics(x, dt, x_hat, dxdt_hat, x_truth, dxdt_truth)
        print('RMS error in velocity: ', rms_dxdt)

def __rms_error__(a, e):
    if _np.max(_np.abs(a-e)) > 1e16:
        return 1e16
    s_error = _np.ravel((a - e))**2
    ms_error = _np.mean(s_error)
    rms_error = _np.sqrt(ms_error)
    return rms_error

def metrics(x, dt, x_hat, dxdt_hat, x_truth=None, dxdt_truth=None, padding=None):
    if padding is None or padding == 'auto':
        padding = int(0.025*len(x))
        if padding < 1:
            padding = 1
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
    if padding is None or padding == 'auto':
        padding = int(0.025*len(dxdt_hat))
        if padding < 1:
            padding = 1
    errors = (dxdt_hat[padding:-padding] - dxdt_truth[padding:-padding])
    r = _scipy_stats.linregress( dxdt_truth[padding:-padding]- _np.mean(dxdt_truth[padding:-padding]), errors)
    return r.rvalue**2

def rmse(dxdt_hat, dxdt_truth, padding=None):
    if padding is None or padding == 'auto':
        padding = int(0.025*len(dxdt_hat))
        if padding < 1:
            padding = 1
    rmse = _np.sqrt(_np.mean((dxdt_hat[padding:-padding] - dxdt_truth[padding:-padding])**2))
    return rmse