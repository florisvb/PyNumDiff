import numpy as _np 
import matplotlib.pyplot as _plt 

# local imports
from pynumdiff.utils import utility as _utility
_finite_difference = _utility.finite_difference

def plot(x, dt, x_hat, dxdt_hat, x_truth, dxdt_truth, xlim=None, ax_x=None, ax_dxdt=None, show_error=True):
    if ax_x is None and ax_dxdt is None:
        fig = _plt.figure(figsize=(20,6))
        ax_x = fig.add_subplot(121)
        ax_dxdt = fig.add_subplot(122)

    if xlim is None:
        xlim = [0, len(x_truth)]

    ax_x.plot(x_hat, color='blue')
    ax_x.plot(x_truth, '--', color='orange')
    ax_x.plot(x, '*', color='green', zorder=-100)
    ax_x.set_ylabel('Position')
    ax_x.set_xlabel('Time')

    
    ax_dxdt.plot(dxdt_hat, color='blue')
    ax_dxdt.plot(dxdt_truth, '--', color='orange')
    ax_dxdt.set_ylabel('Velocity')
    ax_x.set_xlabel('Time')

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
    if _np.isnan(x_hat).any():
        return _np.nan, _np.nan, _np.nan

    # RMS dxdt
    if dxdt_truth is not None:
        rms_dxdt = __rms_error__(dxdt_hat, dxdt_truth)
    else:
        rms_dxdt = None

    # RMS x
    if x_truth is not None:
        rms_x = __rms_error__(x_hat, x_truth)
    else:
        rms_x = None

    # RMS reconstructed x
    rec_x = _utility.integrate_dxdt_hat(dxdt_hat, dt)
    x0 = _utility.estimate_initial_condition(x, rec_x)
    rec_x = rec_x + x0
    rms_rec_x = __rms_error__(rec_x, x)

    return rms_rec_x, rms_x, rms_dxdt