import numpy as np
from pytest import skip

from ..finite_difference import first_order as iterated_finite_difference
from ..smooth_finite_difference import butterdiff
from ..basis_fit import spectraldiff
from ..polynomial_fit import polydiff, savgoldiff, splinediff
from ..total_variation_regularization import velocity, acceleration, iterative_velocity
from ..kalman_smooth import rtsdiff
from ..optimize import optimize
from ..utils.simulate import pi_cruise_control
from ..utils.evaluate import rmse


dt = 0.01
x, x_truth, dxdt_truth = pi_cruise_control(duration=2, noise_type='normal', noise_parameters=[0, 0.01], dt=dt)
cutoff_frequency = 3 # in Hz
tvgamma = np.exp(-1.6 * np.log(cutoff_frequency) - 0.71 * np.log(dt) - 5.1)


def test_parallel_same_as_serial():
    """Ensure running optimize across several processes returns the same result as running in a single process"""
    params_parallel, val_parallel = optimize(rtsdiff, x, dt, tvgamma=tvgamma, parallel=True)
    params_serial, val_serial = optimize(rtsdiff, x, dt, tvgamma=tvgamma, parallel=False)

    assert np.allclose(val_serial, val_parallel)
    assert params_serial == params_parallel


def test_targeting_rmse_vs_tvgamma_loss():
    """Ensure optimization properly targets different metrics"""
    params_rmse, val_rmse = optimize(splinediff, x, dt, dxdt_truth=dxdt_truth)
    params_loss, val_loss = optimize(splinediff, x, dt, tvgamma=tvgamma)
    
    x_hat, dxdt_hat = splinediff(x, dt, **params_loss)
    loss_rmse = rmse(dxdt_truth, dxdt_hat)

    assert val_rmse < loss_rmse < 1.1*val_rmse # This exact bound might break if using a different diff method or data series, but the point is they should be close


def test_search_space_updates_applied():
    """Ensure search space updates are used in optimization"""
    params2, _ = optimize(butterdiff, x, dt, search_space_updates={'filter_order':2}, tvgamma=tvgamma)
    params3, _ = optimize(butterdiff, x, dt, search_space_updates={'filter_order':3}, tvgamma=tvgamma)

    assert params2['filter_order'] == 2
    assert params3['filter_order'] == 3
