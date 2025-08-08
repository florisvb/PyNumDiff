import numpy as np
from pytest import skip

from ..finite_difference import first_order as iterated_finite_difference # actually second order
from ..smooth_finite_difference import mediandiff, meandiff, gaussiandiff, friedrichsdiff, butterdiff, splinediff
from ..linear_model import spectraldiff, polydiff, savgoldiff
from ..total_variation_regularization import velocity, acceleration, iterative_velocity
from ..optimize import optimize
from ..utils.simulate import pi_cruise_control


# simulation
dt = 0.01
x, x_truth, dxdt_truth = pi_cruise_control(duration=2, noise_type='normal', noise_parameters=[0, 0.01], dt=dt)
cutoff_frequency = 10 # in Hz
log_gamma = -1.6 * np.log(cutoff_frequency) - 0.71 * np.log(dt) - 5.1
tvgamma = np.exp(log_gamma)


def test_finite_difference():
    params1, val1 = optimize(iterated_finite_difference, x, dt, dxdt_truth=dxdt_truth, padding='auto')
    params2, val2 = optimize(iterated_finite_difference, x, dt, tvgamma=tvgamma, dxdt_truth=None, padding='auto')
    assert params1['num_iterations'] == 5
    assert params2['num_iterations'] == 1

def test_mediandiff():
    params1, val1 = optimize(mediandiff, x, dt, search_space={'num_iterations':1}, dxdt_truth=dxdt_truth, padding='auto')
    params2, val2 = optimize(mediandiff, x, dt, search_space={'num_iterations':1}, tvgamma=tvgamma, dxdt_truth=None, padding='auto')
    assert params1['window_size'] == 5
    assert params2['window_size'] == 1

def test_meandiff():
    params1, val1 = optimize(meandiff, x, dt, search_space={'num_iterations':1}, dxdt_truth=dxdt_truth, padding='auto')
    params2, val2 = optimize(meandiff, x, dt, search_space={'num_iterations':1}, tvgamma=tvgamma, dxdt_truth=None, padding='auto')
    assert params1['window_size'] == 5
    assert params2['window_size'] == 1

def test_gaussiandiff():
    params1, val1 = optimize(gaussiandiff, x, dt, search_space={'num_iterations':1}, dxdt_truth=dxdt_truth, padding='auto')
    params2, val2 = optimize(gaussiandiff, x, dt, search_space={'num_iterations':1}, tvgamma=tvgamma, dxdt_truth=None, padding='auto')
    assert params1['window_size'] == 9
    assert params2['window_size'] == 1

def test_friedrichsdiff():
    params1, val1 = optimize(friedrichsdiff, x, dt, search_space={'num_iterations':1}, dxdt_truth=dxdt_truth, padding='auto')
    params2, val2 = optimize(friedrichsdiff, x, dt, search_space={'num_iterations':1}, tvgamma=tvgamma, dxdt_truth=None, padding='auto')
    assert params1['window_size'] == 9
    assert params2['window_size'] == 1

def test_iterative_velocity():
    params1, val1 = optimize(iterative_velocity, x, dt, search_space={'num_iterations':1}, dxdt_truth=dxdt_truth, padding='auto')
    params2, val2 = optimize(iterative_velocity, x, dt, search_space={'num_iterations':1}, tvgamma=tvgamma, dxdt_truth=None, padding='auto')
    
    np.testing.assert_almost_equal(params1['gamma'], 0.0001, decimal=4)
    np.testing.assert_almost_equal(params2['gamma'], 0.0001, decimal=4)

def test_velocity():
    try: import cvxpy
    except: skip("could not import cvxpy, skipping test_velocity")

    params1, val1 = optimize(velocity, x, dt, dxdt_truth=dxdt_truth, padding='auto', maxiter=20)
    params2, val2 = optimize(velocity, x, dt, tvgamma=tvgamma, dxdt_truth=None, padding='auto', maxiter=20)

    np.testing.assert_almost_equal(params1['gamma'], 0.0769, decimal=3)
    np.testing.assert_almost_equal(params2['gamma'], 0.010, decimal=3)

def test_acceleration():
    try: import cvxpy
    except: pytest.skip("could not import cvxpy, skipping test_acceleration")

    params1, val1 = optimize(acceleration, x, dt, dxdt_truth=dxdt_truth, padding='auto', maxiter=20)
    params2, val2 = optimize(acceleration, x, dt, tvgamma=tvgamma, padding='auto', maxiter=20)

    np.testing.assert_almost_equal(params1['gamma'], 0.147, decimal=3)
    np.testing.assert_almost_equal(params2['gamma'], 0.0046, decimal=4)

def test_savgoldiff():
    params1, val1 = optimize(savgoldiff, x, dt, dxdt_truth=dxdt_truth, padding='auto')
    params2, val2 = optimize(savgoldiff, x, dt, tvgamma=tvgamma, dxdt_truth=None, padding='auto')
    assert (params1['poly_order'], params1['window_size'], params1['smoothing_win']) == (7, 41, 3)
    assert (params2['poly_order'], params2['window_size'], params2['smoothing_win']) == (3, 3, 5)

def test_spectraldiff():
    params1, val1 = optimize(spectraldiff, x, dt, dxdt_truth=dxdt_truth, padding='auto')
    params2, val2 = optimize(spectraldiff, x, dt, tvgamma=tvgamma, padding='auto')
    np.testing.assert_almost_equal(params1['high_freq_cutoff'], 0.105, decimal=2)
    np.testing.assert_almost_equal(params2['high_freq_cutoff'], 0.105, decimal=2)

def test_polydiff():
    params1, val1 = optimize(polydiff, x, dt, search_space={'step_size':1}, dxdt_truth=dxdt_truth, padding='auto')
    params2, val2 = optimize(polydiff, x, dt, search_space={'step_size':1}, tvgamma=tvgamma, dxdt_truth=None, padding='auto')
    assert (params1['poly_order'], params1['window_size']) == (6, 50)
    assert (params2['poly_order'], params2['window_size']) == (4, 10)
