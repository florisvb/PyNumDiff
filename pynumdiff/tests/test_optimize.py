"""
Unit tests for optimization module
"""
# pylint: skip-file
import numpy as np
from unittest import TestCase
import pytest 

from pynumdiff.optimize.finite_difference import first_order
from pynumdiff.optimize.smooth_finite_difference import mediandiff, meandiff, gaussiandiff, \
    friedrichsdiff, butterdiff, splinediff
from pynumdiff.optimize.total_variation_regularization import *
from pynumdiff.optimize.linear_model import *
from pynumdiff.optimize.kalman_smooth import constant_velocity, constant_acceleration, \
    constant_jerk
from pynumdiff.utils import simulate



# simulation
noise_type = 'normal'
noise_parameters = [0, 0.01]
dt = 0.01
timeseries_length = 2
problem = 'pi_control'
x, x_truth, dxdt_truth, extras = simulate.__dict__[problem](timeseries_length,
                                                            noise_parameters=noise_parameters,
                                                            dt=dt)
cutoff_frequency = 0.1
log_gamma = -1.6 * np.log(cutoff_frequency) - 0.71 * np.log(dt) - 5.1
tvgamma = np.exp(log_gamma)

def get_err_msg(actual_params, desired_params):
    err_msg = 'Actual params were: ' + ', '.join(map(str, actual_params)) + ' instead of: ' + ', '.join(map(str, desired_params))
    return err_msg

class TestOPT(TestCase):
    def test_first_order(self):
        params_1, val_1 = first_order(x, dt, params=None, options={'iterate': True},
                                      tvgamma=tvgamma, dxdt_truth=dxdt_truth)
        params_2, val_2 = first_order(x, dt, params=None, options={'iterate': True},
                                      tvgamma=0, dxdt_truth=None)
        self.assertListEqual(params_1, [5])
        self.assertListEqual(params_2, [1])
    
    def test_mediandiff(self):
        params_1, val_1 = mediandiff(x, dt, params=None, options={'iterate': False},
                                     tvgamma=tvgamma, dxdt_truth=dxdt_truth)
        params_2, val_2 = mediandiff(x, dt, params=None, options={'iterate': False},
                                     tvgamma=0, dxdt_truth=None)
        self.assertListEqual(params_1, [5])
        self.assertListEqual(params_2, [1])
    
    def test_meandiff(self):
        params_1, val_1 = meandiff(x, dt, params=None, options={'iterate': False},
                                   tvgamma=tvgamma, dxdt_truth=dxdt_truth)
        params_2, val_2 = meandiff(x, dt, params=None, options={'iterate': False},
                                   tvgamma=0, dxdt_truth=None)
        self.assertListEqual(params_1, [5])
        self.assertListEqual(params_2, [1])
    
    def test_gaussiandiff(self):
        params_1, val_1 = gaussiandiff(x, dt, params=None, options={'iterate': False},
                                       tvgamma=tvgamma, dxdt_truth=dxdt_truth)
        params_2, val_2 = gaussiandiff(x, dt, params=None, options={'iterate': False},
                                       tvgamma=0, dxdt_truth=None)
        self.assertListEqual(params_1, [9])
        self.assertListEqual(params_2, [1])
    
    def test_friedrichsdiff(self):
        params_1, val_1 = friedrichsdiff(x, dt, params=None, options={'iterate': False},
                                         tvgamma=tvgamma, dxdt_truth=dxdt_truth)
        params_2, val_2 = friedrichsdiff(x, dt, params=None, options={'iterate': False},
                                         tvgamma=0, dxdt_truth=None)
        self.assertListEqual(params_1, [9])
        self.assertListEqual(params_2, [1])
    
    def test_butterdiff(self):
        params_1, val_1 = butterdiff(x, dt, params=None, tvgamma=tvgamma, dxdt_truth=dxdt_truth)
        params_2, val_2 = butterdiff(x, dt, params=None, tvgamma=0, dxdt_truth=None)

        np.testing.assert_array_less( np.abs(params_1[0] - 9), 1.001, err_msg=get_err_msg(params_1, [9, 0.157]))
        np.testing.assert_array_less( np.abs(params_1[1] - 0.157), 0.01, err_msg=get_err_msg(params_1, [9, 0.157]))
        #np.testing.assert_almost_equal(params_1, [9, 0.157], decimal=3, err_msg=get_err_msg(params_1, [9, 0.157]))
        np.testing.assert_almost_equal(params_2, [2, 0.99], decimal=3, err_msg=get_err_msg(params_2, [2, 0.99]))
    
    def test_splinediff(self):
        params_1, val_1 = splinediff(x, dt, params=None, options={'iterate': True},
                                     tvgamma=tvgamma, dxdt_truth=dxdt_truth)
        params_2, val_2 = splinediff(x, dt, params=None, options={'iterate': True},
                                     tvgamma=0, dxdt_truth=None)
        np.testing.assert_almost_equal(params_1, [5, 0.0147, 1], decimal=2)
        np.testing.assert_almost_equal(params_2, [5, 0.0147, 1], decimal=2)
    
    def test_iterative_velocity(self):
        params_1, val_1 = iterative_velocity(x, dt, params=None, tvgamma=tvgamma, dxdt_truth=dxdt_truth)
        params_2, val_2 = iterative_velocity(x, dt, params=None, tvgamma=0, dxdt_truth=None)
        np.testing.assert_array_less( np.abs(params_1[0] - 2), 1.001)
        np.testing.assert_array_less( np.abs(params_2[0] - 2), 1.001)
        
        np.testing.assert_almost_equal(params_1[1], 0.0001, decimal=4)
        np.testing.assert_almost_equal(params_2[1], 0.0001, decimal=4)
        
        #self.assertListEqual(params_1, [2, 0.0001])
        #self.assertListEqual(params_2, [2, 0.0001])
    
    def test_velocity(self):
        try:
            import cvxpy
        except:
            pytest.skip("could not import cvxpy, skipping test_velocity", allow_module_level=True)

        params_1, val_1 = velocity(x, dt, params=None, tvgamma=tvgamma, dxdt_truth=dxdt_truth)
        params_2, val_2 = velocity(x, dt, params=None, tvgamma=0, dxdt_truth=None)
        param_1_error = np.abs(params_1[0] - 0.07218)
        param_2_error = np.abs(params_2[0] - 0.0001)

        np.testing.assert_array_less(param_1_error, 2)
        np.testing.assert_array_less(param_2_error, 2)

        return
    
    def test_acceleration(self):
        try:
            import cvxpy
        except:
            pytest.skip("could not import cvxpy, skipping test_acceleration", allow_module_level=True)

        params_1, val_1 = acceleration(x, dt, params=None, tvgamma=tvgamma, dxdt_truth=dxdt_truth)
        params_2, val_2 = acceleration(x, dt, params=None, tvgamma=0, dxdt_truth=None)
        param_1_error = np.abs(params_1[0] - 0.1447)
        param_2_error = np.abs(params_2[0] - 0.0001)

        np.testing.assert_array_less(param_1_error, 2)
        np.testing.assert_array_less(param_2_error, 2)

        return
    
    def test_savgoldiff(self):
        params_1, val_1 = savgoldiff(x, dt, params=None, tvgamma=tvgamma, dxdt_truth=dxdt_truth)
        params_2, val_2 = savgoldiff(x, dt, params=None, tvgamma=0, dxdt_truth=None)
        self.assertListEqual(params_1, [10, 59, 3])
        self.assertListEqual(params_2, [9, 3, 3])

    def test_spectraldiff(self):
        params_1, val_1 = spectraldiff(x, dt, params=None, tvgamma=tvgamma, dxdt_truth=dxdt_truth)
        params_2, val_2 = spectraldiff(x, dt, params=None, tvgamma=0, dxdt_truth=None)
        np.testing.assert_almost_equal(params_1, [0.0912], decimal=3)
        np.testing.assert_almost_equal(params_2, [0.575], decimal=3)
    
    def test_polydiff(self):
        params_1, val_1 = polydiff(x, dt, params=None, tvgamma=tvgamma, dxdt_truth=dxdt_truth)
        params_2, val_2 = polydiff(x, dt, params=None, tvgamma=0, dxdt_truth=None)
        self.assertListEqual(params_1, [6, 50])
        self.assertListEqual(params_2, [4, 10])
    
    # def test_chebydiff(self):
    #     try:
    #         import pychebfun
    #     except:
    #         pytest.skip("could not import pychebfun, skipping test_chebydiff", allow_module_level=True)

    #     params_1, val_1 = chebydiff(x, dt, params=None, tvgamma=tvgamma, dxdt_truth=dxdt_truth)
    #     params_2, val_2 = chebydiff(x, dt, params=None, tvgamma=0, dxdt_truth=None)
    #     self.assertListEqual(params_1, [9, 108])
    #     self.assertListEqual(params_2, [9, 94])
