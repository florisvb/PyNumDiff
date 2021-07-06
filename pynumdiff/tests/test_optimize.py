"""
Unit tests for optimization module
"""
# pylint: skip-file
import numpy as np
from unittest import TestCase

from pynumdiff.optimize.finite_difference import first_order
from pynumdiff.optimize.smooth_finite_difference import mediandiff, meandiff, gaussiandiff, \
    friedrichsdiff, butterdiff, splinediff
from pynumdiff.optimize.total_variation_regularization import velocity, iterative_velocity, \
    acceleration, smooth_acceleration, jerk, jerk_sliding
from pynumdiff.optimize.linear_model import savgoldiff, spectraldiff, \
    polydiff, chebydiff, lineardiff
from pynumdiff.optimize.kalman_smooth import constant_velocity, constant_acceleration, \
    constant_jerk
from pynumdiff.utils import simulate


# simulation
noise_type = 'normal'
noise_parameters = [0, 0.01]
dt = 0.1
simdt = 0.01
timeseries_length = 2
problem = 'pi_control'
x, x_truth, dxdt_truth, extras = simulate.__dict__[problem](timeseries_length,
                                                            noise_parameters=noise_parameters,
                                                            dt=dt, simdt=0.01)
cutoff_frequency = 0.1
log_gamma = -1.6 * np.log(cutoff_frequency) - 0.71 * np.log(dt) - 5.1
tvgamma = np.exp(log_gamma)


class TestOPT(TestCase):
    # def test_first_order(self):
    #     params_1, val_1 = first_order(x, dt, params=None, options={'iterate': True},
    #                                   tvgamma=tvgamma, dxdt_truth=dxdt_truth)
    #     params_2, val_2 = first_order(x, dt, params=None, options={'iterate': True},
    #                                   tvgamma=0, dxdt_truth=None)
    #     self.assertListEqual(params_1, [12])
    #     self.assertListEqual(params_2, [1])
    #
    # def test_mediandiff(self):
    #     params_1, val_1 = mediandiff(x, dt, params=None, options={'iterate': True},
    #                                  tvgamma=tvgamma, dxdt_truth=dxdt_truth)
    #     params_2, val_2 = mediandiff(x, dt, params=None, options={'iterate': True},
    #                                  tvgamma=0, dxdt_truth=None)
    #     self.assertListEqual(params_1, [5, 5])
    #     self.assertListEqual(params_2, [5, 5])
    #
    # def test_meandiff(self):
    #     params_1, val_1 = meandiff(x, dt, params=None, options={'iterate': True},
    #                                tvgamma=tvgamma, dxdt_truth=dxdt_truth)
    #     params_2, val_2 = meandiff(x, dt, params=None, options={'iterate': True},
    #                                tvgamma=0, dxdt_truth=None)
    #     self.assertListEqual(params_1, [5, 1])
    #     self.assertListEqual(params_2, [5, 1])
    #
    # def test_gaussiandiff(self):
    #     params_1, val_1 = gaussiandiff(x, dt, params=None, options={'iterate': True},
    #                                    tvgamma=tvgamma, dxdt_truth=dxdt_truth)
    #     params_2, val_2 = gaussiandiff(x, dt, params=None, options={'iterate': True},
    #                                    tvgamma=0, dxdt_truth=None)
    #     self.assertListEqual(params_1, [5, 5])
    #     self.assertListEqual(params_2, [5, 1])
    #
    # def test_friedrichsdiff(self):
    #     params_1, val_1 = friedrichsdiff(x, dt, params=None, options={'iterate': True},
    #                                      tvgamma=tvgamma, dxdt_truth=dxdt_truth)
    #     params_2, val_2 = friedrichsdiff(x, dt, params=None, options={'iterate': True},
    #                                      tvgamma=0, dxdt_truth=None)
    #     self.assertListEqual(params_1, [5, 4])
    #     self.assertListEqual(params_2, [5, 1])
    #
    # def test_butterdiff(self):
    #     return
    #
    # def test_splinediff(self):
    #     params_1, val_1 = splinediff(x, dt, params=None, options={'iterate': True},
    #                                  tvgamma=tvgamma, dxdt_truth=dxdt_truth)
    #     params_2, val_2 = splinediff(x, dt, params=None, options={'iterate': True},
    #                                  tvgamma=0, dxdt_truth=None)
    #     self.assertListEqual(params_1, [3, 0.5, 1])
    #     self.assertListEqual(params_2, [5, 0.5, 1])
    #
    # def test_iterative_velocity(self):
    #     params_1, val_1 = iterative_velocity(x, dt, params=None, tvgamma=tvgamma, dxdt_truth=dxdt_truth)
    #     params_2, val_2 = iterative_velocity(x, dt, params=None, tvgamma=0, dxdt_truth=None)
    #     np.testing.assert_almost_equal(np.array(params_1), np.array([1, 0.00035]), decimal=3)
    #     self.assertListEqual(params_2, [1, 0.0001])
    #
    # def test_velocity(self):
    #     return
    #
    # def test_velocity(self):
    #     return
    #
    # def test_acceleration(self):
    #     return
    #
    # def test_smooth_acceleration(self):
    #     return
    #
    # def test_savgoldiff(self):
    #     return
    #
    # def test_spectraldiff(self):
    #     params_1, val_1 = spectraldiff(x, dt, params=None, tvgamma=tvgamma, dxdt_truth=dxdt_truth)
    #     params_2, val_2 = spectraldiff(x, dt, params=None, tvgamma=0, dxdt_truth=None)
    #     np.testing.assert_almost_equal(np.array(params_1), np.array([0.0875]), decimal=3)
    #     np.testing.assert_almost_equal(np.array(params_2), np.array([0.4475]), decimal=3)
    #
    # def test_polydiff(self):
    #     params_1, val_1 = polydiff(x, dt, params=None, tvgamma=tvgamma, dxdt_truth=dxdt_truth)
    #     params_2, val_2 = polydiff(x, dt, params=None, tvgamma=0, dxdt_truth=None)
    #     self.assertListEqual(params_1, [3, 47])
    #     self.assertListEqual(params_2, [8, 10])

    def test_chebydiff(self):
        params_1, val_1 = chebydiff(x, dt, params=None, tvgamma=tvgamma, dxdt_truth=dxdt_truth)
        params_2, val_2 = chebydiff(x, dt, params=None, tvgamma=0, dxdt_truth=None)
        self.assertListEqual(params_1, [3, 10])
        self.assertListEqual(params_2, [8, 18])

    def test_template(self):
        params_1, val_1 = mediandiff(x, dt, params=None, tvgamma=tvgamma, options={'iterate': True}, dxdt_truth=dxdt_truth)
        params_2, val_2 = mediandiff(x, dt, params=None, tvgamma=0, options={'iterate': True}, dxdt_truth=None)
        self.assertListEqual(params_1, [5, 5])
        self.assertListEqual(params_2, [5, 5])
