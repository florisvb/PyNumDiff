"""
Unit tests for kalman smoothing methods
"""
# pylint: skip-file

import numpy as np
from unittest import TestCase
from pynumdiff.kalman_smooth import constant_velocity, constant_acceleration, \
    constant_jerk, known_dynamics


x = np.array([1., 4., 9., 3., 20.,
              8., 16., 2., 15., 10.,
              15., 3., 5., 7., 4.])
dt = 0.01


class TestKS(TestCase):
    def test_constant_velocity(self):
        params = [1e-4, 1e-5]
        x_hat, dxdt_hat = constant_velocity(x, dt, params)
        x_smooth = np.array([7.952849, 7.714494, 7.769948, 7.81768, 8.330625, 8.332996,
                             8.46594, 8.243244, 8.458473, 8.367324, 8.284892, 7.947729,
                             7.998362, 8.123646, 8.303191])
        dxdt = np.array([88.750804,  93.567378, 102.828004,  62.994815,  92.01605,
                         60.395089,  47.494064,  27.626483,  21.537133,  14.105156,
                         8.138253,   7.996629,   4.016616,  -0.1122,   2.319358])
        np.testing.assert_almost_equal(x_smooth, x_hat, decimal=3)
        np.testing.assert_almost_equal(dxdt, dxdt_hat, decimal=3)

    def test_constant_acceleration(self):
        params = [1e-4, 1e-2]
        x_hat, dxdt_hat = constant_acceleration(x, dt, params)
        x_smooth = np.array([5.069524,  6.137091,  7.191819,  8.062104,  9.112695,  9.616349,
                             10.029422,  9.945811, 10.05048,  9.703503,  9.180588,  8.309991,
                             7.546839,  6.581819,  5.421122])
        dxdt = np.array([170.225553,  164.483647,  156.524187,  103.452558,  113.776639,
                         64.258467,   33.813842,   -1.889904,  -25.372839,  -48.272303,
                         -69.60202,  -81.885049, -101.379641, -122.551681, -140.214842])
        np.testing.assert_almost_equal(x_smooth, x_hat, decimal=3)
        np.testing.assert_almost_equal(dxdt, dxdt_hat, decimal=3)

    def test_constant_jerk(self):
        params = [1e-4, 1e-4]
        x_hat, dxdt_hat = constant_jerk(x, dt, params)
        x_smooth = np.array([5.066536,  6.135826,  7.191131,  8.061294,  9.110784,  9.613802,
                             10.026445,  9.943029, 10.047933,  9.701807,  9.179971,  8.310492,
                             7.547672,  6.582594,  5.421728])
        dxdt = np.array([170.262874,  164.484367,  156.478206,  103.371112,  113.682324,
                         64.169044,   33.742701,   -1.935552,  -25.398252,  -48.273806,
                         -69.59001,  -81.873115, -101.384521, -122.579907, -140.269899])
        np.testing.assert_almost_equal(x_smooth, x_hat, decimal=3)
        np.testing.assert_almost_equal(dxdt, dxdt_hat, decimal=3)

    def test_known_dynamics(self):
        return


