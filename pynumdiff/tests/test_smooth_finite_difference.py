"""
Unit tests for smoothing + finite difference methods
"""
# pylint: skip-file
import numpy as np
from unittest import TestCase
from pynumdiff.smooth_finite_difference import mediandiff, meandiff, gaussiandiff, \
    friedrichsdiff, butterdiff, splinediff


x = np.array([1., 4., 9., 3., 20.,
              8., 16., 2., 15., 10.,
              15., 3., 5., 7., 4.])
dt = 0.01


class TestSFD(TestCase):
    def test_mediandiff(self):
        params = [3, 2]
        x_hat, dxdt_hat = mediandiff(x, dt, params, options={'iterate': True})
        x_smooth = np.array([1., 4., 4., 8., 9., 8., 15., 10., 15., 10., 10., 5., 5., 5., 4.])
        dxdt = np.array([300., 150., 200., 250., 0., 300., 100., 0.,
                         0., -250., -250., -250., 0., -50., -100.])
        np.testing.assert_array_equal(x_smooth, x_hat)
        np.testing.assert_array_equal(dxdt, dxdt_hat)

    def test_meandiff(self):
        params = [3, 2]
        x_hat, dxdt_hat = meandiff(x, dt, params, options={'iterate': True})
        x_smooth = np.array([2.889, 4., 6.889, 8.778, 11.889, 11.222, 11.444, 9.556,
                             11.111, 10.556, 10.111, 7.333,  6., 5.111, 5.111])
        dxdt = np.array([111.111, 200., 238.889, 250., 122.222, -22.222,
                         -83.333, -16.667, 50., -50., -161.111, -205.556,
                         -111.111, -44.444, 0.])
        np.testing.assert_almost_equal(x_smooth, x_hat, decimal=3)
        np.testing.assert_almost_equal(dxdt, dxdt_hat, decimal=3)

    def test_gaussiandiff(self):
        params = [5]
        x_hat, dxdt_hat = gaussiandiff(x, dt, params, options={'iterate': False})
        x_smooth = np.array([1.805, 4.377, 6.66, 8.066, 13.508, 12.177, 11.278,  8.044,
                             11.116, 11.955, 11.178, 6.187, 5.127, 5.819, 4.706])
        dxdt = np.array([257.235, 242.77, 184.438, 342.42, 205.553, -111.535,
                         -206.61, -8.093, 195.509, 3.089, -288.392, -302.545,
                         -18.409, -21.032, -111.263])
        np.testing.assert_almost_equal(x_smooth, x_hat, decimal=3)
        np.testing.assert_almost_equal(dxdt, dxdt_hat, decimal=3)

    def test_friedrichsdiff(self):
        params = [5]
        x_hat, dxdt_hat = friedrichsdiff(x, dt, params, options={'iterate': False})
        x_smooth = np.array([1.884, 4.589, 5.759, 9.776, 11.456, 13.892, 9.519, 9.954,
                             9.697, 12.946, 9.992, 7.124,  5., 5.527, 4.884])
        dxdt = np.array([270.539, 193.776, 259.335, 284.855, 205.809, -96.888,
                         -196.888, 8.921, 149.586, 14.73, -291.079, -249.586,
                         -79.875, -5.809, -64.316])
        np.testing.assert_almost_equal(x_smooth, x_hat, decimal=3)
        np.testing.assert_almost_equal(dxdt, dxdt_hat, decimal=3)

    def test_butterdiff(self):
        params = [3, 0.074]
        x_hat, dxdt_hat = butterdiff(x, dt, params, options={'iterate': False})
        x_smooth = np.array([3.445, 4.753, 5.997, 7.131, 8.114, 8.914, 9.51, 9.891,
                             10.058, 10.02, 9.798, 9.42, 8.919, 8.332, 7.699])
        dxdt = np.array([130.832, 127.617, 118.881, 105.827, 89.169, 69.833, 48.871,
                         27.381, 6.431, -12.992, -30.023, -43.972, -54.368, -60.98,
                         -63.326])
        np.testing.assert_almost_equal(x_smooth, x_hat, decimal=3)
        np.testing.assert_almost_equal(dxdt, dxdt_hat, decimal=3)

    def test_splinediff(self):
        params = [5, 2]
        x_hat, dxdt_hat = splinediff(x, dt, params, options={'iterate': False})
        x_smooth = np.array([0.995, 4.035, 8.874, 3.279, 19.555, 8.564, 15.386, 2.603,
                             14.455, 10.45, 14.674, 3.193, 4.916, 7.023, 3.997])
        dxdt = np.array([303.996, 393.932, -37.815, 534.063, 264.225, -208.442,
                         -298.051, -46.561, 392.365, 10.93, -362.858, -487.87,
                         191.508, -45.968, -302.579])
        np.testing.assert_almost_equal(x_smooth, x_hat, decimal=3)
        np.testing.assert_almost_equal(dxdt, dxdt_hat, decimal=3)


