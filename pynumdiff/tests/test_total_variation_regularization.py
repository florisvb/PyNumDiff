"""
Unit tests for total variation regularization
"""
# pylint: skip-file

import numpy as np
from unittest import TestCase
from pynumdiff.total_variation_regularization import velocity, iterative_velocity, \
    acceleration, smooth_acceleration, jerk, jerk_sliding

x = np.array([1., 4., 9., 3., 20.,
              8., 16., 2., 15., 10.,
              15., 3., 5., 7., 4.])
dt = 0.01


class TestTVR(TestCase):
    def test_velocity(self):
        params = [0.5]
        x_hat, dxdt_hat = velocity(x, dt, params)
        x_smooth = np.array([1.602,  3.843,  6.083,  8.323, 14.766, 12.766, 10.766,  7.702,
                             11.433, 11.433, 11.433,  6.423,  5.783,  5.143,  4.502])
        dxdt = np.array([2.240400e+02,  2.240400e+02,  2.240399e+02,  4.341530e+02,
                         2.221282e+02, -2.000098e+02, -2.531812e+02,  3.333867e+01,
                         1.865157e+02,  1.453819e-03, -2.504603e+02, -2.824824e+02,
                         -6.404261e+01, -6.404261e+01, -6.404260e+01])
        np.testing.assert_almost_equal(x_smooth, x_hat, decimal=3)
        np.testing.assert_almost_equal(dxdt, dxdt_hat, decimal=3)

    def test_iterative_velocity(self):
        params = [1, 0.05]
        x_hat, dxdt_hat = iterative_velocity(x, dt, params)
        x_smooth = np.array([1.256,  3.254,  5.249,  7.197,  8.96, 10.287, 11.08, 11.407,
                             11.305, 10.875, 10.235,  9.371,  8.305,  7.174,  6.042])
        dxdt = np.array([199.802,  199.742,  199.222,  190.43 ,  162.105,  103.282,
                         55.311,   10.12 ,  -30.571,  -55.409,  -72.603, -100.119,
                         -113.097, -113.097, -113.464])
        np.testing.assert_almost_equal(x_smooth, x_hat, decimal=3)
        np.testing.assert_almost_equal(dxdt, dxdt_hat, decimal=3)

    def test_acceleration(self):
        params = [1]
        x_hat, dxdt_hat = acceleration(x, dt, params)
        x_smooth = np.array([0.877102,  4.444465,  7.311582,  9.478455, 10.945084, 11.711467,
                             11.781232, 11.556045, 11.035907, 10.220818,  9.110778,  7.705786,
                             6.414181,  5.235964,  4.171134])
        dxdt = np.array([391.748501,  321.724019,  251.699538,  181.675056,  111.650573,
                         41.807405,   -7.771095,  -37.266227,  -66.761349,  -96.256475,
                         -125.751604, -134.829808, -123.491084, -112.152359, -100.813634])
        np.testing.assert_almost_equal(x_smooth, x_hat, decimal=3)
        np.testing.assert_almost_equal(dxdt, dxdt_hat, decimal=3)

    def test_smooth_acceleration(self):
        params = [5, 30]
        x_hat, dxdt_hat = smooth_acceleration(x, dt, params)
        x_smooth = np.array([4.166173,  5.530249,  6.771252,  7.873337,  8.795298,  9.504775,
                             9.97939, 10.207285, 10.187111,  9.927596,  9.446796,  8.771074,
                             7.93386,  6.974318,  5.935967])
        dxdt = np.array([136.40761,  129.914652,  118.285976,  102.131007,   82.261336,
                         59.633925,   35.289026,   10.290105,  -14.324909,  -37.578227,
                         -58.581681,  -76.562799,  -90.879981, -101.028383, -106.64176])
        np.testing.assert_almost_equal(x_smooth, x_hat, decimal=3)
        np.testing.assert_almost_equal(dxdt, dxdt_hat, decimal=3)

    def test_jerk(self):
        params = [10]
        x_hat, dxdt_hat = jerk(x, dt, params)
        x_smooth = np.array([0.710131,  4.514052,  7.426194,  9.53278 , 10.920035, 11.674183,
                             11.881448, 11.628054, 11.000226, 10.084188,  8.966164,  7.732378,
                             6.469055,  5.262418,  4.198693])
        dxdt = np.array([420.669933,  335.803167,  250.9364,  174.692056,  107.070136,
                         48.070639,   -2.306435,  -44.061086,  -77.193313, -101.703117,
                         -117.590498, -124.855455, -123.497989, -113.518099, -103.538209])
        np.testing.assert_almost_equal(x_smooth, x_hat, decimal=3)
        np.testing.assert_almost_equal(dxdt, dxdt_hat, decimal=3)

    def test_jerk_sliding(self):
        params = [10]
        x_hat, dxdt_hat = jerk_sliding(x, dt, params)
        x_smooth = np.array([0.710131, 4.514052, 7.426194, 9.53278, 10.920035, 11.674183,
                             11.881448, 11.628054, 11.000226, 10.084188, 8.966164, 7.732378,
                             6.469055, 5.262418, 4.198693])
        dxdt = np.array([420.669933, 335.803167, 250.9364, 174.692056, 107.070136,
                         48.070639, -2.306435, -44.061086, -77.193313, -101.703117,
                         -117.590498, -124.855455, -123.497989, -113.518099, -103.538209])
        np.testing.assert_almost_equal(x_smooth, x_hat, decimal=3)
        np.testing.assert_almost_equal(dxdt, dxdt_hat, decimal=3)

