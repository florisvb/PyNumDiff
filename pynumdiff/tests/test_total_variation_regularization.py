"""
Unit tests for total variation regularization
"""
# pylint: skip-file

import numpy as np
from unittest import TestCase
import pytest
from pynumdiff.total_variation_regularization import *

x = np.array([1., 4., 9., 3., 20.,
              8., 16., 2., 15., 10.,
              15., 3., 5., 7., 4.])
dt = 0.01


class TestTVR(TestCase):
    def test_velocity(self):
        try:
            import cvxpy
        except:
            pytest.skip("could not import cvxpy, skipping test_velocity", allow_module_level=True)

        params = [0.5]
        x_hat, dxdt_hat = velocity(x, dt, params, options={'solver': 'CVXOPT'})
        x_smooth = np.array([1.60206974,  3.84254116,  6.08301239,  8.32348272, 14.76608638,
                             12.76589239, 10.76569864,  7.70248886, 11.43239643, 11.4325017,
                             11.43260691,  6.42354819,  5.78305309,  5.14255819,  4.50206322])
        dxdt = np.array([2.24047187e+02,  2.24047133e+02,  2.24047078e+02,  4.34153700e+02,
                         2.22120483e+02, -2.00019387e+02, -2.53170177e+02,  3.33348898e+01,
                         1.86500642e+02,  1.05238579e-02, -2.50447675e+02, -2.82477691e+02,
                         -6.40494998e+01, -6.40494935e+01, -6.40494871e+01])
        np.testing.assert_almost_equal(x_smooth, x_hat, decimal=2)
        np.testing.assert_almost_equal(dxdt, dxdt_hat, decimal=2)

    def test_iterative_velocity(self):
        params = [1, 0.05]
        x_hat, dxdt_hat = iterative_velocity(x, dt, params)
        x_smooth = np.array([1.256,  3.254,  5.249,  7.197,  8.96, 10.287, 11.08, 11.407,
                             11.305, 10.875, 10.235,  9.371,  8.305,  7.174,  6.042])
        dxdt = np.array([199.802,  199.742,  199.222,  190.43,  162.105,  103.282,
                         55.311,   10.12,  -30.571,  -55.409,  -72.603, -100.119,
                         -113.097, -113.097, -113.464])
        np.testing.assert_almost_equal(x_smooth, x_hat, decimal=2)
        np.testing.assert_almost_equal(dxdt, dxdt_hat, decimal=2)

    def test_acceleration(self):
        try:
            import cvxpy
        except:
            pytest.skip("could not import cvxpy, skipping test_acceleration", allow_module_level=True)

        params = [1]
        x_hat, dxdt_hat = acceleration(x, dt, params, options={'solver': 'CVXOPT'})
        x_smooth = np.array([0.87728375,  4.44441238,  7.31141687,  9.47829719, 10.94505335,
                             11.7116852, 11.78131319, 11.5560333, 11.03584752, 10.2207553,
                             9.11075633,  7.7058506,  6.41426253,  5.23599238,  4.17104012])
        dxdt = np.array([391.71907211,  321.70665613,  251.69424015,  181.6818242,
                         111.66940057,   41.81299196,   -7.78259499,  -37.27328368,
                         -66.76389967,  -96.25455924, -125.74523529, -134.82469003,
                         -123.49291116, -112.16112081, -100.82933046])
        np.testing.assert_almost_equal(x_smooth, x_hat, decimal=2)
        np.testing.assert_almost_equal(dxdt, dxdt_hat, decimal=2)

    def test_smooth_acceleration(self):
        try:
            import cvxpy
        except:
            pytest.skip("could not import cvxpy, skipping test_smooth_acceleration", allow_module_level=True)

        params = [5, 30]
        x_hat, dxdt_hat = smooth_acceleration(x, dt, params, options={'solver': 'CVXOPT'})
        x_smooth = np.array([4.16480747,  5.52913444,  6.77037146,  7.87267273,  8.79483088,
                             9.5044844,  9.97926076, 10.20730827, 10.18728338,  9.92792114,
                             9.44728533,  8.77174094,  7.93472066,  6.97538656,  5.93725369])
        dxdt = np.array([136.43269721,  129.9388182,  118.30858578,  102.15166804,
                         82.27996127,   59.65074227,   35.30453082,   10.30497111,
                         -14.30994982,  -37.56249817,  -58.56466324,  -76.54421499,
                         -90.85984169, -101.00697716, -106.61959829])
        np.testing.assert_almost_equal(x_smooth, x_hat, decimal=2)
        np.testing.assert_almost_equal(dxdt, dxdt_hat, decimal=2)

    def test_jerk(self):
        try:
            import cvxpy
        except:
            pytest.skip("could not import cvxpy, skipping test_jerk", allow_module_level=True)

        params = [10]
        x_hat, dxdt_hat = jerk(x, dt, params, options={'solver': 'CVXOPT'})
        x_smooth = np.array([0.71013072,  4.51405229,  7.42619407,  9.53278029, 10.92003519,
                             11.674183, 11.88144796, 11.6280543, 11.00022625, 10.08418804,
                             8.9661639,  7.73237808,  6.4690548,  5.2624183,  4.19869281])
        dxdt = np.array([420.66993476,  335.80316742,  250.93640008,  174.69205619,
                         107.07013572,   48.07063861,   -2.30643522,  -44.06108581,
                         -77.19331317, -101.70311726, -117.59049798, -124.85545525,
                         -123.49798898, -113.51809914, -103.5382093])
        np.testing.assert_almost_equal(x_smooth, x_hat, decimal=2)
        np.testing.assert_almost_equal(dxdt, dxdt_hat, decimal=2)

    def test_jerk_sliding(self):
        try:
            import cvxpy
        except:
            pytest.skip("could not import cvxpy, skipping test_jerk_sliding", allow_module_level=True)

        params = [10]
        x_hat, dxdt_hat = jerk_sliding(x, dt, params, options={'solver': 'CVXOPT'})
        x_smooth = np.array([0.71013072,  4.51405229,  7.42619407,  9.53278029, 10.92003519,
                             11.674183, 11.88144796, 11.6280543, 11.00022625, 10.08418804,
                             8.9661639,  7.73237808,  6.4690548,  5.2624183,  4.19869281])
        dxdt = np.array([420.66993476,  335.80316742,  250.93640008,  174.69205619,
                         107.07013572,   48.07063861,   -2.30643522,  -44.06108581,
                         -77.19331317, -101.70311726, -117.59049798, -124.85545525,
                         -123.49798898, -113.51809914, -103.5382093])
        np.testing.assert_almost_equal(x_smooth, x_hat, decimal=2)
        np.testing.assert_almost_equal(dxdt, dxdt_hat, decimal=2)
