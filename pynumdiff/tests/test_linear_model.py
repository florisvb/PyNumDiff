"""
Unit tests for linear (smoothing) model
"""
# pylint: skip-file

import numpy as np
from unittest import TestCase
import logging as _logging

_logging.basicConfig(
    level=_logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        _logging.FileHandler("debug.log"),
        _logging.StreamHandler()
    ]
)


from pynumdiff.linear_model import *


x = np.array([1., 4., 9., 3., 20.,
              8., 16., 2., 15., 10.,
              15., 3., 5., 7., 4.])
dt = 0.01


class TestLM(TestCase):
    def test_savgoldiff(self):
        params = [2, 4, 4]
        x_hat, dxdt_hat = savgoldiff(x, dt, params)
        x_smooth = np.array([4.669816,  4.374363,  6.46848,  8.899164, 10.606681, 11.059424,
                             10.519283, 10.058375, 10.191014, 10.193343,  9.208019,  7.445465,
                             5.880869,  5.49672,  6.930156])
        dxdt = np.array([-29.5453,  156.853147,  261.970245,  224.16657,  117.336993,
                         -26.788542,  -81.239512,  -10.942197,   37.470096,  -37.004311,
                         -160.060586, -192.450136, -120.46908,   43.639278,  243.047964])
        np.testing.assert_almost_equal(x_smooth, x_hat, decimal=2)
        np.testing.assert_almost_equal(dxdt, dxdt_hat, decimal=2)

    def test_spectraldiff(self):
        params = [0.1]
        x_hat, dxdt_hat = spectraldiff(x, dt, params)
        x_smooth = np.array([3.99, 5.038, 6.635, 8.365, 9.971, 11.201, 11.86, 11.86,
                             11.231, 10.113, 8.722, 7.296, 6.047, 5.116, 4.556])
        dxdt = np.array([104.803, 147., 172.464, 173.547, 147.67, 98.194,
                         33.754, -33.769, -92.105, -131.479, -146.761, -138.333,
                         -111.508, -74.752, -37.276])
        np.testing.assert_almost_equal(x_smooth, x_hat, decimal=2)
        np.testing.assert_almost_equal(dxdt, dxdt_hat, decimal=2)

    def test_polydiff(self):
        params = [2, 3]
        x_hat, dxdt_hat = polydiff(x, dt, params)
        x_smooth = np.array([1.16153, 4.506877, 6.407802, 8.544663, 13.431766, 14.051294,
                             10.115687, 7.674865, 10.471466, 13.612046, 11.363571, 5.68407,
                             4.443968, 6.213507, 4.695931])
        dxdt = np.array([330.730385, 284.267456, 299.891801, 305.441626, 205.475727,
                         -145.229037, -279.41178, 15.428548, 244.252341, 20.343789,
                         -326.727498, -288.988297, 33.647456, 27.861175, -344.695033])
        np.testing.assert_almost_equal(x_smooth, x_hat, decimal=2)
        np.testing.assert_almost_equal(dxdt, dxdt_hat, decimal=2)

    # def test_chebydiff(self):
    #     try:
    #         import pychebfun
    #     except:
    #         __warning__ = '\nCannot import pychebfun, skipping chebydiff test.'
    #         _logging.info(__warning__)
    #         return

    #     params = [2, 3]
    #     x_hat, dxdt_hat = chebydiff(x, dt, params)
    #     x_smooth = np.array([1., 4.638844, 7.184256, 6.644655, 15.614775, 11.60484,
    #                          12.284141, 6.082226, 12.000615, 12.058705, 12.462283, 5.018101,
    #                          4.674378, 6.431201, 4.])
    #     dxdt = np.array([202.732652, 346.950235, -140.713336, 498.719617, 212.717775,
    #                      -185.13847, -266.604056, -51.792587, 377.969849,  -0.749768,
    #                      -297.654931, -455.876155, 197.575692, -24.809441, -150.109487])
    #     np.testing.assert_almost_equal(x_smooth, x_hat, decimal=2)
    #     np.testing.assert_almost_equal(dxdt, dxdt_hat, decimal=2)

    def test_lineardiff(self):
        try:
            import cvxpy
        except:
            __warning__ = '\nCannot import cvxpy, skipping lineardiff test.'
            _logging.info(__warning__)
            return

        params = [3, 5, 10]
        x_hat, dxdt_hat = lineardiff(x, dt, params, options={'solver': 'CVXOPT'})
        x_smooth = np.array([3.070975,  3.435072,  6.363585, 10.276584, 12.033974, 10.594136,
                             9.608228,  9.731326, 10.333255, 10.806791,  9.710448,  7.456045,
                             5.70695,  4.856271,  5.685251])
        dxdt = np.array([36.409751,  164.630545,  342.075623,  283.519415,   15.877598,
                         -121.287252,  -43.140514,   36.251305,   53.773231,  -31.140351,
                         -167.537258, -200.174883, -129.988725,   -1.084955,   82.897991])
        np.testing.assert_almost_equal(x_smooth, x_hat, decimal=2)
        np.testing.assert_almost_equal(dxdt, dxdt_hat, decimal=2)
