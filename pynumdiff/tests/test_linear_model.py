"""
Unit tests for linear (smoothing) model
"""
# pylint: skip-file

import numpy as np
from unittest import TestCase
from pynumdiff.linear_model import savgoldiff, spectraldiff, \
    polydiff, chebydiff, lineardiff

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
        np.testing.assert_almost_equal(x_smooth, x_hat, decimal=3)
        np.testing.assert_almost_equal(dxdt, dxdt_hat, decimal=3)

    def test_spectraldiff(self):
        params = [0.1]
        x_hat, dxdt_hat = spectraldiff(x, dt, params)
        x_smooth = np.array([3.99, 5.038, 6.635, 8.365, 9.971, 11.201, 11.86, 11.86,
                             11.231, 10.113, 8.722, 7.296, 6.047, 5.116, 4.556])
        dxdt = np.array([104.803, 147., 172.464, 173.547, 147.67, 98.194,
                         33.754, -33.769, -92.105, -131.479, -146.761, -138.333,
                         -111.508, -74.752, -37.276])
        np.testing.assert_almost_equal(x_smooth, x_hat, decimal=3)
        np.testing.assert_almost_equal(dxdt, dxdt_hat, decimal=3)

    def test_polydiff(self):
        params = [2, 3]
        x_hat, dxdt_hat = polydiff(x, dt, params)
        x_smooth = np.array([1.16153, 4.506877, 6.407802, 8.544663, 13.431766, 14.051294,
                             10.115687, 7.674865, 10.471466, 13.612046, 11.363571, 5.68407,
                             4.443968, 6.213507, 4.695931])
        dxdt = np.array([330.730385, 284.267456, 299.891801, 305.441626, 205.475727,
                         -145.229037, -279.41178, 15.428548, 244.252341, 20.343789,
                         -326.727498, -288.988297, 33.647456, 27.861175, -344.695033])
        np.testing.assert_almost_equal(x_smooth, x_hat, decimal=3)
        np.testing.assert_almost_equal(dxdt, dxdt_hat, decimal=3)

    def test_chebydiff(self):
        params = [2, 3]
        x_hat, dxdt_hat = chebydiff(x, dt, params)
        x_smooth = np.array([1., 4.638844, 7.184256, 6.644655, 15.614775, 11.60484,
                             12.284141, 6.082226, 12.000615, 12.058705, 12.462283, 5.018101,
                             4.674378, 6.431201, 4.])
        dxdt = np.array([202.732652, 346.950235, -140.713336, 498.719617, 212.717775,
                         -185.13847, -266.604056, -51.792587, 377.969849,  -0.749768,
                         -297.654931, -455.876155, 197.575692, -24.809441, -150.109487])
        np.testing.assert_almost_equal(x_smooth, x_hat, decimal=3)
        np.testing.assert_almost_equal(dxdt, dxdt_hat, decimal=3)

    def test_lineardiff(self):
        params = [3, 5, 10]
        x_hat, dxdt_hat = lineardiff(x, dt, params)
        x_smooth = np.array([3.070917, 3.435011, 6.36357, 10.276495, 12.033908, 10.594193,
                             9.608234, 9.731333, 10.333141, 10.806873, 9.710675, 7.456206,
                             5.706933, 4.856216, 5.68522])
        dxdt = np.array([36.409323, 164.63261, 342.074243, 283.516901, 15.884867,
                         -121.28366, -43.142966, 36.245352, 53.776983, -31.123294,
                         -167.533366, -200.18714, -129.999465, -1.085623, 82.900375])
        np.testing.assert_almost_equal(x_smooth, x_hat, decimal=3)
        np.testing.assert_almost_equal(dxdt, dxdt_hat, decimal=3)
