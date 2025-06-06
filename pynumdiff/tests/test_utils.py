"""
Unit tests for utility functions
"""
# pylint: skip-file

import numpy as np
from pynumdiff.utils import utility, simulate, evaluate


def test_integrate_dxdt_hat():
    dt = 0.1
    for dxdt,expected in [(np.ones(10), np.arange(0, 1, 0.1)), # constant derivative
            (np.linspace(0, 1, 11), [0, 0.005, 0.02, 0.045, 0.08, 0.125, 0.18, 0.245, 0.32, 0.405, 0.5]), # linear derivative
            (np.array([1.0]), [0])]: # edge case of just one entry
        x_hat = utility.integrate_dxdt_hat(dxdt, dt)
        np.testing.assert_allclose(x_hat, expected)
        assert len(x_hat) == len(dxdt)

def test_estimate_initial_condition():
    for x,x_hat,c in [([1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 1.0, 2.0, 3.0, 4.0], 1), # Perfect alignment case, xhat shifted by 1
        (np.ones(5)*10, np.ones(5)*5, 5),
        ([0], [1], -1)]:
        x0 = utility.estimate_initial_condition(x, x_hat)
        np.testing.assert_allclose(x0, float(c), rtol=1e-3)

    np.random.seed(42) # Noisy case. Seed for reproducibility
    x0 = utility.estimate_initial_condition([1.0, 2.0, 3.0, 4.0, 5.0],
        np.array([0.0, 1.0, 2.0, 3.0, 4.0]) + np.random.normal(0, 0.1, 5))
    assert 0.9 < x0 < 1.1 # The result should be close to 1.0, but not exactly due to noise


def test_simulate():
    return

def test_evaluate():
    return
