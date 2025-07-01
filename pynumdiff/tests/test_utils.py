"""
Unit tests for utility functions
"""
# pylint: skip-file

import numpy as np
from pynumdiff.utils import utility, simulate, evaluate
np.random.seed(42) # The answer to life, the universe, and everything


def test_integrate_dxdt_hat():
    """For known simple functions, make sure integration is as expected"""
    dt = 0.1
    for dxdt,expected in [(np.ones(10), np.arange(0, 1, dt)), # constant derivative
            (np.linspace(0, 1, 11), [0, 0.005, 0.02, 0.045, 0.08, 0.125, 0.18, 0.245, 0.32, 0.405, 0.5]), # linear derivative
            (np.array([1.0]), [0])]: # edge case of just one entry
        x_hat = utility.integrate_dxdt_hat(dxdt, dt)
        assert np.allclose(x_hat, expected)
        assert len(x_hat) == len(dxdt)


def test_estimate_initial_condition():
    """For known simple functions, make sure the initial condition is as expected"""
    for x,x_hat,c in [([1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 1.0, 2.0, 3.0, 4.0], 1), # Perfect alignment case, xhat shifted by 1
        (np.ones(5)*10, np.ones(5)*5, 5),
        ([0], [1], -1)]:
        x0 = utility.estimate_initial_condition(x, x_hat)
        assert np.allclose(x0, float(c), rtol=1e-3)

    np.random.seed(42) # Noisy case. Seed for reproducibility
    x0 = utility.estimate_initial_condition([1.0, 2.0, 3.0, 4.0, 5.0],
        np.array([0.0, 1.0, 2.0, 3.0, 4.0]) + np.random.normal(0, 0.1, 5))
    assert 0.9 < x0 < 1.1 # The result should be close to 1.0, but not exactly due to noise


def test_hankel_matrix():
    """Ensure Hankel matrix comes back as defined"""
    assert np.allclose(utility.hankel_matrix([1, 2, 3, 4, 5], 3), [[1, 2, 3],[2, 3, 4],[3, 4, 5]])


def test_peakdet(request):
    """Verify peakdet finds peaks and valleys"""
    t = np.arange(0, 10, 0.001)
    x = 0.3*np.sin(t) + np.sin(1.3*t) + 0.9*np.sin(4.2*t) + 0.02*np.random.randn(10000)
    maxtab, mintab = utility.peakdet(x, 0.5, t)

    if request.config.getoption("--plot"):
        from matplotlib import pyplot
        pyplot.plot(t, x)
        pyplot.plot(mintab[:,0], mintab[:,1], 'g*')
        pyplot.plot(maxtab[:,0], maxtab[:,1], 'r*')
        pyplot.title('peakdet validataion')
        pyplot.show()

    assert np.allclose(maxtab, [[0.473, 1.59843494], # these numbers validated by eye with --plot
                                [1.795, 1.90920786],
                                [3.314, -0.04585991],
                                [4.992, 0.74798665],
                                [6.345, 1.89597554],
                                [7.778, 0.57190318],
                                [9.424, 0.58764606]])
    assert np.allclose(mintab, [[1.096, 0.30361178],
                                [2.739, -1.12624328],
                                [4.072, -2.00254655],
                                [5.582, -0.31529832],
                                [7.135, -0.58327787],
                                [8.603, -1.71278265]])

def test_slide_function():
    """Verify the slide function's weighting scheme calculates as expected"""
    def identity(x, dt): return x, 0 # should come back the same

    x = np.arange(100)
    kernel = utility.gaussian_kernel(9)

    x_hat, dxdt_hat = utility.slide_function(identity, x, 0.1, kernel, 2)

    assert np.allclose(x, x_hat)


def test_simulate():
    return

def test_evaluate():
    return
