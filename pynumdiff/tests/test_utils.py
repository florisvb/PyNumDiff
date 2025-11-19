"""Unit tests for utility functions"""
# pylint: skip-file
import numpy as np
from pynumdiff.utils import utility, evaluate
from pynumdiff.utils.simulate import sine, triangle, pop_dyn, linear_autonomous, pi_cruise_control, lorenz_x
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


def test_estimate_integration_constant():
    """For known simple functions, make sure the initial condition is as expected"""
    for x,x_hat,c in [(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([0.0, 1.0, 2.0, 3.0, 4.0]), 1), # Perfect alignment case, xhat shifted by 1
            (np.ones(5)*10, np.ones(5)*5, 5),
            (np.array([0]), np.array([1]), -1)]:
        x0 = utility.estimate_integration_constant(x, x_hat)
        assert np.allclose(x0, float(c), rtol=1e-3)

    x_hat = np.sin(np.arange(400)*0.01)
    x = x_hat + np.random.normal(0, 0.1, 400) + 1 # shift data by 1
    x0 = utility.estimate_integration_constant(x, x_hat, M=float('inf'))
    assert 0.95 < x0 < 1.05 # The result should be close to 1.0, but not exactly due to noise

    x[100] = 100 # outlier case
    x0 = utility.estimate_integration_constant(x, x_hat, M=0)
    assert 0.95 < x0 < 1.05
    x0 = utility.estimate_integration_constant(x, x_hat, M=6)
    assert 0.95 < x0 < 1.05


def test_convolutional_smoother():
    """Ensure the convolutional smoother isn't introducing edge effects"""
    x = np.ones(10)
    kernel_odd = np.ones(3)/3
    kernel_even = np.ones(4)/4

    assert np.allclose(utility.convolutional_smoother(x, kernel_odd, num_iterations=3), np.ones(len(x)))
    assert np.allclose(utility.convolutional_smoother(x, kernel_even, num_iterations=3), np.ones(len(x)))


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

    assert np.allclose(maxtab, [[0.447, 1.58575613], # these numbers validated by eye with --plot
                                [1.818, 1.91349239],
                                [3.316,-0.02740252],
                                [4.976, 0.74512778],
                                [6.338, 1.89861691],
                                [7.765, 0.57577842],
                                [9.402, 0.59450898]])
    assert np.allclose(mintab, [[1.139, 0.31325728],
                                [2.752,-1.12769567],
                                [4.098,-2.00326946],
                                [5.507,-0.31714122],
                                [7.211,-0.59708324],
                                [8.612,-1.7118216 ]])

def test_slide_function():
    """Verify the slide function's weighting scheme calculates as expected"""
    def identity(x, dt): return x, 0 # should come back the same

    x = np.arange(100)
    kernel = utility.gaussian_kernel(9)

    x_hat, dxdt_hat = utility.slide_function(identity, x, 0.1, kernel, stride=2)

    assert np.allclose(x, x_hat)


def test_simulations(request):
    """Just sprint through running them all to make sure they go. Optionally plot with flag."""
    if request.config.getoption("--plot"):
        from matplotlib import pyplot
        axes = [pyplot.subplots(2, 3, figsize=(18,7), constrained_layout=True)[1] for i in range(3)]

    for j,dt in enumerate([0.005, 0.01, 0.02]):
        for i,(sim,title) in enumerate(zip([pi_cruise_control, sine, triangle, pop_dyn, linear_autonomous, lorenz_x],
            ["Cruise Control", "Sum of Sines", "Triangles", "Logistic Growth", "Linear Autonomous", "Lorenz First Dimension"])):

            y, x, dxdt = sim(duration=4, dt=dt, noise_type='normal', noise_parameters=[0,0.1], outliers=True)
            assert len(y) == len(x) == len(dxdt) == 4/dt # duration/dt

            if request.config.getoption("--plot"):
                t = np.arange(len(y))*dt
                ax = axes[j][i//3, i%3]
                ax.plot(t, x, 'k--', linewidth=3, label=r"true $x$")
                ax.plot(t, y, '.', color='blue', zorder=-100, markersize=5, label="noisy data")
                if i//3 == 0: ax.set_xticklabels([])
                ax.set_title(title, fontsize=18)
                if i == 5: ax.legend(loc='lower right', fontsize=12)


def test_robust_rme():
    """Ensure the robust error metric is the same as RMSE for big M, and that it does
    better in the presence of outliers"""
    u = np.sin(np.arange(100)*0.1)
    v = u + np.random.randn(100)
    assert np.allclose(evaluate.rmse(u, v), evaluate.robust_rme(u, v, M=6))

    v[40] = 100 # throw an outlier in there
    assert evaluate.robust_rme(u, v, M=2) < evaluate.rmse(u, v)
