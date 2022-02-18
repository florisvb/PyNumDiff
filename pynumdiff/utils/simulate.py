"""
Simulation Related
"""
import numpy as _np
from scipy.integrate import odeint

# local imports
from pynumdiff.utils import utility as _utility
from pynumdiff.utils import _pi_cruise_control
_finite_difference = _utility.finite_difference


# pylint: disable-msg=too-many-locals, too-many-arguments, no-member
def __add_noise__(x, noise_type, noise_parameters, random_seed):
    """
    Adding synthetic noise to the time series data

    :param x: an array of time series data
    :type x: np.array
    :param noise_type: a string representing type of noise, should be compatible with np.random functions
                       (eg. 'normal', 'uniform', 'poisson')
    :type noise_type: string
    :param noise_parameters: a list of parameters of the noise used in np.random
    :type noise_parameters: list
    :param random_seed: an integer seed used to initialize the random number generator
    :type random_seed: int

    :return: noisy time series data
    :rtype: np.array
    """
    _np.random.seed(random_seed)
    timeseries_length = _np.max(x.shape)
    noise = _np.random.__getattribute__(noise_type)(noise_parameters[0], noise_parameters[1],
                                                    timeseries_length)
    return x + noise


def sine(timeseries_length=4, noise_type='normal', noise_parameters=(0, 0.5), random_seed=1,
         dt=0.01, simdt=0.0001, frequencies=(1, 1.7), magnitude=1):
    """
    Create toy example of time series consisted of sinusoidal modes

    :param timeseries_length: a float number representing the period of the time series, not to be confused with number
                              of data points
    :type timeseries_length: float, optional

    :param noise_type: a string representing type of noise, should be compatible with np.random functions
                       (eg. 'normal', 'uniform', 'poisson')
    :type noise_type: string, optional

    :param noise_parameters: a list of parameters of the noise used in np.random
    :type noise_parameters: list, optional

    :param random_seed: an integer seed used to initialize the random number generator
    :type random_seed: int, optional

    :param dt: a float number representing the time step size
    :type dt: float, optional

    :param simdt: a float number representing the the real simulation step size we use to generate the time series,
                  typically smaller than dt to achieve high precision
    :type simdt: float, optional

    :param frequencies: a list of float numbers representing frequencies for each sinusoidal modes
    :type frequencies: list (float), optional

    :param magnitude: magnitude/frequencies[i] is the true magnitude of the ith sinusoidal mode
    :type magnitude: float, optional

    :return: a tuple consisting of:

                 - a noisy time series consisted of several sinusoidal modes;
                 - a noise-free time series consisted of several sinusoidal modes (truth);
                 - a true derivative information of the time series;
                 - None: dummy output

    :rtype: tuple -> (np.array, np.array, np.array, None)
    """
    y_offset = 1

    t = _np.arange(0, timeseries_length, simdt)
    x = y_offset
    dxdt = 0
    for f in frequencies:
        x += magnitude/len(frequencies)*_np.sin(t*2*_np.pi*f)
        dxdt += magnitude/len(frequencies)*_np.cos(t*2*_np.pi*f)*2*_np.pi*f
    actual_vals = _np.array(_np.vstack((x, dxdt)))
    noisy_x = __add_noise__(x, noise_type, noise_parameters, random_seed)
    #
    noisy_measurements = _np.array(noisy_x)
    #
    noisy_pos = _np.ravel(noisy_measurements)
    pos = _np.ravel(actual_vals[0, :])
    vel = _np.ravel(actual_vals[1, :])

    idx = _np.arange(0, len(t), int(dt/simdt))
    return noisy_pos[idx], pos[idx], vel[idx], None


def triangle(timeseries_length=4, noise_type='normal', noise_parameters=(0, 0.5), random_seed=1,
             dt=0.01, simdt=0.0001):
    """
    Create toy example of sharp-edged triangle wave with increasing frequencies

    :param timeseries_length: a float number representing the period of the time series, not to be confused with number
                              of data points
    :type timeseries_length: float, optional

    :param noise_type: a string representing type of noise, should be compatible with np.random functions
                       (eg. 'normal', 'uniform', 'poisson')
    :type noise_type: string, optional

    :param noise_parameters: a list of parameters of the noise used in np.random
    :type noise_parameters: list, optional

    :param random_seed: an integer seed used to initialize the random number generator
    :type random_seed: int, optional

    :param dt: a float number representing the time step size
    :type dt: float, optional

    :param simdt: a float number representing the the real simulation step size we use to generate the time series,
                  typically smaller than dt to achieve high precision
    :type simdt: float, optional

    :return: a tuple consisting of:

                 - a noisy time series consisted of sharp-edged triangles;
                 - a noise-free time series consisted of sharp-edged triangles (truth);
                 - a true derivative information of the time series;
                 - None: dummy output

    :rtype: tuple -> (np.array, np.array, np.array, None)
    """
    t = _np.arange(0, timeseries_length, simdt)
    continuous_x = _np.sin(t*t)

    # find peaks and valleys
    peaks, valleys = _utility.peakdet(continuous_x, 0.1)

    # organize peaks and valleys
    if len(peaks) > 0:
        reversal_idxs = peaks[:, 0].astype(int).tolist()
        reversal_idxs.extend(valleys[:, 0].astype(int).tolist())
        reversal_idxs.extend([0, len(continuous_x)-1])

        reversal_vals = peaks[:, 1].tolist()
        reversal_vals.extend(valleys[:, 1].tolist())
        reversal_vals.extend([0, continuous_x[-1]])
    else:
        reversal_idxs = [0, len(continuous_x)-1]
        reversal_vals = [0, continuous_x[-1]]

    idx = _np.argsort(reversal_idxs)
    reversal_idxs = _np.array(reversal_idxs)[idx]
    reversal_vals = _np.array(reversal_vals)[idx]
    reversal_ts = t[reversal_idxs]

    x = _np.interp(t, reversal_ts, reversal_vals)
    _, dxdt = _finite_difference(x, dt=simdt)

    noisy_x = __add_noise__(x, noise_type, noise_parameters, random_seed)

    actual_vals = _np.array(_np.vstack((x, dxdt)))
    noisy_measurements = _np.array(noisy_x)

    noisy_pos = _np.ravel(noisy_measurements)
    pos = _np.ravel(actual_vals[0, :])
    vel = _np.ravel(actual_vals[1, :])

    idx = _np.arange(0, len(t), int(dt/simdt))
    return noisy_pos[idx], pos[idx], vel[idx], None


def pop_dyn(timeseries_length=4, noise_type='normal', noise_parameters=(0, 0.5), random_seed=1,
            dt=0.01, simdt=0.0001):
    """
    Create toy example of bounded exponential growth: http://www.biologydiscussion.com/population/population-growth/population-growth-curves-ecology/51854

    :param timeseries_length: a float number representing the period of the time series, not to be confused with number
                              of data points
    :type timeseries_length: float, optional

    :param noise_type: a string representing type of noise, should be compatible with np.random functions
                       (eg. 'normal', 'uniform', 'poisson')
    :type noise_type: string, optional

    :param noise_parameters: a list of parameters of the noise used in np.random
    :type noise_parameters: list, optional

    :param random_seed: an integer seed used to initialize the random number generator
    :type random_seed: int, optional

    :param dt: a float number representing the time step size
    :type dt: float, optional

    :param simdt: a float number representing the the real simulation step size we use to generate the time series,
                  typically smaller than dt to achieve high precision
    :type simdt: float, optional

    :return: a tuple consisting of:

                 - a noisy time series consisted of bounded exponential growth;
                 - a noise-free time series consisted of bounded exponential growth (truth);
                 - a true derivative information of the time series;
                 - None: dummy output

    :rtype: tuple -> (np.array, np.array, np.array, None)
    """
    t = _np.arange(0, timeseries_length, simdt)
    K = 2  # carrying capacity
    r = 4  # biotic potential

    x = [0.1]  # population
    dxdt = [r*x[-1]*(1-x[-1]/K)]
    for _ in t[1:]:
        x.append(x[-1] + simdt*dxdt[-1])
        dxdt.append(r*x[-1]*(1-x[-1]/K))

    x = _np.array(x)
    dxdt = _np.array(dxdt)

    noisy_x = __add_noise__(x, noise_type, noise_parameters, random_seed)

    actual_vals = _np.array(_np.vstack((x, dxdt)))
    noisy_measurements = _np.array(noisy_x)

    noisy_pos = _np.ravel(noisy_measurements)
    pos = _np.ravel(actual_vals[0, :])
    vel = _np.ravel(actual_vals[1, :])

    idx = _np.arange(0, len(t), int(dt/simdt))
    return noisy_pos[idx], pos[idx], vel[idx], None


def linear_autonomous(timeseries_length=4, noise_type='normal', noise_parameters=(0, 0.5),
                      random_seed=1, dt=0.01, simdt=0.0001):
    """
    Create toy example of time series from an autonomous linear system

    :param timeseries_length: a float number representing the period of the time series, not to be confused with number
                              of data points
    :type timeseries_length: float, optional

    :param noise_type: a string representing type of noise, should be compatible with np.random functions
                       (eg. 'normal', 'uniform', 'poisson')
    :type noise_type: string, optional

    :param noise_parameters: a list of parameters of the noise used in np.random
    :type noise_parameters: list, optional

    :param random_seed: an integer seed used to initialize the random number generator
    :type random_seed: int, optional

    :param dt: a float number representing the time step size
    :type dt: float, optional

    :param simdt: a float number representing the the real simulation step size we use to generate the time series,
                  typically smaller than dt to achieve high precision
    :type simdt: float, optional

    :return: a tuple consisting of:

                 - a noisy time series from an autonomous linear system;
                 - a noise-free time series from an autonomous linear system (truth);
                 - a true derivative information of the time series;
                 - None: dummy output

    :rtype: tuple -> (np.array, np.array, np.array, None)
    """
    t = _np.arange(0, timeseries_length, simdt)

    A = _np.array([[1, simdt, 0], [0, 1, simdt], [-100, -3, 0.01]])
    x0 = _np.array([[0], [2], [0]])
    xs = x0
    for _ in t:
        x = A@xs[:,[-1]]
        xs = _np.hstack((xs, x))

    x = xs[0,:]
    x *= 2

    smooth_x, dxdt = _finite_difference( _np.ravel(x), simdt)
    noisy_x = __add_noise__(x, noise_type, noise_parameters, random_seed)

    idx = _np.arange(0, len(t), int(dt/simdt))
    return _np.ravel(noisy_x)[1:][idx], smooth_x[1:][idx], dxdt[1:][idx], None


def pi_control(timeseries_length=4, noise_type='normal', noise_parameters=(0, 0.5),
               random_seed=1, dt=0.01):
    """
    Create a toy example of linear proportional integral controller with nonlinear control inputs

    :param timeseries_length: a float number representing the period of the time series, not to be confused with number
                              of data points
    :type timeseries_length: float, optional

    :param noise_type: a string representing type of noise, should be compatible with np.random functions
                       (eg. 'normal', 'uniform', 'poisson')
    :type noise_type: string, optional

    :param noise_parameters: a list of parameters of the noise used in np.random
    :type noise_parameters: list, optional

    :param random_seed: an integer seed used to initialize the random number generator
    :type random_seed: int, optional

    :param dt: a float number representing the time step size
    :type dt: float, optional

    :param simdt: a float number representing the the real simulation step size we use to generate the time series,
                  typically smaller than dt to achieve high precision
    :type simdt: float, optional

    :return: a tuple consisting of:

                 - a noisy time series of linear proportional integral controller with
                    nonlinear control inputs;
                 - a noise-free time series of linear proportional integral controller with
                    nonlinear control inputs (truth);
                 - a true derivative information of the time series;
                 - a list of extra measurements and controls

    :rtype: tuple -> (np.array, np.array, np.array, list (np.array))
    """
    t = _np.arange(0, timeseries_length, dt)

    actual_vals, extra_measurements, controls = _pi_cruise_control.run(timeseries_length, dt)
    x = _np.ravel(actual_vals[0, :])
    dxdt = _np.ravel(actual_vals[1, :])
    
    noisy_x = __add_noise__(x, noise_type, noise_parameters, random_seed)

    actual_vals = _np.array(_np.vstack((x, dxdt)))
    noisy_measurements = _np.array(noisy_x)

    noisy_pos = _np.ravel(noisy_measurements)
    pos = _np.ravel(actual_vals[0, :])
    vel = _np.ravel(actual_vals[1, :])

    return noisy_pos, pos, vel, \
           [_np.array(extra_measurements), _np.array(controls)]


def lorenz_x(timeseries_length=4, noise_type='normal', noise_parameters=(0, 0.5),
             random_seed=1, dt=0.01, simdt=0.0001):
    """
    Create toy example of x component from a lorenz attractor

    :param timeseries_length: a float number representing the period of the time series, not to be confused with number
                              of data points
    :type timeseries_length: float, optional

    :param noise_type: a string representing type of noise, should be compatible with np.random functions
                       (eg. 'normal', 'uniform', 'poisson')
    :type noise_type: string, optional

    :param noise_parameters: a list of parameters of the noise used in np.random
    :type noise_parameters: list, optional

    :param random_seed: an integer seed used to initialize the random number generator
    :type random_seed: int, optional

    :param dt: a float number representing the time step size
    :type dt: float, optional

    :param simdt: a float number representing the the real simulation step size we use to generate the time series,
                  typically smaller than dt to achieve high precision
    :type simdt: float, optional

    :return: a tuple consisting of:

                 - a noisy time series of x component from Lorenz system;
                 - a noise-free time series of x component from Lorenz system (truth);
                 - a true derivative information of the time series;
                 - None: dummy output

    :rtype: tuple -> (np.array, np.array, np.array, list (np.array))
    """
    noisy_measurements, actual_vals, _ = lorenz_xyz(timeseries_length, noise_type, noise_parameters,
                                                    random_seed, dt, simdt)

    noisy_pos = _np.ravel(noisy_measurements[0, :])
    pos = _np.ravel(actual_vals[0, :])
    vel = _np.ravel(actual_vals[3, :])

    return noisy_pos, pos, vel, None


def lorenz_xyz(timeseries_length=4, noise_type='normal', noise_parameters=(0, 0.5), random_seed=1,
               dt=0.01, simdt=0.0001, x0=(5, 1, 3), normalize=True):
    """
    Simulation of Lorenz system with Eular method

    :param timeseries_length: a float number representing the period of the time series, not to be confused with number
                              of data points
    :type timeseries_length: float, optional

    :param noise_type: a string representing type of noise, should be compatible with np.random functions
                       (eg. 'normal', 'uniform', 'poisson')
    :type noise_type: string, optional

    :param noise_parameters: a list of parameters of the noise used in np.random
    :type noise_parameters: list, optional

    :param random_seed: an integer seed used to initialize the random number generator
    :type random_seed: int, optional

    :param dt: a float number representing the time step size
    :type dt: float, optional

    :param simdt: a float number representing the the real simulation step size we use to generate the time series,
                  typically smaller than dt to achieve high precision
    :type simdt: float, optional

    :param x0: a tuple of initial state of the Lorenz system
    :type x0: tuple, optional

    :param normalize: whether to roughly normalize the time series
    :type normalize: boolean, optional

    :return: a tuple consisting of:

                 - noisy_measurements: noisy time series from Lorenz system;
                 - actual_vals: noise-free time series from Lorenz system;
                 - None: dummy output

    :rtype: tuple -> (np.array, np.array, None)
    """
    t = _np.arange(0, timeseries_length, simdt)

    sigma = 10
    beta = 8/3
    rho = 45

    x = x0[0]
    y = x0[1]
    z = x0[2]
    xyz = _np.array([[x], [y], [z]])
    xyz_dot = None

    for _ in t:
        x, y, z = _np.ravel(xyz[:, -1])

        xdot = sigma*(y-x)
        ydot = x*(rho-z)-y
        zdot = x*y - beta*z

        new_xyz_dot = _np.array([[xdot], [ydot], [zdot]])
        if xyz_dot is None:
            xyz_dot = new_xyz_dot
        else:
            xyz_dot = _np.hstack((xyz_dot, new_xyz_dot))

        new_xyz = xyz[:, [-1]] + simdt*new_xyz_dot
        xyz = _np.hstack((xyz, new_xyz))

    if normalize:
        f = 20
    else:
        f = 1

    x = xyz[0, 0:-1] / f
    dxdt = xyz_dot[0, :] / f

    y = xyz[1, 0:-1] / f
    dydt = xyz_dot[1, :] / f

    z = xyz[2, 0:-1] / f
    dzdt = xyz_dot[2, :] / f

    noisy_x = __add_noise__(x, noise_type, noise_parameters, random_seed)
    noisy_y = __add_noise__(y, noise_type, noise_parameters, random_seed+1)
    noisy_z = __add_noise__(z, noise_type, noise_parameters, random_seed+2)

    actual_vals = _np.array(_np.vstack((x, y, z, dxdt, dydt, dzdt)))
    noisy_measurements = _np.array(_np.vstack((noisy_x, noisy_y, noisy_z)))

    idx = _np.arange(0, len(t), int(dt/simdt))
    return noisy_measurements[:, idx], actual_vals[:, idx], None


def rk4_lorenz_xyz(timeseries_length=4, noise_type='normal', noise_parameters=(0, 0.5),
                   random_seed=1, dt=0.01, normalize=True):
    """
    :param timeseries_length: a float number representing the period of the time series, not to be confused with number
                              of data points
    :type timeseries_length: float, optional

    :param noise_type: a string representing type of noise, should be compatible with np.random functions
                       (eg. 'normal', 'uniform', 'poisson')
    :type noise_type: string, optional

    :param noise_parameters: a list of parameters of the noise used in np.random
    :type noise_parameters: list, optional

    :param random_seed: an integer seed used to initialize the random number generator
    :type random_seed: int, optional

    :param dt: a float number representing the time step size
    :type dt: float, optional

    :param normalize: whether to roughly normalize the time series
    :type normalize: boolean, optional

    :return: a tuple consisting of:

                 - noisy_measurements: noisy time series from Lorenz system;
                 - actual_vals: noise-free time series from Lorenz system;
                 - None: dummy output

    :rtype: tuple -> (np.array, np.array, None)
    """

    sigma = 10
    beta = 8/3
    rho = 45

    def dxyz_dt(xyz):
        """
        right hand side of Lorenz system

        :param xyz: (list of floats) state variables
        :return: (list of floats) derivatives of the state variables
        """
        x, y, z = xyz
        xdot = sigma*(y-x)
        ydot = x*(rho-z)-y
        zdot = x*y - beta*z
        return [xdot, ydot, zdot]

    ts = _np.linspace(0, timeseries_length, timeseries_length/dt)
    xyz_0 = [5, 1, 3]

    vals, _ = odeint(dxyz_dt, xyz_0, ts, full_output=True)
    vals = vals.T

    if normalize:
        x = vals[0, :]/20.
        y = vals[1, :]/20.
        z = vals[2, :]/20.
    else:
        x = vals[0, :]
        y = vals[1, :]
        z = vals[2, :]

    noisy_x = __add_noise__(x, noise_type, noise_parameters, random_seed)
    noisy_y = __add_noise__(y, noise_type, noise_parameters, random_seed+1)
    noisy_z = __add_noise__(z, noise_type, noise_parameters, random_seed+2)

    _, dxdt = _finite_difference(x, dt)
    _, dydt = _finite_difference(y, dt)
    _, dzdt = _finite_difference(z, dt)

    actual_vals = _np.array(_np.vstack((x, y, z, dxdt, dydt, dzdt)))
    noisy_measurements = _np.array(_np.vstack((noisy_x, noisy_y, noisy_z)))

    return noisy_measurements, actual_vals, None
