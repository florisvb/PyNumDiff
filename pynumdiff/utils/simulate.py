"""
Simulation Related
"""
import numpy as np
from scipy.integrate import odeint

# local imports
from pynumdiff.utils.utility import peakdet
from pynumdiff.finite_difference import first_order as _finite_difference


# pylint: disable-msg=too-many-locals, too-many-arguments, no-member
def _add_noise(x, random_seed, noise_type, noise_parameters):
    """Add synthetic noise to data

    :param np.array[float] x: data
    :param int random_seed: an integer seed used to initialize the random number generator
    :param str noise_type: type of noise, compatible with :code:`np.random` functions
                        (eg. 'normal', 'uniform', 'poisson')
    :param noise_parameters: parameters of the noise used in :code:`np.random`, leaving off :code:`size`

    :return: (np.array) -- noisy time series data
    """
    np.random.seed(random_seed)
    noise = np.random.__getattribute__(noise_type)(*noise_parameters, x.shape[0])
    return x + noise


def sine(duration=4, noise_type='normal', noise_parameters=(0, 0.5), random_seed=1,
         dt=0.01, simdt=0.0001, frequencies=(1, 1.7), magnitude=1):
    """Create toy example of time series consisted of sinusoidal modes

    :param float duration: governs the length of the series, duration/dt
    :param str noise_type: type of noise, compatible with :code:`np.random` functions
                        (eg. 'normal', 'uniform', 'poisson')
    :param noise_parameters: parameters of the noise used in :code:`np.random`, leaving off :code:`size`
    :param int random_seed: an integer seed used to initialize the random number generator
    :param float dt: the step size
    :param float simdt: simulation step size used to generate the time series, typically smaller than
            :code:`dt` to achieve high precision
    :param list[float] frequencies: frequencies for sinusoidal modes
    :param float magnitude: magnitude/frequencies[i] is the true magnitude of the ith sinusoidal mode

    :return: tuple[np.array, np.array, np.array, None] of\n
            - **noisy_pos** -- a noisy time series consisted of several sinusoidal modes;
            - **pos** -- a noise-free time series consisted of several sinusoidal modes (truth);
            - **vel** -- a true derivative information of the time series;
            - None -- dummy output
    """
    y_offset = 1

    t = np.arange(0, duration, simdt)
    x = y_offset
    dxdt = 0
    for f in frequencies:
        x += magnitude/len(frequencies)*np.sin(t*2*np.pi*f)
        dxdt += magnitude/len(frequencies)*np.cos(t*2*np.pi*f)*2*np.pi*f
    actual_vals = np.array(np.vstack((x, dxdt)))
    noisy_x = _add_noise(x, random_seed, noise_type, noise_parameters)
    #
    noisy_measurements = np.array(noisy_x)
    #
    noisy_pos = np.ravel(noisy_measurements)
    pos = np.ravel(actual_vals[0, :])
    vel = np.ravel(actual_vals[1, :])

    idx = np.arange(0, len(t), int(dt/simdt))
    return noisy_pos[idx], pos[idx], vel[idx], None


def triangle(duration=4, noise_type='normal', noise_parameters=(0, 0.5), random_seed=1,
             dt=0.01, simdt=0.0001):
    """Create toy example of sharp-edged triangle wave with increasing frequencies

    :param float duration: governs the length of the series, duration/dt
    :param str noise_type: type of noise, compatible with :code:`np.random` functions
                        (eg. 'normal', 'uniform', 'poisson')
    :param noise_parameters: parameters of the noise used in :code:`np.random`, leaving off :code:`size`
    :param int random_seed: an integer seed used to initialize the random number generator
    :param float dt: the step size
    :param float simdt: simulation step size used to generate the time series, typically smaller than
            :code:`dt` to achieve high precision

    :return: tuple[np.array, np.array, np.array, None] of\n
            - **noisy_pos** -- a noisy time series consisted of sharp-edged triangles;
            - **pos** -- a noise-free time series consisted of sharp-edged triangles (truth);
            - **vel** -- a true derivative information of the time series;
            - None -- dummy output
    """
    t = np.arange(0, duration, simdt)
    continuous_x = np.sin(t*t)

    # find peaks and valleys
    peaks, valleys = peakdet(continuous_x, 0.1)

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

    idx = np.argsort(reversal_idxs)
    reversal_idxs = np.array(reversal_idxs)[idx]
    reversal_vals = np.array(reversal_vals)[idx]
    reversal_ts = t[reversal_idxs]

    x = np.interp(t, reversal_ts, reversal_vals)
    _, dxdt = _finite_difference(x, dt=simdt)

    noisy_x = _add_noise(x, random_seed, noise_type, noise_parameters)

    actual_vals = np.array(np.vstack((x, dxdt)))
    noisy_measurements = np.array(noisy_x)

    noisy_pos = np.ravel(noisy_measurements)
    pos = np.ravel(actual_vals[0, :])
    vel = np.ravel(actual_vals[1, :])

    idx = np.arange(0, len(t), int(dt/simdt))
    return noisy_pos[idx], pos[idx], vel[idx], None


def pop_dyn(duration=4, noise_type='normal', noise_parameters=(0, 0.5), random_seed=1,
            dt=0.01, simdt=0.0001):
    """Create toy example of bounded exponential growth:
    http://www.biologydiscussion.com/population/population-growth/population-growth-curves-ecology/51854

    :param float duration: governs the length of the series, duration/dt
    :param str noise_type: type of noise, compatible with :code:`np.random` functions
                        (eg. 'normal', 'uniform', 'poisson')
    :param noise_parameters: parameters of the noise used in :code:`np.random`, leaving off :code:`size`
    :param int random_seed: an integer seed used to initialize the random number generator
    :param float dt: the step size
    :param float simdt: simulation step size used to generate the time series, typically smaller than
            :code:`dt` to achieve high precision
    
    :return: tuple[np.array, np.array, np.array, None] of\n
            - **noisy_pos** -- a noisy time series consisted of bounded exponential growth;
            - **pos** -- a noise-free time series consisted of bounded exponential growth (truth);
            - **vel** -- a true derivative information of the time series;
            - None -- dummy output
    """
    t = np.arange(0, duration, simdt)
    K = 2  # carrying capacity
    r = 4  # biotic potential

    x = [0.1]  # population
    dxdt = [r*x[-1]*(1-x[-1]/K)]
    for _ in t[1:]:
        x.append(x[-1] + simdt*dxdt[-1])
        dxdt.append(r*x[-1]*(1-x[-1]/K))

    x = np.array(x)
    dxdt = np.array(dxdt)

    noisy_x = _add_noise(x, random_seed, noise_type, noise_parameters)

    actual_vals = np.array(np.vstack((x, dxdt)))
    noisy_measurements = np.array(noisy_x)

    noisy_pos = np.ravel(noisy_measurements)
    pos = np.ravel(actual_vals[0, :])
    vel = np.ravel(actual_vals[1, :])

    idx = np.arange(0, len(t), int(dt/simdt))
    return noisy_pos[idx], pos[idx], vel[idx], None


def linear_autonomous(duration=4, noise_type='normal', noise_parameters=(0, 0.5),
                      random_seed=1, dt=0.01, simdt=0.0001):
    """Create toy example of time series from an autonomous linear system

    :param float duration: governs the length of the series, duration/dt
    :param str noise_type: type of noise, compatible with :code:`np.random` functions
                        (eg. 'normal', 'uniform', 'poisson')
    :param noise_parameters: parameters of the noise used in :code:`np.random`, leaving off :code:`size`
    :param int random_seed: an integer seed used to initialize the random number generator
    :param float dt: the step size
    :param float simdt: simulation step size used to generate the time series, typically smaller than
            :code:`dt` to achieve high precision

    :return: tuple[np.array, np.array, np.array, None] of\n
            - **noisy_pos** -- a noisy time series from an autonomous linear system;
            - **pos** -- a noise-free time series from an autonomous linear system (truth);
            - **vel** -- a true derivative information of the time series;
            - None -- dummy output
    """
    t = np.arange(0, duration, simdt)

    A = np.array([[1, simdt, 0], [0, 1, simdt], [-100, -3, 0.01]])
    x0 = np.array([[0], [2], [0]])
    xs = x0
    for _ in t:
        x = A@xs[:,[-1]]
        xs = np.hstack((xs, x))

    x = xs[0,:]
    x *= 2

    smooth_x, dxdt = _finite_difference( np.ravel(x), simdt)
    noisy_x = _add_noise(x, random_seed, noise_type, noise_parameters)

    idx = np.arange(0, len(t), int(dt/simdt))
    return np.ravel(noisy_x)[1:][idx], smooth_x[1:][idx], dxdt[1:][idx], None


def pi_cruise_control(duration=4, noise_type='normal', noise_parameters=(0, 0.5),
               random_seed=1, dt=0.01):
    """Create a toy example of linear proportional integral controller with nonlinear control inputs.
    Simulate proportional integral control of a car attempting to maintain constant velocity while going
    up and down hills. We assume the car has arbitrary power and can achieve whatever acceleration it wants;
    its mass only factors in via -mg pulling it downhill. This is a linear interpretation of something
    similar to what is described in Astrom and Murray 2008 Chapter 3.

    :param float duration: governs the length of the series, duration/dt
    :param str noise_type: type of noise, compatible with :code:`np.random` functions
                        (eg. 'normal', 'uniform', 'poisson')
    :param noise_parameters: parameters of the noise used in :code:`np.random`, leaving off :code:`size`
    :param int random_seed: an integer seed used to initialize the random number generator
    :param float dt: the step size
    :param float simdt: simulation step size used to generate the time series, typically smaller than
            :code:`dt` to achieve high precision

    :return: tuple[np.array, np.array, np.array, None] of\n
            - **noisy_pos** -- a noisy time series of linear proportional integral controller with
                               nonlinear control inputs;
            - **pos** -- a noise-free time series of linear proportional integral controller with
                         nonlinear control inputs (truth);
            - **vel** -- a true derivative information of the time series;
            - **[measurements, controls]** -- a list of extra measurements and controls
    """
    # disturbance
    t = np.arange(0, duration, dt)
    slope = 0.01*(np.sin(2*np.pi*t) + 0.3*np.sin(4*2*np.pi*t + 0.5) + 1.2*np.sin(1.7*2*np.pi*t + 0.5))

    # parameters
    mg = 10000 # mass*gravity
    fr = 0.9 # friction
    ki = 0.05 # integral control
    kp = 0.25 # proportional control
    vd = 0.5 # desired velocity

    # Here state is [pos, vel, accel, cumulative pos error]
    A = np.array([[1,  dt, (dt**2)/2, 0], # Taylor expand out to accel
                  [0,   1,    dt,     0],
                  [0, -fr,     0,    ki/(dt**2)], # (pos error) / dt^2 puts it in units of accel
                  [0,   0,     0,     1]])

    # Here inputs are [slope, vel_desired - vel_estimated]
    B = np.array([[0,   0],
                  [0,   0],
                  [-mg, kp/dt], # (vel error) / dt puts it in units of accel
                  [0,   dt]])

    # run simulation
    states = [np.array([0, 0, 0, 0])] # x0 is all zeros
    controls = []
    for i in range(len(slope)):
        u = np.array([slope[i], vd - states[-1][1]]) # current vel is in 1st position of last state
        xnew = A @ states[-1] + B @ u
        states.append(xnew)
        controls.append(u)

    states = np.vstack(states).T
    controls = np.vstack(controls).T

    x = np.ravel(states[0, :])
    dxdt = np.ravel(states[1, :])
    
    noisy_x = _add_noise(x, random_seed, noise_type, noise_parameters)

    states = np.array(np.vstack((x, dxdt)))
    noisy_measurements = np.array(noisy_x)

    noisy_pos = np.ravel(noisy_measurements)
    pos = np.ravel(states[0, :])
    vel = np.ravel(states[1, :])

    return noisy_pos, pos, vel, [slope, controls]


def lorenz_x(duration=4, noise_type='normal', noise_parameters=(0, 0.5),
             random_seed=1, dt=0.01, simdt=0.0001):
    """Create toy example of x component from a lorenz attractor

    :param float duration: governs the length of the series, duration/dt
    :param str noise_type: type of noise, compatible with :code:`np.random` functions
                        (eg. 'normal', 'uniform', 'poisson')
    :param noise_parameters: parameters of the noise used in :code:`np.random`, leaving off :code:`size`
    :param int random_seed: an integer seed used to initialize the random number generator
    :param float dt: the step size
    :param float simdt: simulation step size used to generate the time series, typically smaller than
            :code:`dt` to achieve high precision

    :return: tuple[np.array, np.array, np.array, None] of\n
            - **noisy_pos** -- a noisy time series of x component from Lorenz system;
            - **pos** -- a noise-free time series of x component from Lorenz system (truth);
            - **vel** -- a true derivative information of the time series;
            - None -- dummy output
    """
    noisy_measurements, actual_vals, _ = _lorenz_xyz(duration, noise_type, noise_parameters,
                                                    random_seed, dt, simdt)

    noisy_pos = np.ravel(noisy_measurements[0, :])
    pos = np.ravel(actual_vals[0, :])
    vel = np.ravel(actual_vals[3, :])

    return noisy_pos, pos, vel, None


def _lorenz_xyz(duration=4, noise_type='normal', noise_parameters=(0, 0.5), random_seed=1,
               dt=0.01, simdt=0.0001, x0=(5, 1, 3), normalize=True):
    """Simulation of Lorenz system with Eular method

    :param float duration: governs the length of the series, duration/dt
    :param str noise_type: type of noise, compatible with :code:`np.random` functions
                        (eg. 'normal', 'uniform', 'poisson')
    :param noise_parameters: parameters of the noise used in :code:`np.random`, leaving off :code:`size`
    :param int random_seed: an integer seed used to initialize the random number generator
    :param float dt: the step size
    :param float simdt: simulation step size used to generate the time series, typically smaller than
            :code:`dt` to achieve high precision
    :param tuple x0: a tuple of initial state of the Lorenz system
    :param bool normalize: whether to roughly normalize the time series

    :return: tuple[np.array, np.array, None] of\n
            - **noisy_measurements** -- noisy time series from Lorenz system;
            - **actual_vals** -- noise-free time series from Lorenz system;
            - None -- dummy output
    """
    t = np.arange(0, duration, simdt)

    sigma = 10
    beta = 8/3
    rho = 45

    x = x0[0]
    y = x0[1]
    z = x0[2]
    xyz = np.array([[x], [y], [z]])
    xyz_dot = None

    for _ in t:
        x, y, z = np.ravel(xyz[:, -1])

        xdot = sigma*(y-x)
        ydot = x*(rho-z)-y
        zdot = x*y - beta*z

        new_xyz_dot = np.array([[xdot], [ydot], [zdot]])
        if xyz_dot is None:
            xyz_dot = new_xyz_dot
        else:
            xyz_dot = np.hstack((xyz_dot, new_xyz_dot))

        new_xyz = xyz[:, [-1]] + simdt*new_xyz_dot
        xyz = np.hstack((xyz, new_xyz))

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

    noisy_x = _add_noise(x, random_seed, noise_type, noise_parameters)
    noisy_y = _add_noise(y, random_seed+1, noise_type, noise_parameters)
    noisy_z = _add_noise(z, random_seed+2, noise_type, noise_parameters)

    actual_vals = np.array(np.vstack((x, y, z, dxdt, dydt, dzdt)))
    noisy_measurements = np.array(np.vstack((noisy_x, noisy_y, noisy_z)))

    idx = np.arange(0, len(t), int(dt/simdt))
    return noisy_measurements[:, idx], actual_vals[:, idx], None


def _rk4_lorenz_xyz(duration=4, noise_type='normal', noise_parameters=(0, 0.5),
                   random_seed=1, dt=0.01, normalize=True):
    """
    :param float duration: governs the length of the series, duration/dt
    :param str noise_type: type of noise, compatible with :code:`np.random` functions
                        (eg. 'normal', 'uniform', 'poisson')
    :param noise_parameters: parameters of the noise used in :code:`np.random`, leaving off :code:`size`
    :param int random_seed: an integer seed used to initialize the random number generator
    :param float dt: the step size
    :param float simdt: simulation step size used to generate the time series, typically smaller than
            :code:`dt` to achieve high precision
    :param bool normalize: whether to roughly normalize the time series

    :return: tuple[np.array, np.array, np.array, None] of\n
            - **noisy_measurements** -- noisy time series from Lorenz system;
            - **actual_vals** -- noise-free time series from Lorenz system;
            - None -- dummy output
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

    ts = np.linspace(0, duration, int(duration/dt))
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

    noisy_x = _add_noise(x, random_seed, noise_type, noise_parameters)
    noisy_y = _add_noise(y, random_seed+1, noise_type, noise_parameters)
    noisy_z = _add_noise(z, random_seed+2, noise_type, noise_parameters)

    _, dxdt = _finite_difference(x, dt)
    _, dydt = _finite_difference(y, dt)
    _, dzdt = _finite_difference(z, dt)

    actual_vals = np.array(np.vstack((x, y, z, dxdt, dydt, dzdt)))
    noisy_measurements = np.array(np.vstack((noisy_x, noisy_y, noisy_z)))

    return noisy_measurements, actual_vals, None
