"""
Simulation Related
"""
import numpy as np
from scipy.integrate import odeint

# local imports
from pynumdiff.utils.utility import peakdet
from pynumdiff.finite_difference import second_order as finite_difference


# pylint: disable-msg=too-many-locals, too-many-arguments, no-member
def _add_noise(x, random_seed, noise_type, noise_parameters, outliers=False):
    """Add synthetic noise to data

    :param np.array[float] x: data
    :param int random_seed: an integer seed used to initialize the random number generator
    :param str noise_type: type of noise, compatible with :code:`np.random` functions
                        (eg. 'normal', 'uniform', 'poisson')
    :param array-like noise_parameters: parameters of the noise used in :code:`np.random`, leaving off :code:`size`
    :param bool outliers: whether to corrupt 1% of the data points with out-of-distribution values

    :return: (np.array) -- noisy time series data
    """
    np.random.seed(random_seed)
    noise = np.random.__getattribute__(noise_type)(*noise_parameters, x.shape[0])
    if outliers:
        ndxs = np.random.choice(len(noise), len(noise)//100, replace=False) # select 1% of locations
        noise[ndxs] = (np.random.choice([-1, 1], size=len(ndxs)) + np.random.uniform(-0.5, 0.5, len(ndxs)))*(np.max(x) - np.min(x))
    return x + noise


def sine(duration=4, noise_type='normal', noise_parameters=(0, 0.5), outliers=False, random_seed=1,
         dt=0.01, simdt=0.0001, frequencies=(1, 1.7), magnitude=1):
    """Create toy example of time series consisted of sinusoidal modes

    :param float duration: governs the length of the series, duration/dt
    :param str noise_type: type of noise, compatible with :code:`np.random` functions
                        (eg. 'normal', 'uniform', 'poisson')
    :param noise_parameters: parameters of the noise used in :code:`np.random`, leaving off :code:`size`
    :param bool outliers: whether to corrupt 1% of the data points with out-of-distribution values
    :param int random_seed: an integer seed used to initialize the random number generator
    :param float dt: the step size
    :param float simdt: simulation step size used to generate the time series, typically smaller than
            :code:`dt` to achieve high precision
    :param list[float] frequencies: frequencies for sinusoidal modes
    :param float magnitude: magnitude/frequencies[i] is the true magnitude of the ith sinusoidal mode

    :return: tuple[np.array, np.array, np.array] of\n
            - **noisy_pos** -- a noisy time series consisted of several sinusoidal modes;
            - **pos** -- a noise-free time series consisted of several sinusoidal modes (truth);
            - **vel** -- a true derivative information of the time series
    """
    t = np.arange(0, duration, simdt)
    pos = 1 # initial y offset
    vel = 0
    for f in frequencies:
        pos += magnitude/len(frequencies)*np.sin(t*2*np.pi*f)
        vel += magnitude/len(frequencies)*np.cos(t*2*np.pi*f)*2*np.pi*f

    idx = slice(0, len(t), int(dt/simdt)) # downsample so things are dt apart
    noisy_pos = _add_noise(pos[idx], random_seed, noise_type, noise_parameters, outliers)

    return noisy_pos, pos[idx], vel[idx]


def triangle(duration=4, noise_type='normal', noise_parameters=(0, 0.5), outliers=False, random_seed=1,
             dt=0.01, simdt=0.0001):
    """Create toy example of sharp-edged triangle wave with increasing frequencies

    :param float duration: governs the length of the series, duration/dt
    :param str noise_type: type of noise, compatible with :code:`np.random` functions
                        (eg. 'normal', 'uniform', 'poisson')
    :param noise_parameters: parameters of the noise used in :code:`np.random`, leaving off :code:`size`
    :param bool outliers: whether to corrupt 1% of the data points with out-of-distribution values
    :param int random_seed: an integer seed used to initialize the random number generator
    :param float dt: the step size
    :param float simdt: simulation step size used to generate the time series, typically smaller than
            :code:`dt` to achieve high precision

    :return: tuple[np.array, np.array, np.array] of\n
            - **noisy_pos** -- a noisy time series consisted of sharp-edged triangles;
            - **pos** -- a noise-free time series consisted of sharp-edged triangles (truth);
            - **vel** -- a true derivative information of the time series
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

    pos = np.interp(t, reversal_ts, reversal_vals)
    _, vel = finite_difference(pos, dt=simdt)
    noisy_pos = _add_noise(pos, random_seed, noise_type, noise_parameters, outliers)

    idx = np.arange(0, len(t), int(dt/simdt))
    return noisy_pos[idx], pos[idx], vel[idx]


def pop_dyn(duration=4, noise_type='normal', noise_parameters=(0, 0.5), outliers=False, random_seed=1,
            dt=0.01, simdt=0.0001):
    """Create toy example of bounded exponential growth:
    http://www.biologydiscussion.com/population/population-growth/population-growth-curves-ecology/51854

    :param float duration: governs the length of the series, duration/dt
    :param str noise_type: type of noise, compatible with :code:`np.random` functions
                        (eg. 'normal', 'uniform', 'poisson')
    :param noise_parameters: parameters of the noise used in :code:`np.random`, leaving off :code:`size`
    :param bool outliers: whether to corrupt 1% of the data points with out-of-distribution values
    :param int random_seed: an integer seed used to initialize the random number generator
    :param float dt: the step size
    :param float simdt: simulation step size used to generate the time series, typically smaller than
            :code:`dt` to achieve high precision
    
    :return: tuple[np.array, np.array, np.array] of\n
            - **noisy_pos** -- a noisy time series consisted of bounded exponential growth;
            - **pos** -- a noise-free time series consisted of bounded exponential growth (truth);
            - **vel** -- a true derivative information of the time series
    """
    t = np.arange(0, duration, simdt)
    K = 2  # carrying capacity
    r = 4  # biotic potential

    pos = [0.1]  # population
    vel = [r*pos[-1]*(1-pos[-1]/K)]
    for _ in t[1:]:
        pos.append(pos[-1] + simdt*vel[-1])
        vel.append(r*pos[-1]*(1-pos[-1]/K)) # logistic growth
    
    idx = slice(0, len(t), int(dt/simdt)) # downsample so things are dt apart
    pos = np.array(pos[idx])
    vel = np.array(vel[idx])
    noisy_pos = _add_noise(pos, random_seed, noise_type, noise_parameters, outliers)

    return noisy_pos, pos, vel


def linear_autonomous(duration=4, noise_type='normal', noise_parameters=(0, 0.5), outliers=False,
                      random_seed=1, dt=0.01, simdt=0.0001):
    """Create toy example of time series from an autonomous linear system

    :param float duration: governs the length of the series, duration/dt
    :param str noise_type: type of noise, compatible with :code:`np.random` functions
                        (eg. 'normal', 'uniform', 'poisson')
    :param noise_parameters: parameters of the noise used in :code:`np.random`, leaving off :code:`size`
    :param bool outliers: whether to corrupt 1% of the data points with out-of-distribution values
    :param int random_seed: an integer seed used to initialize the random number generator
    :param float dt: the step size
    :param float simdt: simulation step size used to generate the time series, typically smaller than
            :code:`dt` to achieve high precision

    :return: tuple[np.array, np.array, np.array] of\n
            - **noisy_pos** -- a noisy time series from an autonomous linear system;
            - **pos** -- a noise-free time series from an autonomous linear system (truth);
            - **vel** -- a true derivative information of the time series
    """
    t = np.arange(0, duration, simdt)

    A = np.array([[1, simdt, (simdt**2)/2], [0, 1, simdt], [-100, -3, 0.01]]) # All |eigs| <1, so stable
    xs = [np.array([0, 20, 0])]
    for _ in t:
        xs.append(A @ xs[-1])

    xs = np.vstack(xs).T
    pos = xs[0,:]

    smooth_pos, vel = finite_difference(pos, simdt)
    noisy_pos = _add_noise(pos, random_seed, noise_type, noise_parameters, outliers)

    idx = slice(0, len(t), int(dt/simdt)) # downsample so things are dt apart
    return noisy_pos[1:][idx], smooth_pos[1:][idx], vel[1:][idx]


def pi_cruise_control(duration=4, noise_type='normal', noise_parameters=(0, 0.5), outliers=False,
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
    :param bool outliers: whether to corrupt 1% of the data points with out-of-distribution values
    :param int random_seed: an integer seed used to initialize the random number generator
    :param float dt: the step size
    :param float simdt: simulation step size used to generate the time series, typically smaller than
            :code:`dt` to achieve high precision

    :return: tuple[np.array, np.array, np.array] of\n
            - **noisy_pos** -- a noisy time series of linear proportional integral controller with
                               nonlinear control inputs;
            - **pos** -- a noise-free time series of linear proportional integral controller with
                         nonlinear control inputs (truth);
            - **vel** -- a true derivative information of the time series
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

    pos = states[0, :]
    vel = states[1, :]
    noisy_pos = _add_noise(pos, random_seed, noise_type, noise_parameters, outliers)

    return noisy_pos, pos, vel


def lorenz_x(duration=4, noise_type='normal', noise_parameters=(0, 0.5), outliers=False,
             random_seed=1, dt=0.01, simdt=0.0001):
    """Create toy example of x component from a lorenz attractor

    :param float duration: governs the length of the series, duration/dt
    :param str noise_type: type of noise, compatible with :code:`np.random` functions
                        (eg. 'normal', 'uniform', 'poisson')
    :param noise_parameters: parameters of the noise used in :code:`np.random`, leaving off :code:`size`
    :param bool outliers: whether to corrupt 1% of the data points with out-of-distribution values
    :param int random_seed: an integer seed used to initialize the random number generator
    :param float dt: the step size
    :param float simdt: simulation step size used to generate the time series, typically smaller than
            :code:`dt` to achieve high precision

    :return: tuple[np.array, np.array, np.array] of\n
            - **noisy_pos** -- a noisy time series of x component from Lorenz system;
            - **pos** -- a noise-free time series of x component from Lorenz system (truth);
            - **vel** -- a true derivative information of the time series
    """
    sigma = 10
    beta = 8/3
    rho = 45

    def _lorenz_xyz_euler(x0=(5, 1, 3), normalize=True):
        """Simulation of Lorenz system with Eular method"""
        t = np.arange(0, duration, simdt)

        xyz = [np.array(x0)]
        xyz_dot = []
        for _ in t:
            x, y, z = xyz[-1]

            xdot = sigma*(y-x)
            ydot = x*(rho-z)-y
            zdot = x*y - beta*z

            xyz_dot.append(np.array([xdot, ydot, zdot]))
            xyz.append(xyz[-1] + simdt*xyz_dot[-1])

        idx = slice(0, len(t), int(dt/simdt)) # downsample so things are dt apart
        xyz = np.vstack(xyz[idx]).T
        xyz_dot = np.vstack(xyz_dot[idx]).T

        f = 20 if normalize else 1
        xyz /= f # because xyz is one longer than xyz_dot, leave off last entry
        xyz_dot /= f

        noisy_xyz = np.vstack([_add_noise(xyz[i,:], random_seed+i, noise_type, noise_parameters, outliers) for i in range(3)])
        return noisy_xyz, xyz, xyz_dot

    def _lorenz_xyz_odeint(normalize=True):
        """Simulate the Lorenz system with scipy's ODE solver"""
        t = np.linspace(0, duration, int(duration/dt))

        def dxyz_dt(xyz, t_): # t_ is the time, unused because the Lorenz system is autonomous
            """Right hand side of Lorenz system

            :param xyz: (list of floats) state variables
            :return: (list of floats) derivatives of the state variables
            """
            x, y, z = xyz
            xdot = sigma*(y-x)
            ydot = x*(rho-z)-y
            zdot = x*y - beta*z
            return [xdot, ydot, zdot]

        xyz = odeint(dxyz_dt, [5, 1, 3], t).T
        xyz_dot = np.array([dxyz_dt(xyz[:,i], None) for i in range(len(t))]).T # evaluate the RHS at all the locations
        if normalize:
            xyz /= 20
            xyz_dot /= 20

        noisy_xyz = np.vstack([_add_noise(xyz[i,:], random_seed+i, noise_type, noise_parameters, outliers) for i in range(3)])
        return noisy_xyz, xyz, xyz_dot

    noisy_pos, pos, vel = _lorenz_xyz_euler()
    return noisy_pos[0,:], pos[0,:], vel[0,:]
