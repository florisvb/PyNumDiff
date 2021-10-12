"""
Simulation and Control of a cruise ?
"""

import numpy as np
from pynumdiff.utils import utility

parameters = {'omega_m': 420,
              'T_m': 190,
              'beta': 0.4,
              'Cr': 0.01,
              'Cd': 0.32,
              'A': 2.4,
              'g': 9.8,
              'm': 3000,
              'rho': 1.3,
              'v_r': 30,
              'k_p': 2,
              'k_i': 2
              }


def triangle(iterations, dt):
    """
    Create sawtooth pattern of hills for the car to drive over.

    :param iterations: number of time points in time series
    :type iterations: int

    :param dt: time step in seconds
    :type dt: float

    :return: time series of hills (angle of road as a function of time)
    :rtype: np.matrix

    """
    t = np.arange(0, iterations*dt, dt)
    continuous_x = np.sin(0.02*t*np.sqrt(t))

    # find peaks and valleys
    peaks, valleys = utility.peakdet(continuous_x, 0.1)

    # organize peaks and valleys
    if len(peaks) > 0:
        reversal_idxs = peaks[:, 0].astype(int).tolist()
        reversal_vals = peaks[:, 1].tolist()
    else:
        reversal_idxs = []
        reversal_vals = []
    if len(valleys) > 0:
        reversal_idxs.extend(valleys[:, 0].astype(int).tolist())
        reversal_vals.extend(valleys[:, 1].tolist())

    reversal_idxs.extend([0, len(continuous_x)-1])
    reversal_vals.extend([0, continuous_x[-1]])

    idx = np.argsort(reversal_idxs)
    reversal_idxs = np.array(reversal_idxs)[idx]
    reversal_vals = np.array(reversal_vals)[idx]
    reversal_ts = t[reversal_idxs]

    x = np.interp(t, reversal_ts, reversal_vals)
    x = np.matrix(x)

    return x


def effective_wheel_radius(v=20):
    """
    Allow effective wheel radius to be a function of velocity.

    :param v: ignored for now
    :type v:

    :return: effective wheel radius, constant for now
    :rtype: float
    """
    return v


def torque(omega):
    """
    Convert throttle input to Torque. See Astrom and Murray 2008 Chapter 3.

    :param omega: throttle
    :type omega: float

    :return: motor torque
    :rtype: float
    """
    omega_m = parameters['omega_m']
    t_m = parameters['T_m']
    beta = parameters['beta']
    return t_m*(1 - beta*(omega / omega_m - 1)**2)


# pylint: disable-msg=too-many-locals
def step_forward(state_vals, disturbances, desired_v, dt):
    """
    One-step Euler integrator that takes the current state, disturbances, and desired velocity and calculates the subsequent state.

    :param state_vals: current state [position, velocity, road_angle]
    :type state_vals: np.matrix

    :param disturbances: current hill angle
    :type disturbances: np.matrix

    :param desired_velocity: current desired velocity
    :type desired_velocity: np.matrix

    :param dt: time step (seconds)
    :type dt: float

    :return: a tuple consisting of:
            - new_state: new state
            - u: control inputs
    :rtype: tuple -> (np.matrix, np.matrix)
    """
    p = state_vals[0, -1]
    v = state_vals[1, -1]
    theta = disturbances[2, -1]
    m = parameters['m']
    g = parameters['g']
    Cr = parameters['Cr']
    rho = parameters['rho']
    Cd = parameters['Cd']
    A = parameters['A']
    v_r = desired_v[0, -1]
    alpha_n = effective_wheel_radius(v)
    z = np.sum(desired_v[0, :] - state_vals[1, :])*dt
    k_p = parameters['k_p']
    k_i = parameters['k_i']
    u = k_p*(v_r-v) + k_i*z

    # rolling friction
    Fr = m*g*Cr*np.sign(v)
    # aerodynamic drag
    Fa = 0.5*rho*Cd*A*np.abs(v)*v
    # forces due to gravity
    Fg = m*g*np.sin(theta)
    # driving force
    Fd = alpha_n*u*torque(alpha_n*v)
    vdot = 1/m*(Fd - (Fr + Fa + Fg))
    new_state = np.matrix([[p + dt*v], [v + vdot*dt], [theta]])
    return new_state, np.matrix(u)


# disturbance
def hills(iterations, dt, factor):
    """
    Wrapper for creating a hill profile for the car to drive over that has an appropriate length and magnitude

    :param iterations: number of time points to simulate
    :type iterations: int

    :param dt: timestep of simulation in seconds
    :type dt: float

    :param factor: determines magnitude of the hills
    :type factor: int


    :return: hills, the output of the triangle function, a [1,M] matrix where M is the number of time points simulated
    :rtype: np.matrix
    """
    return triangle(iterations, dt)*0.3/factor


# desired velocity
def desired_velocity(n, factor):
    """
    Wrapper for defining the desired velocity as a matrix with size [1, M], where M is the number of time points to simulate
    See function "run" for how this function gets used.

    :param n: number of time points to simulate
    :type n: int

    :param factor: factor that determines the magnitude of the desired velocity
    :type factor: float

    :return: desired velocity as function of time, a [1,M] matrix, M is the number of time points to simulate
    :rtype: np.matrix
    """
    return np.matrix([2/factor]*n)


def run(timeseries_length=4, dt=0.01):
    """
    Simulate proportional integral control of a car attempting to maintain constant velocity while going up and down hills.
    This function is used for testing differentiation methods.

    See Astrom and Murray 2008 Chapter 3.

    :param timeseries_length: number of seconds to simulate
    :type timeseries_length: float

    :param dt: timestep in seconds
    :type dt: float

    :return: a tuple consisting of arrays of size [N, M], where M is the number of time steps.:
            - state_vals: state of the car, i.e. position and velocity as a function of time
            - disturbances: disturbances, ie. hills, that the car is subjected to
            - controls: control inputs applied by the car
    :rtype: tuple -> (np.array, np.array, np.array)
    """
    t = np.arange(0, timeseries_length, dt)
    iterations = len(t)

    # hills
    disturbances = np.matrix(np.zeros([3, iterations+1]))
    h = hills(iterations+1, dt, factor=0.5*timeseries_length/2)
    disturbances[2, :] = h[:, 0:disturbances.shape[1]]

    # controls
    controls = np.matrix([[0]])

    # initial condition
    state_vals = np.matrix([[0], [0], [0]])

    # desired vel
    v_r = desired_velocity(iterations, factor=0.5*iterations*dt/2)

    for i in range(1, iterations+1):
        new_state, u = step_forward(state_vals, disturbances[:, 0:i], v_r[:, 0:i], dt)
        state_vals = np.hstack((state_vals, new_state))
        controls = np.hstack((controls, u))

    return state_vals[0:2, 1:], disturbances[2, 1:], controls
