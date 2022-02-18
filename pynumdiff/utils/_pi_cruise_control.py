import numpy as _np

def run(timeseries_length=4, dt=0.01):
    """
    Simulate proportional integral control of a car attempting to maintain constant velocity while going up and down hills.
    This function is used for testing differentiation methods.

    This is a linear interpretation of something similar to what is described in Astrom and Murray 2008 Chapter 3.

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
    
    t = _np.arange(0, timeseries_length+dt, dt)

    # disturbance
    hills = _np.sin(2*_np.pi*t) + 0.3*_np.sin(4*2*_np.pi*t + 0.5) + 1.2*_np.sin(1.7*2*_np.pi*t + 0.5)
    hills = 0.01*hills

    # parameters
    mg = 10000 # mass*gravity
    fr = 0.9 # friction
    ki = 5/0.01*dt # integral control
    kp = 25/0.01*dt # proportional control
    vd = 0.5 # desired velocity

    A = _np.array([[1, dt, 0, 0, 0],
                   [0, 1, dt, 0, 0],
                   [0, -fr, 0, -mg, ki],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1]])

    B = _np.array([[0, 0],
                   [0, 0],
                   [0, kp],
                   [1, 0],
                   [0, 1]])

    x0 = _np.array([0, 0, 0, hills[0], 0]).reshape(A.shape[0], 1)

    # run simulation
    xs = [x0]
    us = [_np.array([0, 0]).reshape([2,1])]
    for i in range(1, len(hills)-1):
        u = _np.array([hills[i], vd - xs[-1][1,0]]).reshape([2,1])
        xnew = A@xs[-1] + B@u
        xs.append(xnew)
        us.append(u)

    xs = _np.hstack(xs)
    us = _np.hstack(us)

    if len(hills.shape) == 1:
        hills = _np.reshape(hills, [1, len(hills)])

    return xs, hills, us
