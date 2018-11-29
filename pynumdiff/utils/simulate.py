import numpy as _np

# local imports
from pynumdiff.utils import utility as _utility
from pynumdiff.utils import __pi_cruise_control__ as __pi_cruise_control__
_finite_difference = _utility.finite_difference

def __add_noise__(x, noise_type, noise_parameters, random_seed):
    _np.random.seed(1)
    timeseries_length = _np.max(x.shape)
    noise = _np.random.__getattribute__(noise_type)(noise_parameters[0], noise_parameters[1], timeseries_length)
    return x + noise

def sine(timeseries_length=500, noise_type='normal', noise_parameters=[0, 0.5], random_seed=1, dt=0.01):

    # Parameters
    ############
    f1 = 1
    f2 = 1.7
    y_offset = 10

    t = _np.arange(0, timeseries_length*dt, dt)
    x = (0.5*_np.sin(t*2*_np.pi*f1) + 0.5*_np.sin(t*2*_np.pi*f2)) + y_offset
    dxdt = 0.5*_np.cos(t*2*_np.pi*f1)*2*_np.pi*f1 + 0.5*_np.cos(t*2*_np.pi*f2)*2*_np.pi*f2
    #actual_vals = _finite_difference(_np.matrix(x), params=None, dt=dt)
    actual_vals = _np.matrix(_np.vstack((x, dxdt)))


    noisy_x = __add_noise__(x, noise_type, noise_parameters, random_seed)

    noisy_measurements = _np.matrix(noisy_x)
    extra_measurements = _np.matrix([])

    noisy_pos = _np.ravel(noisy_measurements)
    pos = _np.ravel(actual_vals[0,:])
    vel = _np.ravel(actual_vals[1,:])
    extras = None
    return noisy_pos, pos, vel, extras 

def triangle(timeseries_length=500, noise_type='normal', noise_parameters=[0, 0.5], random_seed=1, dt=0.01):
    t = _np.arange(0, timeseries_length*dt, dt)
    continuous_x = _np.sin(t*t)

    # find peaks and valleys
    peaks, valleys = _utility.peakdet(continuous_x, 0.1)

    # organize peaks and valleys
    if len(peaks) > 0:
        reversal_idxs = peaks[:,0].astype(int).tolist()
        reversal_idxs.extend(valleys[:, 0].astype(int).tolist())
        reversal_idxs.extend([0, len(continuous_x)-1])

        reversal_vals = peaks[:,1].tolist()
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
    x = _np.matrix(x)

    smooth_x, dxdt = _finite_difference(x, dt=dt)
    dxdt = _np.matrix(dxdt)

    noisy_x = __add_noise__(x, noise_type, noise_parameters, random_seed)

    actual_vals = _np.matrix(_np.vstack((x, dxdt[1,:])))
    noisy_measurements = _np.matrix(noisy_x)
    extra_measurements = _np.matrix([])

    noisy_pos = _np.ravel(noisy_measurements)
    pos = _np.ravel(actual_vals[0,:])
    vel = _np.ravel(actual_vals[1,:])
    extras = None
    return noisy_pos, pos, vel, extras 

def pop_dyn(timeseries_length=500, noise_type='normal', noise_parameters=[0, 0.5], random_seed=1, dt=0.01):
    # http://www.biologydiscussion.com/population/population-growth/population-growth-curves-ecology/51854

    t = _np.arange(0, timeseries_length*dt, dt)
    K = 2 # carrying capacity
    r = 4 # biotic potential

    x = [0.1] # population
    dxdt = [r*x[-1]*(1-x[-1]/K)]
    for _t_ in t[1:]:
        x.append(x[-1] + dt*dxdt[-1])
        dxdt.append(r*x[-1]*(1-x[-1]/K)) 

    x = _np.matrix(x)
    dxdt = _np.matrix(dxdt)

    noisy_x = __add_noise__(x, noise_type, noise_parameters, random_seed)

    actual_vals = _np.matrix(_np.vstack((x, dxdt)))
    noisy_measurements = _np.matrix(noisy_x)
    extra_measurements = _np.matrix([])

    noisy_pos = _np.ravel(noisy_measurements)
    pos = _np.ravel(actual_vals[0,:])
    vel = _np.ravel(actual_vals[1,:])
    extras = None
    return noisy_pos, pos, vel, extras 

def linear_autonomous(timeseries_length=500, noise_type='normal', noise_parameters=[0, 0.5], random_seed=1, dt=0.01):
    A = _np.matrix([[1, dt], [0.005, 0.97]])
    x0 = _np.matrix([[0], [5]])
    xs = x0
    for i in range(timeseries_length):
        x = A*xs[:,-1]
        xs = _np.hstack((xs, x))

    x = xs[0,:]
    x += _np.min(x)
    x /= _np.max(x)
    x *= 2

    smooth_x, ffd = _finite_difference(x, dt)
    dxdt = ffd[1,:]

    noisy_x = __add_noise__(x, noise_type, noise_parameters, random_seed)

    actual_vals = _np.matrix(_np.vstack((x, dxdt)))
    noisy_measurements = _np.matrix(noisy_x)
    extra_measurements = None

    noisy_pos = _np.ravel(noisy_measurements[1:])
    pos = _np.ravel(actual_vals[0,1:])
    vel = _np.ravel(actual_vals[1,1:])
    extras = None
    return noisy_pos, pos, vel, extras 

def pi_control(timeseries_length=500, noise_type='normal', noise_parameters=[0, 0.5], random_seed=1, dt=0.01):
    actual_vals, extra_measurements = __pi_cruise_control__.run(timeseries_length, dt)
    x = actual_vals[0,:]
    dxdt = actual_vals[1,:]

    noisy_x = __add_noise__(x, noise_type, noise_parameters, random_seed)

    actual_vals = _np.matrix(_np.vstack((x, dxdt)))
    noisy_measurements = _np.matrix(noisy_x)

    noisy_pos = _np.ravel(noisy_measurements)
    pos = _np.ravel(actual_vals[0,:])
    vel = _np.ravel(actual_vals[1,:])
    extras = None
    return noisy_pos, pos, vel, extra_measurements 

def lorenz_x(timeseries_length=500, noise_type='normal', noise_parameters=[0, 0.5], random_seed=1, dt=0.01):

    sigma = 10
    beta = 8/3
    rho = 28

    x = 5
    y = 1
    z = 3
    xyz = _np.matrix([[x], [y], [z]])
    xyz_dot = None

    for _t_ in _np.arange(0, timeseries_length*dt, dt):
        x, y, z = _np.ravel(xyz[:,-1])

        xdot = sigma*(y-x)
        ydot = x*(rho-z)-y
        zdot = x*y - beta*z

        new_xyz_dot = _np.matrix([[xdot], [ydot], [zdot]])
        if xyz_dot is None:
            xyz_dot = new_xyz_dot
        else:
            xyz_dot = _np.hstack((xyz_dot, new_xyz_dot))

        new_xyz = xyz[:,-1] + dt*new_xyz_dot
        xyz = _np.hstack((xyz, new_xyz))

    x = xyz[0,0:-1] / 20.
    dxdt = xyz_dot[0,:] / 20.

    noisy_x = __add_noise__(x, noise_type, noise_parameters, random_seed)

    actual_vals = _np.matrix(_np.vstack((x, dxdt)))
    noisy_measurements = _np.matrix(noisy_x)
    extra_measurements = _np.matrix([])

    noisy_pos = _np.ravel(noisy_measurements)
    pos = _np.ravel(actual_vals[0,:])
    vel = _np.ravel(actual_vals[1,:])
    extras = None
    return noisy_pos, pos, vel, extra_measurements