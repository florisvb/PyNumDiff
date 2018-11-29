import numpy as np
import matplotlib.pyplot as plt
from pynumdiff.utils import utility as utility

parameters = {'omega_m': 420, # rad / sec
              'T_m': 190, # M,
              'beta': 0.4, #
              'Cr': 0.01, 
              'Cd': 0.32,
              'A': 2.4,
              'g': 9.8, # m/s^2
              'm': 3000, # kg
              'rho': 1.3, # kg/m^3
              'v_r': 30, # m/s 
              'k_p': 10,
              'k_i': 4,
             }

def triangle(timeseries_length=500, dt=0.01):
    t = np.arange(0, timeseries_length*dt, dt)
    continuous_x = np.sin(t*np.sqrt(t))

    #return np.matrix(continuous_x)

    # find peaks and valleys
    peaks, valleys = utility.peakdet(continuous_x, 0.1)

    # organize peaks and valleys
    if len(peaks) > 0:
        reversal_idxs = peaks[:,0].astype(int).tolist()
        reversal_vals = peaks[:,1].tolist()
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

def effective_wheel_radius(v):
    return 5

def Torque(omega):
    omega_m = parameters['omega_m']
    T_m = parameters['T_m']
    beta = parameters['beta']
    return T_m*(1 - beta*(omega / omega_m - 1)**2)

def step_forward(state_vals, disturbances, desired_velocity, dt):
    # state_vals = [position, velocity, road_angle]
    
    p = state_vals[0,-1]
    v = state_vals[1,-1]
    theta = disturbances[2,-1]
    
    m = parameters['m']
    g = parameters['g']
    Cr = parameters['Cr']
    
    rho = parameters['rho']
    Cd = parameters['Cd']
    A = parameters['A']
    
    v_r = desired_velocity[0,-1] #parameters['v_r']
    
    alpha_n = effective_wheel_radius(v)
    z = np.sum(desired_velocity[0,:] - state_vals[1,:])*dt
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
    Fd = alpha_n*u*Torque(alpha_n*v)
    
    vdot = 1/m*(Fd - (Fr + Fa + Fg))
    
    new_state = np.matrix([[p + dt*v], [v + vdot*dt], [theta]])
    
    return new_state

# disturbance
def hills(n, dt, factor):
    #t = np.linspace(0,n,n)
    #y = 1*np.sin(0*t*200/np.max(t)) + 5*np.sin(t*100/np.max(t)) + 100*np.sin(t*20/np.max(t)) + 10*np.sin(t*7/np.max(t)) 
    #return y*np.pi/180.*1e-2
    return triangle(n, dt)*0.1/factor

# desired velocity
def desired_velocity(n, factor):
    return np.matrix([2/factor]*n)


def run(iterations=400, dt=0.01):

    # hills
    disturbances = np.matrix(np.zeros([3, iterations+1]))
    disturbances[2,:] = hills(iterations+1, dt=0.01, factor=1.8*iterations*dt/2)

    # initial condition
    state_vals = np.matrix([[0], [0], [0]])

    # desired vel
    v_r = desired_velocity(iterations, factor=1.8*iterations*dt/2)

    for i in range(1, iterations+1):
        new_state = step_forward(state_vals, disturbances[:,0:i], v_r[:,0:i], dt)
        state_vals = np.hstack((state_vals, new_state))

    return state_vals[0:2,1:], disturbances[2,1:]