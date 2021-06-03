import os
import sys
import matplotlib.pyplot as plt
import numpy as np

import figurefirst as fifi
import scipy.fftpack

import pynumdiff

import pickle
import time

from multiprocessing import Pool
import multiprocessing

PADDING = 'auto'

def get_data(problem, noise, dt, timeseries_length, simdt=0.0001):
    r = pynumdiff.utils.simulate.__dict__[problem](timeseries_length, noise_parameters=[0, noise], dt=dt, simdt=simdt)
    x, x_truth, dxdt_truth, _ = r
    t = np.linspace(0, timeseries_length, len(x))
    #dt = np.mean(np.diff(t))
    return x, x_truth, dxdt_truth, t, dt

def get_data_sine(problem, freq, noise, dt, timeseries_length, magnitude=1):
    if type(freq) != list:
        freq = [freq]
    r = pynumdiff.utils.simulate.__dict__[problem](timeseries_length, noise_parameters=[0, noise], dt=dt, frequencies=freq, magnitude=magnitude)
    x, x_truth, dxdt_truth, _ = r
    t = np.linspace(0, timeseries_length, len(x))
    #dt = np.mean(np.diff(t))
    return x, x_truth, dxdt_truth, t, dt

def get_rmse_errcorr_for_params(x, x_truth, dxdt_truth, dt, method_parent, method, params):
    
    #params, v = pynumdiff.optimize.__dict__[method_parent].__dict__[method](wind_speed, dt, tvgamma=gamma_general)
    x_smooth, xdot_smooth = pynumdiff.__dict__[method_parent].__dict__[method](x, dt, params)
    
    if np.max(np.abs(xdot_smooth - dxdt_truth)) > 10000:
        rmse = 10000
        errcorr = 1000
    else:
        rmse = pynumdiff.utils.evaluate.rmse(xdot_smooth, dxdt_truth, padding=PADDING)
        errcorr = pynumdiff.utils.evaluate.error_correlation(xdot_smooth, dxdt_truth, padding=PADDING)
    
    tv = pynumdiff.utils.utility.total_variation(xdot_smooth)
    return rmse, errcorr, tv



def get_params_for_method(method, method_parent):
    if method_parent == 'linear_model' and method == 'savgoldiff':
        params_list = []
        for p1 in range(1,12):
            for p2 in np.unique(np.logspace(0,3,33).astype(int)):
                if p1 >= p2 or p2 < 3:
                    continue
                for p3 in np.unique(np.logspace(0,3,23).astype(int)):
                    params_list.append([p1,p2,p3])
    if method_parent == 'kalman_smooth' and method == 'constant_acceleration':
        params_list = []
        for p1 in np.logspace(-8, 8, 75):
            for p2 in np.logspace(-8, 8, 75):
                params_list.append([p1,p2])
    if method_parent == 'smooth_finite_difference' and method == 'butterdiff':
        params_list = []
        for p1 in range(2,12):
            for p2 in np.logspace(-8, -0.1, 500):
                params_list.append([p1,p2])
    if method_parent == 'total_variation_regularization' and method == 'jerk':
        params_list = []
        for p1 in np.logspace(-8, 8, 500):
                params_list.append([p1])
                
                
    return params_list



def get_pareto_plot_data_for_params_list(inputs):
    x, x_truth, dxdt_truth, dt, method_parent, method, params = inputs
    try:
        r, e, tv = get_rmse_errcorr_for_params(x, x_truth, dxdt_truth, dt, method_parent, method, params)
    except:
        r = None
        e = None
        tv = None
    return r, e, tv


def get_pareto_plot_data(x, x_truth, dxdt_truth, dt, method, method_parent, gamma_range, num_gammas=10, padding=PADDING):
    params_list = get_params_for_method(method, method_parent)
    
    # parallel params computations
    inputs = []
    for p in params_list:
        i = [x, x_truth, dxdt_truth, dt, method_parent, method, p]
        inputs.append(i)


    pool = Pool(20)
    result = pool.map(get_pareto_plot_data_for_params_list, inputs)
    pool.close()
    pool.join()

    rmses = np.vstack(result)[:,0].tolist()
    errcorrs = np.vstack(result)[:,1].tolist()
    tvs = np.vstack(result)[:,2].tolist()
        
    rmses_gamma = []
    errcorrs_gamma = []
    params_gamma = []
    successful_gammas = []
    tvs_gamma = []
    
    gammas = np.exp(np.linspace(np.log(gamma_range[0]), np.log(gamma_range[1]), num_gammas)) 
    for gamma in gammas:
        #try:
        params, v = pynumdiff.optimize.__dict__[method_parent].__dict__[method](x, dt, tvgamma=gamma, padding=padding)
        x_smooth, xdot_smooth = pynumdiff.__dict__[method_parent].__dict__[method](x, dt, params)
        r, e, tv = get_rmse_errcorr_for_params(x, x_truth, dxdt_truth, dt, method_parent, method, params)
        tvs_gamma.append(tv)
        rmses_gamma.append(r)
        errcorrs_gamma.append(e)
        successful_gammas.append(gamma)
        params_gamma.append(params)
        #except:
        #    print('FAILED', method, method_parent, gamma)
            
    rmses = np.array(rmses)
    errcorrs = np.array(errcorrs)
    rmses_gamma = np.array(rmses_gamma)
    errcorrs_gamma = np.array(errcorrs_gamma)
    tvs_gamma = np.array(tvs_gamma)
    
    return rmses, errcorrs, tvs, rmses_gamma, errcorrs_gamma, tvs_gamma, params_gamma, successful_gammas


def do_pareto_calcs_for_method(method, method_parent, 
                               problem, noise, dt, timeseries_length, 
                               x, x_truth, dxdt_truth, t, num_gammas=20, frequencies=None,
                               directory='pareto_data', padding=PADDING):
    gamma_range = [1e-4, 1e4]

    
    rmses, errcorrs, tvs, rmses_gamma, errcorrs_gamma, tvs_gamma, params_gamma, successful_gammas = get_pareto_plot_data(x, x_truth, dxdt_truth, dt, 
                                                                        method, method_parent, 
                                                                        gamma_range, num_gammas=num_gammas, padding=padding)

    metadata = {'noise': noise,
                'dt': dt,
                'timeseries_length': timeseries_length,
                'problem': problem,
                'method': method,
                'method_parent': method_parent,
                'params': params_gamma,
                'gammas': successful_gammas,
                'padding': padding}
    
    data = {   'rmses': rmses,
               'errcorrs': errcorrs,
               'rmses_gamma': rmses_gamma,
               'errcorrs_gamma': errcorrs_gamma,
               'metadata': metadata}
    

    fname = directory + '/pareto_data_' + method + '_' + method_parent + '_' + problem + '_' + str(noise) + '_' + str(dt) + '_' + str(timeseries_length) + '_' + str(padding) + '.pickle'
    f = open(fname, 'wb')
    pickle.dump(data, f)
    f.close()

def do_pareto_calcs_for_sine(method, method_parent, 
                               problem, freq, noise, dt, timeseries_length, 
                               x, x_truth, dxdt_truth, t, num_gammas=20,
                               directory='pareto_sine_freq_data_varpadding', padding=PADDING):
    gamma_range = [1e-4, 1e4]

    rmses, errcorrs, tvs, rmses_gamma, errcorrs_gamma, tvs_gamma, params_gamma, successful_gammas = get_pareto_plot_data(x, x_truth, dxdt_truth, dt, 
                                                                        method, method_parent, 
                                                                        gamma_range, num_gammas=20, padding=padding)

    metadata = {'noise': noise,
                'dt': dt,
                'timeseries_length': timeseries_length,
                'problem': problem,
                'method': method,
                'method_parent': method_parent,
                'params': params_gamma,
                'gammas': successful_gammas,
                'freq': freq,
                'padding': padding}
    
    data = {   'rmses': rmses,
               'errcorrs': errcorrs,
               'rmses_gamma': rmses_gamma,
               'errcorrs_gamma': errcorrs_gamma,
               'metadata': metadata}
    

    fname = directory + '/pareto_data_' + method + '_' + method_parent + '_' + problem + '_' + str(freq) + '_' + str(noise) + '_' + str(dt) + '_' + str(timeseries_length) + '_' + str(padding) + '.pickle'
    f = open(fname, 'wb')
    pickle.dump(data, f)
    f.close()

def get_filenames(path, contains, does_not_contain=['~', '.pyc']):
    cmd = 'ls ' + '"' + path + '"'
    ls = os.popen(cmd).read()
    all_filelist = ls.split('\n')
    try:
        all_filelist.remove('')
    except:
        pass
    filelist = []
    for i, filename in enumerate(all_filelist):
        if contains in filename:
            fileok = True
            for nc in does_not_contain:
                if nc in filename:
                    fileok = False
            if fileok:
                filelist.append( os.path.join(path, filename) )
    return filelist



def run_other_analysis(method_parent_pairs):
    problems = ['pi_control', 'sine', 'triangle', 'lorenz_x', 'pop_dyn']

    

    noises = [0.01, 0.1] #[0.001, 0.01, 0.1, 0.5]
    dts = [0.001, 0.01] #[0.001, 0.01, 0.1]
    timeseries_lengths = [4]#, 8, 16]

    for problem in problems:
        print(problem)
        for noise in noises:
            for dt in dts:
                for timeseries_length in timeseries_lengths:
                    x, x_truth, dxdt_truth, t, dt = get_data(problem, noise, dt, timeseries_length)

                    for method_parent_pair in method_parent_pairs:
                        print(method_parent_pair)
                        method, method_parent = method_parent_pair
                        
                        fname = 'pareto_data/pareto_data_' + method + '_' + method_parent + '_' + problem + '_' + str(noise) + '_' + str(dt) + '_' + str(timeseries_length) + '_' + str(padding) + '.pickle'
                        filenames_done = get_filenames('pareto_data/', '.pickle')
                        if fname in filenames_done:
                            continue
                        
                        method, method_parent = method_parent_pair
                        do_pareto_calcs_for_method(method, method_parent, 
                                                   problem, noise, dt, timeseries_length, 
                                                   x, x_truth, dxdt_truth, t)

def run_pareto_analysis_on_specific(noise, dt, timeseries_length, problem, method, method_parent, simdt=0.0001, frequencies=None, read_existing=True, num_gammas=40, padding='auto', magnitude=1):
    fname = 'pareto_data/pareto_data_' + method + '_' + method_parent + '_' + problem + '_' + str(noise) + '_' + str(dt) + '_' + str(timeseries_length) + '_' + str(padding) + '.pickle'
    print(fname)
    filenames_done = get_filenames('pareto_data/', '.pickle')
    if read_existing:
        if fname in filenames_done:
            print('found file: ', fname)
            return fname

    if problem != 'sine':
        x, x_truth, dxdt_truth, t, dt = get_data(problem, noise, dt, timeseries_length, simdt=simdt)
    else:
        x, x_truth, dxdt_truth, t, dt = get_data_sine(problem, frequencies, noise, dt, timeseries_length, magnitude=magnitude)
                                
    
                                                
    print('running experiment')
    do_pareto_calcs_for_method(method, method_parent, problem, noise, dt, timeseries_length, x, x_truth, dxdt_truth, t, num_gammas=num_gammas, frequencies=frequencies)

    return fname

def run_pareto_analysis_on_specific_sine(noise, dt, timeseries_length, problem, freq, method, method_parent, simdt=0.0001, read_existing=True, num_gammas=40, padding='auto', magnitude=1):
    fname = 'pareto_specific_sine_freq_data_varpadding/pareto_data_' + method + '_' + method_parent + '_' + problem + '_' + str(freq) + '_' + str(noise) + '_' + str(dt) + '_' + str(timeseries_length) + '_' + str(padding) + '.pickle'
    print(fname)
    filenames_done = get_filenames('pareto_specific_sine_freq_data_varpadding/', '.pickle')
    if read_existing:
        if fname in filenames_done:
            return fname

    if problem != 'sine':
        print('This is for sine!')
        return None
    else:
        x, x_truth, dxdt_truth, t, dt = get_data_sine(problem, [freq], noise, dt, timeseries_length, magnitude=magnitude)
                                
    
                                                

    do_pareto_calcs_for_sine(method, method_parent, 
                               problem, freq, noise, dt, timeseries_length, 
                               x, x_truth, dxdt_truth, t, 
                               num_gammas=num_gammas, directory='pareto_specific_sine_freq_data_varpadding', padding=padding)

    return fname


def run_sine_analysis():
    problems = ['sine'] #['pi_control', 'sine', 'triangle'] #['lorenz_x', 'pop_dyn']

    method_parent_pairs = [['savgoldiff', 'linear_model'],
                           #['constant_acceleration', 'kalman_smooth'],
                           #['butterdiff', 'smooth_finite_difference'],
                           #['jerk', 'total_variation_regularization'],
                          ]

    noises = [0.01, 0.1, 0.5]
    dts = [0.1, 0.01, 0.001]
    timeseries_lengths = [0.02, 500] #[4]#, 8, 16]
    padding = 'auto'

    frequencies = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]

    for method_parent_pair in method_parent_pairs:
        method, method_parent = method_parent_pair
        for problem in problems:
            print(problem)
            for freq in frequencies:
                for noise in noises:
                    for dt in dts:
                        for timeseries_length in timeseries_lengths:
                            print(problem, method, method_parent, freq, noise, dt, timeseries_length)
                            if timeseries_length < 1/freq:
                                continue
                            if dt > 1/freq/2:
                                continue
                            if timeseries_length >= 100 and dt < 0.01:
                                continue
                            if 1:
                                x, x_truth, dxdt_truth, t, dt = get_data_sine(problem, freq, noise, dt, timeseries_length)

                                

                                    
                                    
                                fname = 'pareto_sine_freq_data_varpadding/pareto_data_' + method + '_' + method_parent + '_' + problem + '_' + str(freq) + '_' + str(noise) + '_' + str(dt) + '_' + str(timeseries_length) + '_' + str(padding) + '.pickle'
                                filenames_done = get_filenames('pareto_sine_freq_data_varpadding/', '.pickle')
                                if fname in filenames_done:
                                    continue
                                    
                                method, method_parent = method_parent_pair
                                do_pareto_calcs_for_sine(method, method_parent, 
                                                               problem, freq, noise, dt, timeseries_length, 
                                                               x, x_truth, dxdt_truth, t, padding=padding)


if __name__ == '__main__':

    if 0:
        run_other_analysis([['savgoldiff', 'linear_model']])
        run_other_analysis([['butterdiff', 'smooth_finite_difference']])
        run_other_analysis([['constant_acceleration', 'kalman_smooth']])
        run_other_analysis([['jerk', 'total_variation_regularization']])
        
    if 1:
        run_sine_analysis()
