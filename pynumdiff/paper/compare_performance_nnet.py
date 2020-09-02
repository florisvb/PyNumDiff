# external imports
import numpy as np
import pandas
import os
import time

import figurefirst

# local imports
import pynumdiff
simulate = pynumdiff.utils.simulate
evaluate = pynumdiff.utils.evaluate
import pynumdiff.paper.plot

import matplotlib.pyplot as plt

from IPython.display import display,SVG,Markdown


# simulation parameters

# noise is generated using np.random, e.g. np.random.normal, np.random.uniform, np.random.poisson
# noise_type and noise_parameters should be compatible with np.random functions 
noise_type = 'normal'
noise_parameters = [0, 0.5]
# time step and time series length
dt = 0.01
timeseries_length = 4 # secs



ylim = [-10, 10]

ticklabels = []

axis_names = {  ('finite_difference', 'first_order'): 'finite_difference',
                ('smooth_finite_difference', 'butterdiff'): 'butterdiff',
                ('smooth_finite_difference', 'splinediff'): 'splinediff',
                ('total_variation_regularization', 'acceleration'): 'tvr_accel',
                ('total_variation_regularization', 'jerk'): 'tvr_jerk',
                ('linear_model', 'spectraldiff'): 'spectraldiff',
                ('linear_model', 'polydiff'): 'polydiff',
                ('kalman_smooth', 'constant_acceleration'): 'kalman',
                ('nnet', 'quasinewton'): 'nnet',
              }

def simulate_all_problems(svg_template, 
                          dt, timeseries_length, noise_type, noise_parameters):
    high_res_sim = True


    layout = figurefirst.svg_to_axes.FigureLayout(svg_template,
                                                  autogenlayers=True,
                                                  make_mplfigures=True,
                                                  hide_layers=[])
    

    ##
    print('sine')
    ax = layout.axes[('position_data', 'sine')]
    x, x_truth, dxdt_truth, extras = simulate.sine(dt=dt,
                                                       timeseries_length=timeseries_length,
                                                       noise_type=noise_type, 
                                                       noise_parameters=noise_parameters)
    if high_res_sim:
        _, x_truth, _, _ = simulate.sine(dt=0.0001,
                                         timeseries_length=timeseries_length,
                                         noise_type=noise_type, 
                                         noise_parameters=noise_parameters)
    ylim = [0, 2]
    xlim = [0, timeseries_length]
    pynumdiff.paper.plot.plot_position(dt, x, None, x_truth, ax_x=ax, xlim=xlim, ylim=ylim, 
                                       ticklabels=['left', 'bottom'], spines=['left', 'bottom'])

    ##
    print('lorenz_x')
    ax = layout.axes[('position_data', 'lorenz_x')]
    x, x_truth, dxdt_truth, extras = simulate.lorenz_x(dt=dt,
                                                       timeseries_length=timeseries_length,
                                                       noise_type=noise_type, 
                                                       noise_parameters=noise_parameters)
    if high_res_sim:
        _, x_truth, _, _ = simulate.lorenz_x(dt=0.0001,
                                             timeseries_length=timeseries_length,
                                             noise_type=noise_type, 
                                             noise_parameters=noise_parameters)
    ylim = [-1, 1]
    xlim = [0, timeseries_length]
    pynumdiff.paper.plot.plot_position(dt, x, None, x_truth, ax_x=ax, xlim=xlim, ylim=ylim, 
                                       ticklabels=[], spines=[])

    ##
    print('triangle')
    ax = layout.axes[('position_data', 'triangle')]
    x, x_truth, dxdt_truth, extras = simulate.triangle(dt=dt,
                                                       timeseries_length=timeseries_length,
                                                       noise_type=noise_type, 
                                                       noise_parameters=noise_parameters)
    if high_res_sim:
        _, x_truth, _, _ = simulate.triangle(dt=0.0001,
                                             timeseries_length=timeseries_length,
                                             noise_type=noise_type, 
                                             noise_parameters=noise_parameters)
    ylim = [-1, 1]
    xlim = [0, timeseries_length]
    pynumdiff.paper.plot.plot_position(dt, x, None, x_truth, ax_x=ax, xlim=xlim, ylim=ylim, 
                                       ticklabels=[], spines=[])

    ##
    print('pop_dyn')
    ax = layout.axes[('position_data', 'pop_dyn')]
    x, x_truth, dxdt_truth, extras = simulate.pop_dyn(dt=dt,
                                                       timeseries_length=timeseries_length,
                                                       noise_type=noise_type, 
                                                       noise_parameters=noise_parameters)
    if high_res_sim:
        _, x_truth, _, _ = simulate.pop_dyn(dt=0.0001,
                                             timeseries_length=timeseries_length,
                                             noise_type=noise_type, 
                                             noise_parameters=noise_parameters)
    ylim = [0, 2]
    xlim = [0, timeseries_length]
    pynumdiff.paper.plot.plot_position(dt, x, None, x_truth, ax_x=ax, xlim=xlim, ylim=ylim, 
                                       ticklabels=[], spines=[])

    ##
    print('pi_control')
    ax = layout.axes[('position_data', 'pi_control')]
    x, x_truth, dxdt_truth, extras = simulate.pi_control(dt=dt,
                                                       timeseries_length=timeseries_length,
                                                       noise_type=noise_type, 
                                                       noise_parameters=noise_parameters)
    if high_res_sim:
        _, x_truth, _, _ = simulate.pi_control(dt=0.0001,
                                             timeseries_length=timeseries_length,
                                             noise_type=noise_type, 
                                             noise_parameters=noise_parameters)
    ylim = [0, 2]
    xlim = [0, timeseries_length]
    pynumdiff.paper.plot.plot_position(dt, x, None, x_truth, ax_x=ax, xlim=xlim, ylim=ylim, 
                                       ticklabels=[], spines=[])

    ##
    layout.append_figure_to_layer(layout.figures['position_data'], 'position_data', cleartarget=True)
    svg_name = 'nnet_comparison_dt_'+str(dt)+'_len_'+str(timeseries_length)+'_noisetype_'+noise_type+'_noiseparam0_'+str(noise_parameters[0])+'_noiseparam1_'+str(noise_parameters[1])+'.svg'
    svg_name = os.path.join(os.path.dirname(svg_template), svg_name)
    layout.write_svg(svg_name)

def evaluate_method_on_problem(method_parent, method, problem, layout, 
                               dt, timeseries_length, noise_type, noise_parameters):

    ax = layout.axes[(axis_names[(method_parent, method)], problem)]

    x, x_truth, dxdt_truth, extras = simulate.__dict__[problem](dt=dt,
                                                       timeseries_length=timeseries_length,
                                                       noise_type=noise_type, 
                                                       noise_parameters=noise_parameters)

    t0 = time.time()
    # neural net methods cannot be auto tuned
    #params, v = pynumdiff.optimize.__dict__[method_parent].__dict__[method](x, 
    #                                                                        dt, 
    #                                                                        dxdt_truth=dxdt_truth, 
    #                                                                        padding=0)

    if method == 'quasinewton':
        params = []
    elif method == 'adam':
        params = [300, 10, 0.001, 200]


    time_to_optimize = time.time() - t0

    ylim = [-10, 10]
    if problem == 'pop_dyn':
        ylim = [-1, 2.5]
    elif problem == 'pi_control':
        ylim = [-0.1, 1]


    xlim = [0, timeseries_length]

    t0 = time.time()
    x_hat, dxdt_hat = pynumdiff.__dict__[method_parent].__dict__[method](x, dt, params) 
                                                                         #options={'iterate': False})
    time_to_run = time.time() - t0

    pynumdiff.paper.plot.plot_velocity(dt, dxdt_hat, dxdt_truth, ax_dxdt=ax, xlim=xlim, ylim=ylim, 
                                       ticklabels=ticklabels, spines=[])

    # RMS error
    rms_err = np.sqrt( np.sum((dxdt_truth - dxdt_hat)**2) / len(dxdt_truth))
    tv = pynumdiff.utils.utility.total_variation(dxdt_hat)

    sample_data = {'noise_type': noise_type, 
                   'noise_param_0': noise_parameters[0],
                   'noise_param_1': noise_parameters[1],
                   'dt': dt,
                   'timeseries_length': timeseries_length,
                   'total_variation': tv,
                   'rms_error_vel': rms_err,
                   'method_parent': method_parent,
                   'method': method,
                   'problem': problem,
                   'time_to_optimize': time_to_optimize,
                   'time_to_run': time_to_run,
                   'max_error': np.max(np.abs((dxdt_truth - dxdt_hat))),
                   'median_error': np.median(np.abs((dxdt_truth - dxdt_hat))),
                   'std_error': np.std(np.abs((dxdt_truth - dxdt_hat))),
                   'params': params}

    ds = pandas.Series(sample_data)

    return ds

def evaluate_method_on_all_problems(method_parent, method, layout, 
                                    dt, timeseries_length, noise_type, noise_parameters):
    problems = ['sine', 'lorenz_x', 'triangle', 'pop_dyn', 'pi_control']
    df = pandas.DataFrame()
    
    for problem in problems:
        try:
            ds = evaluate_method_on_problem(method_parent, method, problem, layout, 
                                            dt, timeseries_length, noise_type, noise_parameters)
            df = df.append(ds, ignore_index=True)
        except:
            print('Failed: ', problem, method_parent, method, dt, timeseries_length, noise_type, noise_parameters)

    return df

def evaluate_all_methods_on_all_problems(svg_template, 
                                         dt, timeseries_length, noise_type, noise_parameters):
    print('simulating')
    simulate_all_problems(svg_template, 
                          dt, timeseries_length, noise_type, noise_parameters)
    print('simulation done')

    svg_name = 'nnet_comparison_dt_'+str(dt)+'_len_'+str(timeseries_length)+'_noisetype_'+noise_type+'_noiseparam0_'+str(noise_parameters[0])+'_noiseparam1_'+str(noise_parameters[1])+'.svg'
    svg_name = os.path.join(os.path.dirname(svg_template), svg_name)
    #layout.write_svg(svg_name)
    #layout = figurefirst.svg_to_axes.FigureLayout(svg_name,
    #                                              autogenlayers=False,
    #                                              make_mplfigures=False,
    #                                              hide_layers=[])
    
    method_parent_and_methods = [  ['nnet', 'quasinewton'],]

    dfs = []
    for method_parent_and_method in method_parent_and_methods:
        layout = figurefirst.svg_to_axes.FigureLayout(svg_name,
                                              autogenlayers=True,
                                              make_mplfigures=True,
                                              hide_layers=[])
        
        method_parent, method = method_parent_and_method
        df = evaluate_method_on_all_problems(method_parent, method, layout, 
                                    dt, timeseries_length, noise_type, noise_parameters)
        dfs.append(df)
        
        layout.append_figure_to_layer(layout.figures[axis_names[(method_parent, method)]], axis_names[(method_parent, method)], cleartarget=True)
        layout.write_svg(svg_name)

    df = pandas.concat(dfs, ignore_index=True)
    pickle_name = svg_name.replace('.svg', '.pickle')
    df.to_pickle(pickle_name)
    

# explore noise
if 1:
    noise_type = 'normal'
    noise_parameters_list = [[0, 0.001], [0, 0.0078125], [0, 0.015625], [0, 0.03125], [0, 0.0625], [0, 0.125], [0, 0.25], [0, 0.5], [0, 1]]
    # time step and time series length
    dts = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075]
    timeseries_length = 4 # secs

    for dt in dts:
        for noise_parameters in noise_parameters_list:
            evaluate_all_methods_on_all_problems('../../figures/combined/toy_problem_comparison.svg',
                                                 dt, timeseries_length, noise_type, noise_parameters)
