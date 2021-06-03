#!/usr/bin/env python
import os
import time
import numpy as np
import h5py
import atexit
import pandas
import matplotlib.pyplot as plt

def read_young_wind_data_to_pandas(filename):
    h = h5py.File(filename)

    index = np.arange(0, len(h['data'].value))
    d = {}
    for attribute in h['data'].value.dtype.fields.keys():
        d.setdefault(attribute, h['data'].value[attribute].flat)
    df = pandas.DataFrame(d, index=index)
    df = df[(df.T != 0).any()] # remove zero rows
    return df

def plot_wind(filename):
    df = read_young_wind_data_to_pandas(filename)

    fig = plt.figure()
    ax_x = fig.add_subplot(3,1,1)
    ax_y = fig.add_subplot(3,1,2)
    ax_z = fig.add_subplot(3,1,3)

    ax_x.plot(df.time, df.wind_x)
    ax_y.plot(df.time, df.wind_y)
    ax_z.plot(df.time, df.wind_z)

    ax_z.set_xlabel('time')
    ax_x.set_ylabel('wind x')
    ax_y.set_ylabel('wind y')
    ax_z.set_ylabel('wind z')

