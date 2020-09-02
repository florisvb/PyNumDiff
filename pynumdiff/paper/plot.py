import figurefirst
import matplotlib.pyplot as plt
import numpy as np

def plot_velocity(dt, dxdt_hat, dxdt_truth, xlim=None, ylim=None, ax_dxdt=None, spines=['left','bottom'], 
                  ticklabels=['left', 'bottom'], params=None):
    
    t = np.arange(0,len(dxdt_truth))*dt
    if xlim is None:
        xlim = [0, t[-1]]

    if dxdt_hat is not None:
        ax_dxdt.plot(t, dxdt_hat, color='red', linewidth=1)
    ax_dxdt.plot(t, dxdt_truth, '--', color='black', linewidth=0.5)
    
    ax_dxdt.set_xlim(xlim[0], xlim[-1])
    if ylim is not None:
        ax_dxdt.set_ylim(ylim[0], ylim[-1])

    xticks = [0, 1, 2, 3, 4]
    yticks = [ylim[0], 0, ylim[-1]]
    spine_locations = {spine: 6 for spine in spines}
    figurefirst.mpl_functions.adjust_spines(ax_dxdt, spines, xticks=xticks, yticks=yticks, spine_locations=spine_locations, tick_length=2.5, linewidth=0.5)

    if 'left' not in ticklabels:
        ax_dxdt.set_yticklabels(['', '', '', '', ''])
    else:
        ax_dxdt.set_yticklabels([str(ylim[0]), '', str(ylim[-1])])
        ax_dxdt.set_ylabel('Velocity')
        ax_dxdt.yaxis.labelpad = -9

    if 'bottom' not in ticklabels:
        ax_dxdt.set_xticklabels(['', '', '', '', ''])
    else:
        ax_dxdt.set_xticklabels(['0', '', '', '', '4'])
        ax_dxdt.set_xlabel('Time')
        ax_dxdt.xaxis.labelpad = -3.5

    if params is not None:
        title = 'params: '
        for i, p in enumerate(params):
            title = title + str(p)
            if i+1 < len(params):
                title = title + ', '
        ax_dxdt.set_title(title)

    figurefirst.mpl_functions.set_fontsize(ax_dxdt.figure, 6)
    plt.tight_layout()

def plot_position(dt, x, x_hat, x_truth, xlim=None, ylim=None, ax_x=None, spines=['left','bottom'], 
                  ticklabels=['left', 'bottom'], params=None, simdt=0.0001):
    
    t = np.arange(0,len(x))*dt
    if xlim is None:
        xlim = [0, t[-1]]

    ax_x.set_xlim(xlim[0], xlim[-1])
    if ylim is not None:
        ax_x.set_ylim(ylim[0]-.1, ylim[-1]+.1)

    if x_hat is not None:
        ax_x.plot(t, x_hat, color='red', linewidth=1)
    ax_x.plot(t, x, 'o', color='blue', zorder=-100, markersize=0.5)

    if len(x_truth) > len(x):
        t_truth = np.arange(0,len(x_truth))*simdt
    else:
        t_truth = t
    ax_x.plot(t_truth, x_truth, '--', color='black', linewidth=0.5)
    #ax_x.set_ylabel('Position, m')
    #ax_x.set_xlabel('Time, s')

    xticks = [0, int(xlim[-1]/2), xlim[-1]]

    yticks = [ylim[0], ylim[0]+(ylim[-1]-ylim[0])/2,  ylim[-1]]
    spine_locations = {spine: 6 for spine in spines}
    figurefirst.mpl_functions.adjust_spines(ax_x, spines, xticks=xticks, yticks=yticks, spine_locations=spine_locations, tick_length=2.5, linewidth=0.5)

    if 'left' not in ticklabels:
        ax_x.set_yticklabels(['', '', '', '', ''])
    else:
        ax_x.set_yticklabels([str(ylim[0]), '', str(ylim[-1])])
        ax_x.set_ylabel('Position')
        ax_x.yaxis.labelpad = -5

    if 'bottom' not in ticklabels:
        ax_x.set_xticklabels(['', '', '', '', ''])
    else:
        ax_x.set_xticklabels(['0', '', '', '', '4'])
        ax_x.set_xlabel('Time')
        ax_x.xaxis.labelpad = -3.5

    if params is not None:
        title = 'params: '
        for i, p in enumerate(params):
            title = title + str(p)
            if i+1 < len(params):
                title = title + ', '
        ax_x.set_title(title)

    figurefirst.mpl_functions.set_fontsize(ax_x.figure, 6)
    plt.tight_layout()


