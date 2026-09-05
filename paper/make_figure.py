"""Generates paper/methods_comparison.png. Run from the repo root: python paper/make_figure.py"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pynumdiff.utils import simulate
from pynumdiff import polydiff, tvrdiff, rtsdiff, spectraldiff, butterdiff, lineardiff
from pynumdiff.optimize import optimize

# One color per method, fixed wherever it appears. Okabe-Ito, legible to colorblind readers.
STYLE = {spectraldiff:("spectraldiff","#CC79A7"), polydiff:("polydiff","#0072B2"), tvrdiff:("tvrdiff","#D55E00"),
         butterdiff:("butterdiff","#5D3A9B"), rtsdiff:("rtsdiff","#009E73"), lineardiff:("lineardiff","#E69F00")}

# Six of the seven families, paired so each panel contrasts two. Bandlimits are optimizer inputs, round numbers a
# user reads off a power spectrum; Lorenz is the faster signal. Comment out FITTED to re-run the search (minutes).
panels = [("Sum of sines", simulate.sine, [spectraldiff, polydiff], 1),
          ("Triangle wave", simulate.triangle, [tvrdiff, butterdiff], 1),
          ("Lorenz $x$", simulate.lorenz_x, [rtsdiff, lineardiff], 2)]
FITTED = {"spectraldiff": dict(cutoff_freq=0.0625, even_extension=True, pad_to_zero_dxdt=True),
          "polydiff": dict(stride=7, degree=2, window_size=41, kernel='gaussian'),
          "tvrdiff": dict(gamma=10.6015625, huberM=6.0, order=1),
          "butterdiff": dict(cutoff_freq=0.05, num_iterations=1, filter_order=2),
          "rtsdiff": dict(log_qr_ratio=3.81328125, order=1, forwardbackward=False),
          "lineardiff": dict(gamma=0.00011125, window_size=53, order=2, kernel='friedrichs')}
#FITTED = {}

def main():
    fig, axes = plt.subplots(2, 3, figsize=(13, 5.4), sharex=True,
                             gridspec_kw={'height_ratios':[1, 1.6], 'hspace':0.13, 'wspace':0.2})
    for j, (sname, sim, methods, bandlimit) in enumerate(panels):
        x, x_truth, dxdt_truth = sim(duration=4, dt=0.01, noise_parameters=(0, 0.2), random_seed=3)
        t = np.arange(len(x))*0.01
        top, bot = axes[0, j], axes[1, j]

        top.plot(t, x, '.', color='0.72', markersize=1.6, label="noisy data")
        top.plot(t, x_truth, '-', color='black', linewidth=1.1, label="true $x$")
        top.set_title(sname, fontsize=11, pad=6)
        bot.plot(t, dxdt_truth, '-', color='0.55', linewidth=2.6, label="true $\\dot{x}$", zorder=1)

        for method in methods:
            name, color = STYLE[method]
            # Order 1 represents the triangle's piecewise-constant derivative exactly; without it tvrdiff goes spiky.
            ssu = {'order':{1, 2, 3}} if (method is tvrdiff and sim is simulate.triangle) else {}
            params = FITTED.get(name) or optimize(method, x, 0.01, bandlimit=bandlimit, search_space_updates=ssu)[0]
            print(f"  {sname:<14} {name:<13} {params}", flush=True)
            bot.plot(t, method(x, 0.01, **params)[1], '-', color=color, linewidth=1.1, label=name, zorder=2)
        bot.set_xlabel("time (s)", fontsize=10)

        for ax, pad in ((top, 0.17 if j == 0 else 0.04), (bot, 0.28)): # only the headroom each legend needs
            lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + pad*(hi - lo))
            ax.tick_params(labelsize=8)
            for s in ('top', 'right'): ax.spines[s].set_visible(False)
        bot.legend(fontsize=8, frameon=False, loc='upper center', ncol=3, handlelength=1.6, columnspacing=1.1,
                   borderpad=0.1, title=f"optimized at bandlimit {bandlimit} Hz", title_fontsize=8)

    axes[0, 0].set_ylabel("$x$", fontsize=11)
    axes[1, 0].set_ylabel("$dx/dt$", fontsize=11)
    axes[0, 0].legend(fontsize=8, frameon=False, loc='upper center', ncol=2, handlelength=1.6, borderpad=0.1)
    fig.savefig("paper/methods_comparison.png", dpi=200, bbox_inches='tight')
    print("wrote paper/methods_comparison.png")

if __name__ == "__main__": # spawned workers re-import this module, so nothing heavy may run at import time
    main()
