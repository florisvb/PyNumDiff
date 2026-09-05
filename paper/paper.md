---
title: 'PyNumDiff: Practical Numerical Differentiation for Noisy Data'
tags:
  - Python
  - numerical differentiation
  - time series
  - denoising
  - dynamics
  - signal processing
authors:
  - name: Pavel Komarov
    orcid: 0009-0007-7482-2807
    corresponding: true
    affiliation: 1
  - name: Floris van Breugel
    orcid: 0000-0001-6538-7179
    affiliation: 2
  - name: Maria Protogerou
    affiliation: 3
  - name: J. Nathan Kutz
    orcid: 0000-0002-6004-2275
    affiliation: 4
affiliations:
  - name: Department of Electrical and Computer Engineering, University of Washington, USA
    index: 1
  - name: Department of Mechanical Engineering, University of Nevada, Reno, USA
    index: 2
  - name: Department of Applied Mathematics, University of Washington, USA
    index: 3
  - name: Autodesk Research, London, UK
    index: 4
date: 1 April 2026
bibliography: paper.bib
---

# Summary

Derivatives of measured data are vital across science and engineering: identifying governing equations, designing controllers, and processing sensor streams alike. The textbook remedy, finite differencing, amplifies noise as $1/\Delta t$ and deteriorates rapidly as data grows noisier. Smoothing before differencing helps, but algorithm choice and settings substantially affect the result, and no single approach wins universally.

PyNumDiff is an open-source Python package consolidating a broad suite of numerical differentiation methods under a unified API, along with an optimization scheme to select hyperparameters even in the absence of ground truth. It implements seven algorithm families: (1) prefiltering followed by finite difference calculation, (2) iterated finite differencing, (3) polynomial fitting [@savitzky1964], (4) global (spectral) and local (wavelet, radial) basis function fitting, (5) total variation regularization [@chartrand2011numerical], (6) Kalman smoothing [@kalman1960; @rauch1965], and (7) local approximation with linear models. All PyNumDiff methods return estimates of the true signal and its derivative as a matched pair, `(x_hat, dxdt_hat)`. A companion survey [@komarov2025, Section 7] situates these methods in the literature, covers the theory behind each method, benchmarks all methods across test signals, and guides selection for different application scenarios. This paper describes the package's second generation, through the v0.3 release series.


# Statement of Need

Numerical differentiation has a diverse ecosystem of algorithms, each with different theoretical foundations and putative strengths. But there has been no consolidated home where methods can be easily swapped or tested head-to-head, and practitioners are left assembling solutions piecemeal from disparate packages [@vanBreugel2022].

PyNumDiff addresses this gap: Its unified interface lets users compare methods on the same data, exploit specialized capabilities (e.g. outlier robustness), and tune methods' hyperparameters against known ground truth or to a given signal bandlimit. Derivative estimation trades data fidelity against smoothness, so PyNumDiff uses dual-objective optimization, with a single scalar weight to steer the balance [@vanBreugel2020numerical]. A natural and growing application is SINDy [@brunton2016discovering], which discovers governing equations by regressing measured derivatives, making quality of the derivative estimates a direct determinant of model accuracy.


![Three simulated signals with additive noise (top) and their derivatives (bottom). Each panel contrasts two methods from different families, six of the seven in total, with a fixed color per method. Hyperparameters come from `pynumdiff.optimize` rather than hand-tuning, at the bandlimit noted with each derivative panel: 1 Hz for the sines and triangle, 2 Hz for the faster Lorenz signal. That bandlimit is an input to the optimizer, the sort of round figure a user reads off a power spectrum, rather than a claim about the signals, whose frequency content runs past it. The triangle's piecewise-constant derivative separates the methods most clearly, `tvrdiff` reproducing the steps where `butterdiff` rounds and rings at each corner.\label{fig:comparison}](methods_comparison.png)


# State of the Field

Relevant Python tools exist, but each covers only part of the space. `numpy.gradient` and `scipy.signal.savgol_filter` [@virtanen2020scipy] cover only a couple specific cases; `findiff` [@findiff] offers high-order finite difference stencils suited to clean simulation data, but not to noisy measurements; `pykalman` [@pykalman] does not provide state-space models to turn the filter into a differentiator; and total variation regularization [@chartrand2011numerical] was implemented with lagged-diffusivity iteration that approaches the convex minimizer rather than attaining it. Historically, practitioners have had to patch together packages, standalone scripts, and academic mathematics, with no shared API or principled way to compare results. The `derivative` package [@derivative_pkg] attempts to bring more algorithms under one roof but lacks `NaN` handling and hyperparameter optimization. No existing package spans PyNumDiff's seven method families with a consistent interface.

The original PyNumDiff publication [@vanBreugel2022] established the core method set and optimization framework. This second generation rewrites and consolidates it from the ground up. `kerneldiff`, `rtsdiff`, and `tvrdiff` each cover by parameter what had been separate functions, so the package presents a simplified interface. `rbfdiff`, `waveletdiff`, and `robustdiff` are new, bringing the set to thirteen, all organized into seven modules and callable with consistent keyword-argument signatures. A single optimizer serves every one of them. This generation also adds support for multidimensional data, as well as irregular sample spacing, missing observations, and circular domains where the underlying mathematics allows.


# Software Design

**Package design.** All differentiation methods share the call signature

```python
x_hat, dxdt_hat = method(x, dt_or_t, **hyperparams)
```

where `x` is a NumPy array [@harris2020array] of measurements, `dt_or_t` is either a scalar step size or an array of sample locations, and keyword arguments configure the method, making calls self-documenting. The v0.2 series preserves prior positional signatures behind deprecation warnings; v0.3 removes them, leaving only the keyword-argument interface.

**Architecture.** PyNumDiff is organized into seven differentiation modules plus shared `utils` and `optimize` in a flat structure. Where strong alternatives exist, PyNumDiff delegates rather than reimplements: SciPy [@virtanen2020scipy] provides spline fitting, Savitzky-Golay filtering, and signal processing routines; NumPy [@harris2020array] provides the FFT; PyWavelets [@lee2019pywavelets] provides the discrete wavelet transform for `waveletdiff`; CVXPY [@diamond2016cvxpy] handles convex optimization for `robustdiff`, `tvrdiff`, and `lineardiff`. The public `kalman_filter` and `rts_smooth` primitives let advanced users with known dynamics bypass `rtsdiff`'s constant-derivative model to build their own bespoke differentiator. `utils` houses common subroutines like estimation of integration constants and data scales, calculation of evaluation metrics, and generation of simulated noisy data. `optimize` includes not only cost minimization but a whole framework: per-method default search spaces, bounds with rounding rules, categorical handling, a caching layer, and a configurable objective that can work even in the absence of ground-truth.

**Method capabilities.** Every method handles multidimensional data via `axis`. Table 1 groups methods by their additional talents.

\begin{table}[!ht]
\centering
\begin{tabular}{@{}lp{0.70\linewidth}@{}}
\hline
\textbf{Capability} & \textbf{Methods} \tabularnewline
\hline
Variable step size & \raggedright \texttt{polydiff}, \texttt{splinediff}, \texttt{rbfdiff}, \texttt{rtsdiff}, \texttt{robustdiff}, \texttt{lineardiff} \tabularnewline
Missing data & \raggedright \texttt{polydiff}, \texttt{splinediff}, \texttt{rtsdiff}, \texttt{robustdiff}, \texttt{lineardiff} \tabularnewline
Outlier robustness & \raggedright \texttt{robustdiff}, \texttt{tvrdiff} \tabularnewline
Circular domain & \raggedright \texttt{rtsdiff} \tabularnewline
\hline
\end{tabular}
\caption{Methods by auxiliary capability.}
\end{table}

**Irregular and incomplete sampling.** Whether and how a method supports variable step size depends on the underlying model. For example, basis spline fits are indifferent to spacing, able to place knots with equal ease anywhere in the domain, but Kalman-based methods must compute a discrete-time transition by matrix exponential at each actual interval, while Butterworth filters inflexibly assume a constant sampling rate. Likewise, missing observations, given as NaN entries, can be excluded from fitting and imputed from the model in select cases, useful for real data where sensors may drop samples.

**Outlier robustness.** `robustdiff` replaces the quadratic Kalman cost with Huber loss [@huber1964] terms on both measurement and process residuals, following @aravkin2013, with CVXPY [@diamond2016cvxpy] as the optimization backend, operating over a sparse problem formulation to scale linearly with signal length. `tvrdiff` similarly applies Huber loss on data fidelity, with its total variation penalty on the derivative additionally promoting piecewise-smooth solutions.

**Circular and wrapped domains.** `rtsdiff` accepts `circular=True` for quantities like angles on a periodic domain. Innovation residuals are wrapped to $[-\pi, \pi]$ before each Kalman update via an `innovation_fn` hook, and `x_hat` is returned in the same range, avoiding the large spurious spikes naive smoothers produce when a signal crosses the $\pm\pi$ boundary.

**Hyperparameter optimization.** `pynumdiff.optimize` selects hyperparameters $\Phi$ by minimizing
$$L(\Phi) = \text{RMSE}\big(\textstyle\int\hat{\dot{x}}(\Phi) + c,\; x\big) + \gamma\,\text{TV}\big(\hat{\dot{x}}(\Phi)\big),$$
which requires no ground truth, because fidelity is measured by integrating the estimated derivative back against the measured signal [@vanBreugel2020numerical]. Root mean squared error in the first term can be (and is by default since v0.2.1) replaced with a robust variant, so it is less prone to bias in the presence of outliers. The smoothness weight $\gamma$ is derived internally from the data's sampling rate and user-provided `bandlimit`, the highest frequency of meaningful signal. Default search spaces are defined at the top of the `optimize` module, with opportunities to collapse to fewer search dimensions thoroughly explored, for speed. Minimization requires a continuous space, so discrete numbers like polynomial order and window size are handled by rounding, while truly categorical hyperparameters like kernel type are supported by pooling results across runs employing separate choices. Repeated evaluations are cached to avoid redundant work. A single call to `suggest_method` runs this search across nearly every method in the package and reports the best fit, the practical payoff of putting all behind one interface.

**Testing and continuous integration.** The test suite validates all methods against analytic functions with known derivatives, covering noiseless and noisy cases, scale equivariance, multidimensional application, and missing-data handling. Utilities and optimization are also tested for correct behavior, to verify functionality like robust estimation of data scale and the equivalence of parallel and serial search. The tests avoid tautology, never letting an implementation define its own expected result: Accuracy is asserted against empirically established error bounds, with the suite reporting any method that beats its bound by an order of magnitude so the bound can be ratcheted down. An optional `--plot` flag renders every case for visual inspection. GitHub Actions runs the suite on every push and pull request, and Coveralls tracks line coverage, currently 94%.


# Research Impact Statement

The first generation [@vanBreugel2022] of PyNumDiff is cited in over twenty publications and has been applied in experimental biology (flight kinematics from motion capture) and control engineering (observer design). The PySINDy project [@pysindy] maintains its own differentiation submodule overlapping substantially with PyNumDiff's capabilities, and integration discussions are ongoing.

The companion taxonomy paper [@komarov2025, Section 7] provides accuracy and bias results for every included method and derives recommendations, based on performance against simulated signals, using loss, search spaces, and other choices now encoded in PyNumDiff v0.3. The evidence is reproducible, with experiments runnable from Jupyter notebooks in PyNumDiff's repo, spanning sampling rate, noise distribution, noise scale, outlier contamination, and signal bandlimit. Additional tutorial notebooks cover basic usage, hyperparameter optimization, automatic method suggestion, and demonstrations of the outlier, multidimensional, and circular-domain capabilities described above. Full API documentation is published at [pynumdiff.readthedocs.io](https://pynumdiff.readthedocs.io/master/).


# AI Usage Disclosure

This paper was drafted with assistance from Claude Opus 5 (Anthropic), which also helped draft code revisions before and during review. All outputs were reviewed and heavily remolded by hand, and the authors take full responsibility for accuracy.


# Acknowledgements

The authors thank Yuying Liu and Bingni W. Brunton for their contributions to the original PyNumDiff package [@vanBreugel2022], and Sasha Aravkin for discussions on convex optimization techniques that informed robust differentiation methods. This work was supported by the NSF AI Institute in Dynamic Systems (grant number 2112085).


# References
