---
title: 'PyNumDiff 2.0: Practical Numerical Differentiation for Noisy Data'
tags:
  - Python
  - numerical differentiation
  - time series
  - denoising
  - dynamics
  - signal processing
authors:
  - name: Pavel Komarov
    corresponding: true
    affiliation: 1
  - name: Floris van Breugel
    affiliation: 2
  - name: J. Nathan Kutz
    affiliation: 3
affiliations:
  - name: Department of Electrical and Computer Engineering, University of Washington, USA
    index: 1
  - name: Department of Mechanical Engineering, University of Nevada, Reno, USA
    index: 2
  - name: Autodesk Research, London, UK
    index: 3
date: 1 April 2026
bibliography: paper.bib
---

# Summary

Derivatives of measured data are a prerequisite across science and engineering: identifying governing equations, designing controllers, and processing sensor streams alike. The textbook remedy, finite differencing, amplifies noise as $1/\Delta t$ and deteriorates rapidly as data grows noisier or more finely sampled. Smoothing before differencing helps, but algorithm choice and tuning substantially affect the result, and no single approach wins universally.

PyNumDiff is an open-source Python package consolidating a broad suite of numerical differentiation methods under a unified API. Seven algorithm families are implemented: (1) prefiltering followed by finite difference calculation; (2) iterated finite differencing; (3) polynomial fitting [@savitzky1964]; (4) spectral and radial basis function fitting; (5) total variation regularization [@chartrand2011numerical]; (6) Kalman smoothing [@kalman1960; @rauch1965]; and (7) local approximation with linear models. @ahnert2007 provides a useful taxonomy of these families, distinguishing local methods (which estimate derivatives from a surrounding window) from global methods that fit the entire signal at once. All PyNumDiff methods return a matched pair `(x_hat, dxdt_hat)`. A companion paper [@komarov2025] benchmarks all methods across test signals and guides selection for different application scenarios.


# Statement of Need

Derivative estimation from noisy measurements arises throughout experimental science, data-driven modeling, system identification, and control. The field offers a diverse ecosystem of specialized algorithms, each with different strengths in outlier robustness, computational cost, irregular sampling, or missing observations, but no consolidated home. Without one, practitioners are left assembling solutions piecemeal from disparate packages [@vanBreugel2022].

PyNumDiff addresses this gap. Its unified interface lets users compare methods on the same data, exploit specialized capabilities, and select hyperparameters without ground-truth derivatives. Derivative estimation trades data fidelity against smoothness; PyNumDiff frames this as a multi-objective optimization, finding hyperparameter settings that minimize a weighted combination of the two. A single scalar weight `tvgamma` steers between objectives; when ground-truth derivatives are available, they replace it as the optimization target. This framework spans all methods, enabling principled comparison and selection [@vanBreugel2020numerical]. A natural and growing application is SINDy [@brunton2016discovering], which discovers governing equations by regressing measured derivatives, making the quality of those estimates a direct determinant of model accuracy.


# State of the Field

Relevant Python tools exist, but none covers PyNumDiff's breadth. `numpy.gradient` and `scipy.signal.savgol_filter` [@virtanen2020scipy] handle only a sliver of the method space; `findiff` offers high-order finite difference stencils suited to clean simulation data, not noisy measurements. Historically, practitioners have had to stitch together PyKalman, PyDMD, and standalone TVR scripts [@chartrand2011numerical] with no shared API or principled way to compare results. The `derivative` package [@derivative_pkg] overlaps substantially but lacks multidimensional support, NaN handling, and hyperparameter optimization. No existing package spans PyNumDiff's seven method families with a consistent interface.

The original PyNumDiff publication [@vanBreugel2022] established the core method set and optimization framework. This version substantially revises it: the taxonomy was reorganized and corrected; the interface unified around explicit keyword arguments; test coverage expanded; and four capabilities were added throughout: multidimensional data, irregular sample spacing, missing observations, and circular domains.


# Software Design

**Package design.** All differentiation methods share the call signature

```python
x_hat, dxdt_hat = method(x, dt_or_t, **params)
```

where `x` is a NumPy array [@harris2020array] of measurements; `dt_or_t` is either a scalar step size or an array of sample locations; and keyword arguments configure the method. Explicit keyword arguments make calls self-documenting; prior positional signatures are preserved with deprecation warnings.

**Software architecture.** PyNumDiff is organized into seven method modules plus shared `utils` and `optimize` modules, a flat structure chosen for discoverability. Where strong alternatives exist, PyNumDiff delegates rather than reimplements: SciPy [@virtanen2020scipy] provides spline fitting, Savitzky-Golay filtering, and signal processing routines; NumPy [@harris2020array] provides the FFT; CVXPY [@diamond2016cvxpy] handles convex optimization for `robustdiff` and `tvrdiff`, as an optional dependency keeping the base installation lightweight. The `kalman_filter` and `rts_smooth` primitives are public, letting users with known dynamical models bypass the constant-derivative assumption of `rtsdiff`; an `innovation_fn` hook extends the filter to non-Euclidean spaces.

**Method capabilities.** All non-deprecated methods support multidimensional data via `axis`; Table 1 lists additional specialized capabilities.

| Method | Variable step | Missing Data | Outlier Robust | Circular Domain |
|---|:---:|:---:|:---:|:---:|
| `kerneldiff` | | | | |
| `finitediff` | | | | |
| `polydiff` | $\checkmark$ | $\checkmark$ | | |
| `savgoldiff` | | | | |
| `splinediff` | $\checkmark$ | $\checkmark$ | | |
| `spectraldiff` | | | | |
| `rbfdiff` | $\checkmark$ | | | |
| `tvrdiff` | | | $\checkmark$ | |
| `rtsdiff` | $\checkmark$ | $\checkmark$ | | $\checkmark$ |
| `robustdiff` | $\checkmark$ | $\checkmark$ | $\checkmark$ | |
| `lineardiff` | | | | |

Table: Specialized capabilities by method.

**Variable sample spacing.** Methods that support variable step size accept an array of sample locations in place of a scalar step size. For Kalman-based methods, this means computing the discrete-time transition matrix via matrix exponential at each actual sample interval (not extracting a fixed $\Delta t$ from the first two samples), a subtle distinction whose absence silently corrupts estimates on irregularly sampled data.

**Missing data.** NaN-valued entries are treated as missing observations: excluded from fitting, imputed from the model. This supports sensors that occasionally drop samples without any user preprocessing.

**Outlier robustness.** `robustdiff` replaces the quadratic Kalman cost with Huber loss terms on both measurement and process residuals, following @aravkin2013, with CVXPY [@diamond2016cvxpy] as the optimization backend; the sparse problem formulation scales linearly with signal length. `tvrdiff` similarly applies Huber loss on data fidelity; its total variation penalty on the derivative additionally promotes piecewise-smooth solutions for signals with abrupt transitions.

**Circular and wrapped domains.** `rtsdiff` accepts `circular=True` for quantities like angles on a periodic domain. Innovation residuals are wrapped to $[-\pi, \pi]$ before each Kalman update via an `innovation_fn` hook, and `x_hat` is returned in the same range, avoiding the large spurious spikes naive smoothers produce when a signal crosses the $\pm\pi$ boundary.

**Hyperparameter optimization.** `pynumdiff.optimize` minimizes the weighted combination described above [@vanBreugel2020numerical]. The smoothness weight `tvgamma` can be initialized from the signal's estimated cutoff frequency $f_c$ via
$$\texttt{tvgamma} = \exp(-1.6\ln f_c - 0.71\ln \Delta t - 5.1).$$
Three improvements ship in this version: intermediate evaluations are cached; the loss is robustified via Huber penalty so outliers do not bias parameter selection; and the Kalman parameter space is reduced from two independent noise variances to their log-ratio, the quantity the result actually depends on [@komarov2025].

**Testing and continuous integration.** The test suite validates all methods against analytic functions with known derivatives, covering noiseless and noisy cases across the full expected accuracy range. Care was taken to avoid tautological tests where the implementation directly determines the expected result. Tests run automatically on every push and pull request via GitHub Actions, with line coverage tracked via Coveralls.


# Research Impact

The original PyNumDiff paper [@vanBreugel2022] has accumulated nearly 30 citations since 2022, applied in experimental biology (flight kinematics from motion capture), control engineering (observer design), and data-driven dynamics identification via SINDy [@brunton2016discovering]. The present version ships under the MIT License, is available on PyPI (`pip install pynumdiff`), and is accompanied by Jupyter notebook tutorials and full Sphinx API documentation at [pynumdiff.readthedocs.io](https://pynumdiff.readthedocs.io/master/). The companion Taxonomy paper [@komarov2025], submitted to the Journal of Computational Physics, provides the theoretical underpinning and benchmarks all included methods. The PySINDy project [@pysindy] maintains its own differentiation submodule substantially overlapping with PyNumDiff's capabilities; integration discussions are ongoing.


# AI Usage Disclosure

This paper was drafted with assistance from Claude Sonnet 4.6 (Anthropic), which also implemented successive code revisions based on author feedback; all outputs were reviewed and further edited by hand, and the authors take full responsibility for accuracy.


# Acknowledgements

The authors thank Yuying Liu and Bingni W. Brunton for their contributions to the original PyNumDiff package [@vanBreugel2022], and Sasha Aravkin for discussions on convex optimization techniques that informed the robust differentiation methods. This work was supported by the National Science Foundation AI Institute in Dynamic Systems (grant number 2112085).


# References
