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
    corresponding: true
    affiliation: 1
  - name: Floris van Breugel
    affiliation: 2
  - name: Maria Protogerou
    affiliation: 4
  - name: J. Nathan Kutz
    affiliation: 3
affiliations:
  - name: Department of Electrical and Computer Engineering, University of Washington, USA
    index: 1
  - name: Department of Mechanical Engineering, University of Nevada, Reno, USA
    index: 2
  - name: Autodesk Research, London, UK
    index: 3
  - name: Department of Applied Mathematics, University of Washington, USA
    index: 4
date: 1 April 2026
bibliography: paper.bib
---

# Summary

Derivatives of measured data are a prerequisite across science and engineering: identifying governing equations, designing controllers, and processing sensor streams alike. The textbook remedy, finite differencing, amplifies noise as $1/\Delta t$ and deteriorates rapidly as data grows noisier or more finely sampled. Smoothing before differencing helps, but algorithm choice and tuning substantially affect the result, and no single approach wins universally.

PyNumDiff is an open-source Python package consolidating a broad suite of numerical differentiation methods under a unified API. Seven algorithm families are implemented: (1) prefiltering followed by finite difference calculation; (2) iterated finite differencing; (3) polynomial fitting [@savitzky1964]; (4) spectral, radial basis function, and wavelet fitting; (5) total variation regularization [@chartrand2011numerical]; (6) Kalman smoothing [@kalman1960; @rauch1965]; and (7) local approximation with linear models. @ahnert2007 provides a useful taxonomy of these families, distinguishing local methods (which estimate derivatives from a surrounding window) from global methods that fit the entire signal at once. All PyNumDiff methods return a matched pair `(x_hat, dxdt_hat)`. A companion paper [@komarov2025] benchmarks all methods across test signals and guides selection for different application scenarios. This paper describes the package's second generation. Its release numbers stay below 1.0 by convention, and the version described here is 0.2.3.


# Statement of Need

Numerical differentiation has a diverse ecosystem of specialized algorithms, each with different strengths in outlier robustness, computational cost, irregular sampling, or missing observations, but no consolidated home. Without one, practitioners are left assembling solutions piecemeal from disparate packages [@vanBreugel2022].

PyNumDiff addresses this gap. Its unified interface lets users compare methods on the same data, exploit specialized capabilities, and select hyperparameters without ground-truth derivatives. Derivative estimation trades data fidelity against smoothness, and PyNumDiff frames that tradeoff as a multi-objective optimization spanning all methods, with a single scalar weight steering between the two. When a ground-truth derivative is available, it can serve as the optimization target instead [@vanBreugel2020numerical]. A natural and growing application is SINDy [@brunton2016discovering], which discovers governing equations by regressing measured derivatives, making the quality of those estimates a direct determinant of model accuracy.


# State of the Field

Relevant Python tools exist, but none covers PyNumDiff's breadth. `numpy.gradient` and `scipy.signal.savgol_filter` [@virtanen2020scipy] handle only a sliver of the method space; `findiff` offers high-order finite difference stencils suited to clean simulation data, not noisy measurements. Historically, practitioners have had to stitch together PyKalman, PyDMD, and standalone TVR scripts [@chartrand2011numerical] with no shared API or principled way to compare results. The `derivative` package [@derivative_pkg] overlaps substantially but lacks multidimensional support, NaN handling, and hyperparameter optimization. No existing package spans PyNumDiff's seven method families with a consistent interface.

The original PyNumDiff publication [@vanBreugel2022] established the core method set and optimization framework. This second generation rewrites it from the ground up and consolidates it. `kerneldiff`, `rtsdiff`, and `tvrdiff` each cover by parameter what had been separate functions, so the package presents fewer entry points than before while spanning more methods. `rbfdiff`, `waveletdiff`, and `robustdiff` are new, bringing the set to the twelve methods of Table 1, all organized into the seven families above behind one keyword-argument signature. A single optimizer serves every one of them, and four capabilities now span the package, covering multidimensional data, irregular sample spacing, missing observations, and circular domains.


# Software Design

**Package design.** All differentiation methods share the call signature

```python
x_hat, dxdt_hat = method(x, dt_or_t, **params)
```

where `x` is a NumPy array [@harris2020array] of measurements; `dt_or_t` is either a scalar step size or an array of sample locations; and keyword arguments configure the method, making calls self-documenting. Prior positional signatures are preserved with deprecation warnings.

**Software architecture.** PyNumDiff is organized into seven method modules plus shared `utils` and `optimize` modules in a flat structure. Where strong alternatives exist, PyNumDiff delegates rather than reimplements: SciPy [@virtanen2020scipy] provides spline fitting, Savitzky-Golay filtering, and signal processing routines; NumPy [@harris2020array] provides the FFT; PyWavelets [@lee2019pywavelets] provides the discrete wavelet transform for `waveletdiff`; CVXPY [@diamond2016cvxpy] handles convex optimization for `robustdiff` and `tvrdiff` as an optional dependency. The public `kalman_filter` and `rts_smooth` primitives let users with known dynamics bypass `rtsdiff`'s constant-derivative model.

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
| `waveletdiff` | | | | |
| `tvrdiff` | | | $\checkmark$ | |
| `rtsdiff` | $\checkmark$ | $\checkmark$ | | $\checkmark$ |
| `robustdiff` | $\checkmark$ | $\checkmark$ | $\checkmark$ | |
| `lineardiff` | | | | |

Table: Specialized capabilities by method.

**Irregular and incomplete sampling.** Methods that support variable step size accept an array of sample locations in place of a scalar step, and Kalman-based methods then compute the transition matrix by matrix exponential at each actual interval. NaN entries are treated as missing observations, excluded from fitting and imputed from the model, so sensors that drop samples need no preprocessing.

**Outlier robustness.** `robustdiff` replaces the quadratic Kalman cost with Huber loss terms on both measurement and process residuals, following @aravkin2013, with CVXPY [@diamond2016cvxpy] as the optimization backend; the sparse problem formulation scales linearly with signal length. `tvrdiff` similarly applies Huber loss on data fidelity; its total variation penalty on the derivative additionally promotes piecewise-smooth solutions for signals with abrupt transitions.

**Circular and wrapped domains.** `rtsdiff` accepts `circular=True` for quantities like angles on a periodic domain. Innovation residuals are wrapped to $[-\pi, \pi]$ before each Kalman update via an `innovation_fn` hook, and `x_hat` is returned in the same range, avoiding the large spurious spikes naive smoothers produce when a signal crosses the $\pm\pi$ boundary.

**Hyperparameter optimization.** `pynumdiff.optimize` minimizes the weighted combination described above [@vanBreugel2020numerical], and the smoothness weight `tvgamma` can be initialized automatically from the signal's estimated cutoff frequency. This version robustifies the loss with a Huber penalty so outliers do not bias parameter selection, and it reduces the Kalman parameter space from two independent noise variances to their log-ratio, the only salient factor [@komarov2025]. Categorical and boolean hyperparameters are now supported natively, so discrete choices such as derivative order can be optimized jointly with continuous ones. Repeated evaluations are cached to avoid redundant work. A single call to `suggest_method` runs this search across nearly every method in the package and reports the best fit, which is the practical payoff of putting them all behind one interface.

**Testing and continuous integration.** The test suite validates all methods against analytic functions with known derivatives, covering both noiseless and noisy cases. Care was taken to avoid tautological tests where the implementation directly determines the expected result. GitHub Actions runs the suite on every push and pull request, and Coveralls tracks line coverage, currently 90%.


# Research Impact

The original PyNumDiff paper [@vanBreugel2022] has been applied in experimental biology (flight kinematics from motion capture), control engineering (observer design), and data-driven dynamics identification via SINDy [@brunton2016discovering]. Tutorial notebooks and full API documentation are published at [pynumdiff.readthedocs.io](https://pynumdiff.readthedocs.io/master/). The companion Taxonomy paper [@komarov2025], submitted to the Journal of Computational Physics, provides the theoretical underpinning and benchmarks all included methods. The PySINDy project [@pysindy] maintains its own differentiation submodule substantially overlapping with PyNumDiff's capabilities; integration discussions are ongoing.


# AI Usage Disclosure

This paper was drafted with assistance from Claude Sonnet 4.6 and Claude Opus 4.8 (Anthropic), which also implemented successive code revisions to address recent issues and author feedback. All outputs were reviewed and further edited by hand, and the authors take full responsibility for accuracy.


# Acknowledgements

The authors thank Yuying Liu and Bingni W. Brunton for their contributions to the original PyNumDiff package [@vanBreugel2022], and Sasha Aravkin for discussions on convex optimization techniques that informed the robust differentiation methods. This work was supported by the NSF AI Institute in Dynamic Systems (grant number 2112085).


# References
