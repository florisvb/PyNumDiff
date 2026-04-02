---
title: 'PyNumDiff: A Python Package for Numerical Differentiation of Noisy Data'
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

Computing derivatives from measured data is a foundational requirement across science and engineering. Whether identifying governing equations from experimental observations, designing control laws, or processing sensor streams, researchers routinely need derivative estimates from discrete, noisy measurements. The textbook approach — finite differencing — amplifies noise proportionally to $1/\Delta t$, making it unreliable for real data. Smoothing before differencing helps, but the choice of algorithm and its tuning parameters substantially affect the result, and no single choice is universally best.

PyNumDiff is an open-source Python package that consolidates a broad suite of numerical differentiation methods for noisy data under a unified API. Seven families of algorithms are implemented: (1) prefiltering followed by finite difference calculation; (2) iterated finite differencing; (3) polynomial fitting [@savitzky1964]; (4) spectral and radial basis function fitting; (5) total variation regularization [@chartrand2011numerical]; (6) Kalman smoothing [@kalman1960; @rauch1965]; and (7) local approximation with linear models. The distinction between local methods — which estimate the derivative at each point from a surrounding window — and global methods — which fit a model to the entire signal — is a useful organizing principle discussed in depth by @ahnert2007. All PyNumDiff methods return a smoothed signal estimate and a derivative estimate as a matched pair `(x_hat, dxdt_hat)`. A companion paper [@komarov2025] provides a comprehensive theoretical taxonomy of these methods, benchmarks their relative performance, and guides method selection for different application scenarios.


# Statement of Need

Estimating derivatives from noisy measurements arises throughout experimental science, data-driven modeling, system identification, and control. The field has produced a diverse ecosystem of specialized algorithms, each with different strengths regarding outlier robustness, computational cost, handling of irregular sample spacing, or treatment of missing observations. Without a consolidated library, practitioners must often implement methods themselves or assemble fragmented solutions from disparate packages before arriving at one suitable for their application [@vanBreugel2022].

PyNumDiff addresses this gap. Its unified interface lets users compare methods on the same data, exploit specialized capabilities, and select hyperparameters without requiring ground-truth derivatives. Derivative estimation inherently trades off fidelity to the data against smoothness of the result. PyNumDiff frames this as a multi-objective optimization problem: find hyperparameter settings that minimize a weighted combination of data fidelity (how closely the smoothed signal matches the noisy measurements) and derivative roughness (the total variation of the estimated derivative). A single weight `tvgamma` interpolates between these objectives; when ground-truth derivatives are available, they can be used as a direct optimization target instead. This framework is universal across methods, enabling principled comparison and selection [@vanBreugel2020numerical]. The package is particularly valuable in workflows that use derivatives as regression targets: Sparse Identification of Nonlinear Dynamics (SINDy) [@brunton2016discovering], for example, learns governing equations by regressing measured derivatives, making reliable derivative estimates a prerequisite for accurate model identification.


# State of the Field

Relevant Python tools exist but none covers the full scope of PyNumDiff. `numpy.gradient` and `scipy.signal.savgol_filter` [@virtanen2020scipy] are widely available but address only a narrow slice of the method space. The `findiff` package provides high-order finite difference stencils suited to clean simulation data rather than noisy measurements. Practitioners working with noisy data have historically been forced to assemble solutions from disparate sources — PyKalman for Kalman filtering, PyDMD for spectral methods, standalone scripts for total variation regularization [@chartrand2011numerical] — without a unified API or principled parameter selection. The `derivative` package [@derivative_pkg] implements several of the same methods as PyNumDiff but without multidimensional support, NaN handling, or a hyperparameter optimization framework. No existing Python package combines the breadth of PyNumDiff's seven method families with a consistent API and the practical features this version introduces.

The original PyNumDiff publication [@vanBreugel2022] introduced the core set of methods and the multi-objective optimization framework described above. The present version represents a substantial revision: methods were reorganized into a cleaner taxonomy, corrected, and extended; the interface was unified and made more explicit; test coverage was expanded; and several capabilities — multidimensional support, irregular sample spacing, missing data, and circular domains — were added throughout to make the package more practical and widely applicable.


# Software Design

**Updated package design.** All differentiation methods share the call signature

```python
x_hat, dxdt_hat = method(x, dt_or_t, **params)
```

where `x` is a NumPy array [@harris2020array] of measurements; `dt_or_t` is either a scalar step size or an array of sample locations; and keyword arguments configure the method. The two return values always match the shape of `x`. The updated design favors explicit keyword arguments throughout, making calls self-documenting and eliminating a common source of user error. Prior positional call signatures are preserved with deprecation warnings pending removal in a future release. Using PyNumDiff involves three steps: selecting a method, optionally optimizing its hyperparameters via `pynumdiff.optimize`, and applying the method to data.

**Software architecture.** PyNumDiff is organized into seven method modules plus shared `utils` and `optimize` modules. This flat structure was chosen for discoverability: any method is reachable with a single import. Where quality alternatives exist, PyNumDiff delegates to them rather than reimplementing from scratch — SciPy [@virtanen2020scipy] provides spline fitting, Savitzky-Golay filtering, and signal processing routines; NumPy [@harris2020array] provides the FFT; and CVXPY [@diamond2016cvxpy] handles convex optimization for `robustdiff` and `tvrdiff`. CVXPY is an optional dependency isolated to the methods that require it, keeping the base installation lightweight. The `kalman_filter` and `rts_smooth` primitives are exposed as public functions so that users with known dynamical models can bypass the constant-derivative assumption of `rtsdiff` entirely. The `innovation_fn` hook on `kalman_filter` makes the filter generic to non-Euclidean measurement spaces without modifying the core algorithm.

**Method capabilities.** Table 1 summarizes the specialized capabilities of each method. All non-deprecated methods support multidimensional data via the `axis` parameter.

| Method | Variable Step | Missing Data | Outlier Robust | Circular Domain |
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

Table: Specialized capabilities by method. All methods support multidimensional data via `axis`.

**Variable sample spacing.** Methods that support variable step size accept an array of sample locations in place of a scalar step size. For the Kalman-based methods, this is handled correctly by computing the discrete-time transition matrix via matrix exponential at each actual sample interval rather than extracting a fixed $\Delta t$ from the first two samples — a subtle but important distinction whose absence can silently corrupt derivative estimates on irregularly sampled data.

**Missing data.** Methods that support missing data treat NaN-valued entries as missing observations, excluding them from fitting and imputing estimates via the model. This supports sensors that occasionally drop samples without requiring the user to preprocess or interpolate the input.

**Outlier robustness.** `robustdiff` replaces the quadratic Kalman cost with Huber loss terms on both measurement and process residuals, following the robust smoothing framework of @aravkin2013 and using CVXPY [@diamond2016cvxpy] as the optimization backend. The problem is formulated as a sparse system, so it scales linearly with the number of samples. `tvrdiff` similarly replaces the quadratic data fidelity term with a Huber loss when outliers are present; its total variation penalty on the derivative additionally promotes piecewise-smooth solutions appropriate for signals with abrupt transitions.

**Circular and wrapped domains.** `rtsdiff` accepts a `circular=True` flag for quantities like angles that live on a periodic domain. Innovation residuals are wrapped to $[-\pi, \pi]$ before each Kalman update step via an `innovation_fn` hook on the underlying `kalman_filter` primitive, and `x_hat` is returned wrapped to the same range. This avoids the erroneous large-magnitude spikes that naive smoothers produce when a signal crosses the $\pm\pi$ boundary.

**Hyperparameter optimization.** Every method has tuning parameters, so PyNumDiff provides a multi-objective optimization framework in `pynumdiff.optimize` that minimizes the weighted combination described above [@vanBreugel2020numerical]. The smoothness weight `tvgamma` can be initialized from the signal's estimated cutoff frequency $f_c$ via the empirical formula
$$\texttt{tvgamma} = \exp(-1.6\ln f_c - 0.71\ln \Delta t - 5.1).$$
The optimization was substantially improved in the current version: intermediate evaluations are cached to avoid redundant function calls; the loss function is robustified via Huber penalty so that outliers do not bias hyperparameter selection; and the Kalman parameter space was reduced from two independent noise variances to their log-ratio, which is the quantity the result actually depends on [@komarov2025].

**Testing and continuous integration.** The test suite validates all methods against analytic test functions with known derivatives, checking both noiseless and noisy cases across the full expected accuracy range. Care was taken to avoid tautological tests in which the implementation directly determines the expected result. Tests run automatically on every push and pull request via GitHub Actions, with line coverage tracked via Coveralls.


# Research Impact

The original PyNumDiff publication [@vanBreugel2022] has accumulated nearly 30 citations since 2022 and has been applied in experimental biology to estimate flight kinematics from motion capture data, in control engineering for observer design, and in data-driven dynamics identification via SINDy [@brunton2016discovering]. The present version is distributed under the MIT License, available on PyPI (`pip install pynumdiff`), and accompanied by Jupyter notebook tutorials and full Sphinx API documentation at [pynumdiff.readthedocs.io](https://pynumdiff.readthedocs.io/master/). The companion Taxonomy paper [@komarov2025], submitted to the Journal of Computational Physics, provides the theoretical analysis motivating the method collection and benchmarks all included methods across a range of test signals. Active community engagement is evidenced by ongoing integration discussions with the PySINDy project [@pysindy], which currently maintains its own differentiation submodule that substantially overlaps with PyNumDiff's capabilities.


# AI Usage Disclosure

The draft of this paper was prepared with assistance from Claude Sonnet 4.6 (Anthropic), integrating material from the repository, documentation, release history, and related publications. Claude Sonnet 4.6 was also used to assist with code review and pull request feedback on several recent contributions to the repository, with all outputs reviewed and edited by the authors before acceptance. The authors take full responsibility for the accuracy of all content.


# Acknowledgements

The authors thank Yuying Liu and Bingni W. Brunton for their contributions to the original PyNumDiff package [@vanBreugel2022], and Sasha Aravkin for discussions on convex optimization techniques that informed the robust differentiation methods. This work was supported by the National Science Foundation AI Institute in Dynamic Systems (grant number 2112085).


# References
