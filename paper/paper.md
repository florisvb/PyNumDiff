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

PyNumDiff is an open-source Python package that consolidates a broad suite of numerical differentiation methods for noisy data under a unified API. Seven families of algorithms are implemented: (1) prefiltering followed by finite difference calculation; (2) iterated finite differencing; (3) polynomial fitting; (4) basis function fitting; (5) total variation regularization; (6) Kalman smoothing; and (7) local approximation with linear models. All methods return a smoothed signal estimate and a derivative estimate as a matched pair `(x_hat, dxdt_hat)`. A companion paper [@komarov2025] provides a comprehensive theoretical taxonomy of these methods, benchmarks their relative performance, and guides method selection for different application scenarios.


# Statement of Need

Estimating derivatives from noisy measurements arises throughout experimental science, data-driven modeling, system identification, and control. The field has produced a diverse ecosystem of specialized algorithms, each with different strengths regarding outlier robustness, computational cost, handling of irregular sample spacing, or treatment of missing observations. Without a consolidated library, practitioners either default to the nearest available tool or implement bespoke solutions that are difficult to compare or reproduce.

PyNumDiff addresses this gap. Its unified interface lets users compare methods on the same data, exploit specialized capabilities, and select hyperparameters without requiring ground-truth derivatives. Derivative estimation inherently trades off fidelity to the data against smoothness of the result. PyNumDiff frames this as a multi-objective optimization problem: find hyperparameter settings that minimize a weighted combination of data fidelity (how closely the smoothed signal matches the noisy measurements) and derivative roughness (the total variation of the estimated derivative). A single weight `tvgamma` interpolates between these objectives; when ground-truth derivatives are available, they can be used as a direct optimization target instead. This framework is universal across methods, enabling principled comparison and selection. The package is particularly valuable in workflows that use derivatives as regression targets: Sparse Identification of Nonlinear Dynamics (SINDy) [@brunton2016discovering], for example, learns governing equations by regressing measured derivatives, making reliable derivative estimates a prerequisite for accurate model identification.


# State of the Field

Relevant Python tools exist but none covers the full scope of PyNumDiff. `numpy.gradient` and `scipy.signal.savgol_filter` [@virtanen2020scipy] are widely available but address only a narrow slice of the method space. The `findiff` package provides high-order finite difference stencils suited to clean simulation data rather than noisy measurements. The standalone `TVRegDiff` code [@chartrand2011numerical] implements a single total variation regularization method; PyNumDiff includes and extends this. The `derivative` package [@derivative_pkg] implements several of the same methods but without multidimensional support, principled hyperparameter selection, or the broader set of capabilities described below. No existing Python package combines the breadth of PyNumDiff's seven method families with a consistent API and the practical features this version introduces.

The original PyNumDiff publication [@vanBreugel2022] introduced the core set of methods and the multi-objective optimization framework described above. The present version represents a substantial revision: methods were reorganized into a cleaner taxonomy, corrected, and extended; the interface was unified and made more explicit; test coverage was expanded; and several capabilities — multidimensional support, irregular sample spacing, missing data, and circular domains — were added throughout to make the package more practical and widely applicable.


# Software Design

**Updated package design.** All differentiation methods share the call signature

```python
x_hat, dxdt_hat = method(x, dt_or_t, **params)
```

where `x` is a NumPy array [@harris2020array] of measurements; `dt_or_t` is either a scalar step size or an array of sample locations; and keyword arguments configure the method. The two return values always match the shape of `x`. The updated design favors explicit keyword arguments throughout, making calls self-documenting and eliminating a common source of user error. Prior positional call signatures are preserved with deprecation warnings pending removal in a future release.

**Multidimensional support.** An `axis` parameter controls which dimension is differentiated, allowing all non-deprecated methods to operate on blocks of data simultaneously without reshaping or looping in user code. The implementation iterates over the remaining axes using `np.ndindex`, applying the algorithm to each vector independently.

**Variable sample spacing.** `splinediff`, `polydiff`, `rbfdiff`, `rtsdiff`, and `robustdiff` accept an array of sample locations in place of a scalar step size, enabling differentiation of irregularly sampled data. For the Kalman-based methods, this is handled correctly by computing the discrete-time transition matrix via matrix exponential at each actual sample interval rather than using a fixed $\Delta t$ — a subtle but important distinction whose absence can silently corrupt derivative estimates.

**Missing data.** `splinediff`, `polydiff`, `rtsdiff`, and `robustdiff` treat NaN-valued entries as missing observations, excluding them from fitting and imputing estimates via the model. This supports sensors that occasionally drop samples without requiring the user to preprocess or interpolate the input.

**Outlier robustness.** `robustdiff` replaces the quadratic Kalman cost with Huber loss terms on both measurement and process residuals, following the robust smoothing framework of @aravkin2013 and using `cvxpy` [@diamond2016cvxpy] as the optimization backend. The problem is formulated as a sparse system, so it scales linearly with the number of samples. `tvrdiff` similarly replaces the quadratic data fidelity term with a Huber loss when outliers are present; its total variation penalty on the derivative additionally promotes piecewise-smooth solutions appropriate for signals with abrupt transitions.

**Circular and wrapped domains.** `rtsdiff` accepts a `circular=True` flag for quantities like angles that live on a periodic domain. Innovation residuals are wrapped to $[-\pi, \pi]$ before each Kalman update step via an `innovation_fn` hook on the underlying `kalman_filter` primitive, and `x_hat` is returned wrapped to the same range. This avoids the erroneous large-magnitude spikes that naive smoothers produce when a signal crosses the $\pm\pi$ boundary.

**Hyperparameter optimization.** Every method has tuning parameters, so PyNumDiff provides a multi-objective optimization framework in `pynumdiff.optimize` that minimizes the weighted combination described in the Statement of Need [@vanBreugel2020numerical]. The smoothness weight `tvgamma` can be initialized from the signal's estimated cutoff frequency $f_c$ via the empirical formula
$$\texttt{tvgamma} = \exp(-1.6\ln f_c - 0.71\ln \Delta t - 5.1).$$
The optimization was substantially improved in the current version: intermediate evaluations are cached to avoid redundant function calls; the loss function is robustified via Huber penalty so that outliers do not bias hyperparameter selection; and the Kalman parameter space was reduced from two independent noise variances to their log-ratio, which is the quantity the result actually depends on [@komarov2025].

**Testing and continuous integration.** The test suite validates all methods against analytic test functions with known derivatives, checking both noiseless and noisy cases across the full expected accuracy range. Care was taken to avoid tautological tests in which the implementation directly determines the expected result. Tests run automatically on every push and pull request via GitHub Actions, with line coverage tracked via Coveralls.


# Research Impact

PyNumDiff has been applied in experimental biology to estimate flight kinematics from motion capture, in control engineering for observer design, and in data-driven dynamics identification [@vanBreugel2022]. The package is available on PyPI (`pip install pynumdiff`) and documented at [pynumdiff.readthedocs.io](https://pynumdiff.readthedocs.io/master/). The companion Taxonomy paper [@komarov2025], submitted to the Journal of Computational Physics, provides the theoretical analysis motivating the method collection and benchmarks all included methods.


# AI Usage Disclosure

The draft of this paper was prepared with assistance from Claude Sonnet 4.6 (Anthropic), integrating material from the repository, documentation, release history, and related publications. The authors reviewed and edited all content and take full responsibility for its accuracy.


# Acknowledgements

The authors thank Yuying Liu and Bingni W. Brunton for their contributions to the original PyNumDiff package [@vanBreugel2022], and Sasha Aravkin for discussions on convex optimization techniques that informed the robust differentiation methods. This work was supported by the National Science Foundation AI Institute in Dynamic Systems (grant number 2112085).


# References
