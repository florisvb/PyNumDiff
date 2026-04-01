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

Computing derivatives from measured data is a foundational requirement across science and
engineering. Whether identifying governing equations from experimental observations, designing
control laws, or processing sensor streams, researchers routinely need derivative estimates
from discrete, noisy measurements. The textbook approach — finite differencing — amplifies
noise proportionally to $1/\Delta t$, making it unreliable for real data. Smoothing before
differencing helps, but the choice of algorithm and its tuning parameters substantially
affect the result, and no single choice is universally best.

PyNumDiff is an open-source Python package that consolidates a broad suite of numerical
differentiation methods for noisy data under a unified API. Seven families of algorithms are
organized: (1) prefiltering followed by finite difference calculation; (2) iterated finite
differencing; (3) polynomial fitting; (4) basis function fitting; (5) total variation
regularization; (6) Kalman smoothing; and (7) local approximation with linear models. All
methods return a smoothed signal estimate and a derivative estimate as a matched pair
`(x_hat, dxdt_hat)`. A companion paper [@komarov2025] provides a comprehensive theoretical
taxonomy of these methods, benchmarks their relative performance, and guides method selection
for different application scenarios.


# Statement of Need

Estimating derivatives from noisy measurements arises throughout experimental science,
data-driven modeling, system identification, and control. The field has produced a diverse
ecosystem of specialized algorithms, each with different strengths regarding outlier
robustness, computational cost, handling of irregular time steps, or treatment of missing
data. Without a consolidated library, practitioners either default to the nearest available
tool or implement bespoke solutions that are difficult to compare or reproduce.

PyNumDiff addresses this gap. Its unified interface lets users compare methods on the same
data, exploit specialized capabilities, and select hyperparameters without requiring
ground-truth derivatives. The package is particularly valuable in workflows that use
derivatives as regression targets: Sparse Identification of Nonlinear Dynamics (SINDy)
[@brunton2016discovering], for example, learns governing equations by regressing measured
derivatives, making reliable derivative estimates a prerequisite for accurate model
identification. More broadly, any system identification or control design pipeline that
begins by estimating rates of change from sensor data is a natural user of PyNumDiff.


# State of the Field

Relevant Python tools exist but none covers the full scope of PyNumDiff. `numpy.gradient`
and `scipy.signal.savgol_filter` [@virtanen2020scipy] are widely available but address only
a narrow slice of the method space. The `findiff` package provides high-order finite
difference stencils for clean simulation data rather than noisy measurements. The standalone
`TVRegDiff` code [@chartrand2011numerical] implements a single total variation regularization
method; PyNumDiff includes and extends this. The `derivative` package [@derivative_pkg]
implements several of the same methods, but without multidimensional support, missing data
handling, or a hyperparameter optimization framework. No existing Python package combines
the breadth of PyNumDiff's seven method families with a consistent API and the full set of
capabilities described below.

The original PyNumDiff publication [@vanBreugel2022] introduced the core concept and a
multi-objective optimization framework for hyperparameter selection. The present version
represents a substantial revision of the codebase: methods were reorganized into a cleaner
taxonomy, corrected, and extended; the interface was unified; test coverage was dramatically
improved; and several new capabilities were added throughout.


# Software Design

**Unified API.** All differentiation methods share the call signature

```python
x_hat, dxdt_hat = method(x, dt_or_t, **params)
```

where `x` is a NumPy array [@harris2020array] of measurements, `dt_or_t` is either a scalar
step size or an array of sample locations (for variable step size), and keyword arguments
configure the method. The two return values always match the shape of `x`. The revision
moved away from positional parameter lists toward explicit keyword arguments throughout,
making calls self-documenting and eliminating a common source of user error. Deprecated
positional interfaces are retained with warnings.

**Multidimensional support and interface consistency.** An `axis` parameter controls which
dimension is differentiated, allowing all non-deprecated methods to operate on blocks of
time series simultaneously without reshaping or looping in user code. The implementation
iterates over all non-time axes using `np.ndindex`, applying the algorithm to each vector
independently.

**Missing data and variable step size.** Several methods handle NaN-valued entries as
missing observations and non-uniform sample spacing: `splinediff`, `polydiff`, `rbfdiff`,
`rtsdiff`, and `robustdiff`. This supports realistic experimental scenarios where sensors
drop samples or operate at irregular rates. For the Kalman-based methods, variable step size
is handled correctly by computing the discrete-time transition matrix via matrix exponential
at each actual time increment, rather than using a fixed $\Delta t$ extracted from the first
two samples — a subtle but important distinction for data with irregular spacing.

**Outlier robustness.** `robustdiff` solves a convex optimization problem that replaces the
least-squares Kalman cost with Huber loss terms on both measurement residuals and process
residuals, following the robust smoothing framework of @aravkin2013 and using `cvxpy`
[@diamond2016cvxpy] as the optimization backend. The problem is formulated as a sparse
system so that it runs in time linear in the number of samples, making it practical for
long time series. `tvrdiff` similarly promotes piecewise-smooth solutions by penalizing
the total variation of the derivative, and the data fidelity term can be replaced with a
Huber loss for additional robustness to outliers. Both methods are available via
`pip install pynumdiff[advanced]`.

**Circular and wrapped domains.** `rtsdiff` accepts a `circular=True` flag for quantities
like angles that live on a periodic domain. Innovation residuals are wrapped to
$[-\pi, \pi]$ before each Kalman update step via an `innovation_fn` hook on the underlying
`kalman_filter` primitive, and `x_hat` is returned wrapped to the same range. This avoids
the erroneous large-magnitude spikes that naive smoothers produce when a signal crosses the
$\pm\pi$ boundary.

**Hyperparameter optimization.** Every method has tuning parameters, so PyNumDiff provides
a multi-objective optimization framework in `pynumdiff.optimize` that minimizes a weighted
combination of a smoothness penalty and a data fidelity term [@vanBreugel2020numerical].
When ground-truth derivatives are available, derivative error is minimized directly.
The smoothness weight `tvgamma` can be initialized from the signal's estimated cutoff
frequency $f_c$ via the empirical formula
$$\texttt{tvgamma} = \exp(-1.6\ln f_c - 0.71\ln \Delta t - 5.1).$$
The optimization was substantially improved in the current version: intermediate evaluations
are cached to avoid redundant function calls, the loss function was made robust to outliers
via Huber penalty, and the Kalman parameter space was reduced from two independent noise
variances to their log-ratio, which is the quantity the result actually depends on
[@komarov2025].

**Testing and continuous integration.** The test suite validates all methods against a set
of analytic test functions with known derivatives, checking both noiseless and noisy cases
across the full dynamic range of expected accuracy. Tests are run automatically on every
push and pull request via GitHub Actions, with coverage tracked via Coveralls. Care was
taken to avoid tautological tests in which the implementation directly determines the
expected result.


# Research Impact

PyNumDiff has been applied in experimental biology to estimate flight kinematics from motion
capture, in control engineering for observer design, and in data-driven dynamics
identification [@vanBreugel2022]. The package is available on PyPI
(`pip install pynumdiff`) and documented at
[pynumdiff.readthedocs.io](https://pynumdiff.readthedocs.io/master/). The companion Taxonomy
paper [@komarov2025], submitted to the Journal of Computational Physics, provides the
theoretical analysis motivating the method collection and benchmarks all included methods.


# AI Usage Disclosure

The draft of this paper was prepared with assistance from Claude Sonnet 4.6 (Anthropic),
integrating material from the repository, documentation, release history, and related
publications. The authors reviewed and edited all content and take full responsibility for
its accuracy.


# Acknowledgements

The authors thank Yuying Liu and Bingni W. Brunton for their contributions to the original
PyNumDiff package [@vanBreugel2022], and Sasha Aravkin for discussions on convex
optimization techniques that informed the robust differentiation methods. This work was
supported by the National Science Foundation AI Institute in Dynamic Systems (grant
number 2112085).


# References
