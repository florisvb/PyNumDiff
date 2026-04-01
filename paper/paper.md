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
    orcid: 0000-0000-0000-0000  # TODO: fill in
    corresponding: true
    affiliation: 1
  - name: Floris van Breugel
    orcid: 0000-0000-0000-0000  # TODO: fill in
    affiliation: 2
  - name: J. Nathan Kutz
    orcid: 0000-0003-0944-900X
    affiliation: 1
affiliations:
  - name: Department of Applied Mathematics, University of Washington, USA
    index: 1
  - name: Department of Mechanical Engineering, University of Nevada, Reno, USA
    index: 2
date: 1 April 2026
bibliography: paper.bib
---

# Summary

Computing derivatives from measured data is a foundational requirement across science and
engineering. Whether fitting governing equations from experimental observations, designing
control laws, or processing sensor streams, researchers routinely need derivative estimates
from discrete, noisy measurements. The textbook approach — finite differencing — amplifies
noise proportionally to $1/\Delta t$, making it unreliable for real data. Smoothing before
differencing helps, but the choice of algorithm and its tuning parameters substantially
affect the result, and no single choice is universally best.

PyNumDiff is an open-source Python package that addresses this by providing a broad suite of
numerical differentiation methods for noisy data under a unified API. Seven families of
algorithms are implemented: (1) prefiltering followed by finite difference calculation;
(2) iterated finite differencing; (3) polynomial fitting; (4) basis function fitting;
(5) total variation regularization; (6) Kalman smoothing and its outlier-robust variant;
and (7) local approximation with linear models. All methods return a smoothed signal estimate
and a derivative estimate as a matched pair `(x_hat, dxdt_hat)`. A companion paper
[@komarov2025] provides a comprehensive theoretical taxonomy of these methods, benchmarks
their relative performance, and guides method selection for different application scenarios.


# Statement of Need

Estimating derivatives from noisy measurements arises throughout experimental science,
data-driven modeling, system identification, and control. Standard finite differences amplify
noise, and while smoothing helps, the field has produced a diverse ecosystem of specialized
algorithms — each with different strengths regarding outlier robustness, computational cost,
handling of irregular time steps, or treatment of missing data. Without a consolidated
library, practitioners either default to the nearest available tool or implement bespoke
solutions that are hard to compare or reproduce.

PyNumDiff fills this gap. Its unified interface lets users compare methods directly on the
same data, exploit specialized capabilities, and select hyperparameters without requiring
ground-truth derivatives. The package is particularly valuable in workflows that use
derivatives as regression targets: for example, Sparse Identification of Nonlinear Dynamics
(SINDy) [@brunton2016discovering] learns governing equations by regressing measured
derivatives, making clean, reliable derivative estimates a prerequisite for accurate model
identification.


# State of the Field

Relevant Python tools exist, but none covers the full scope of PyNumDiff. `numpy.gradient`
and `scipy.signal.savgol_filter` [@virtanen2020scipy] are widely available but address only
a narrow slice of the method space. The `findiff` package provides high-order finite
difference stencils for clean simulation data rather than noisy measurements. The standalone
`TVRegDiff` code [@chartrand2011numerical] implements a single total variation regularization
method; PyNumDiff includes and extends this. No existing Python package combines seven method
families with a unified API, NaN and variable-step support, multidimensional capability,
circular domain handling, and integrated hyperparameter optimization.

The original PyNumDiff [@vanBreugel2022] introduced four method families with basic parameter
optimization. The present version substantially extends that foundation in scope, capability,
and software quality.


# Software Design

**Unified API.** All differentiation methods share the call signature

```python
x_hat, dxdt_hat = method(x, dt_or_t, **params)
```

where `x` is a NumPy array [@harris2020array] of measurements, `dt_or_t` is either a
scalar step size or an array of sample locations (for variable step size), and keyword
arguments configure the method. The two return values always match the shape of `x`. This
uniformity makes method comparison straightforward and enables drop-in substitution.

**Multidimensional support.** An `axis` parameter controls which axis is differentiated,
allowing all methods to operate on blocks of time series simultaneously. The implementation
iterates over all non-time axes with `np.ndindex`, applying the algorithm independently to
each vector, so a 2D array of shape `(N, D)` is handled without reshaping or looping in
user code.

**Missing data and variable step size.** Several methods handle NaN-valued entries as missing
observations and non-uniform sample spacing: `splinediff`, `polydiff`, `rbfdiff`, `rtsdiff`,
and `robustdiff`. This supports realistic experimental scenarios where sensors skip samples or
operate at irregular rates.

**Circular and wrapped domains.** `rtsdiff` accepts a `circular=True` flag for quantities
like angles that live on a periodic domain. Innovation residuals are wrapped to $[-\pi, \pi]$
before each Kalman update, and `x_hat` is returned wrapped to the same range. This avoids the
erroneous large-magnitude spikes that naive smoothers produce when a signal crosses the
$\pm\pi$ boundary. The wrapping is injected via an optional `innovation_fn` hook on the
underlying `kalman_filter` primitive, keeping the filter itself general.

**Outlier robustness.** `robustdiff` solves a convex optimization problem that replaces the
least-squares Kalman cost with Huber loss on both measurement and process residuals,
dramatically reducing outlier influence. It uses `cvxpy` [@diamond2016cvxpy] as its
optimization backend, available via `pip install pynumdiff[advanced]`. `tvrdiff` similarly
minimizes a total variation penalty on the derivative, promoting piecewise-smooth solutions
and tolerating abrupt changes; it supports regularization of the velocity, acceleration, or
jerk (orders 1–3).

**Hyperparameter optimization.** Because every method has tuning parameters, PyNumDiff
provides a multi-objective optimization framework in `pynumdiff.optimize` that minimizes a
weighted sum of a smoothness penalty and a data fidelity term [@vanBreugel2020numerical].
When ground-truth derivatives are available, the optimizer minimizes derivative error
directly. The smoothness weight `tvgamma` can be initialized from the signal's estimated
cutoff frequency $f_c$ via the empirical formula
$$\texttt{tvgamma} = \exp(-1.6\ln f_c - 0.71\ln \Delta t - 5.1).$$


# Research Impact

The original PyNumDiff paper [@vanBreugel2022] has accumulated over 60 citations since its
2022 publication, and the package has been used in experimental biology to estimate flight
kinematics from motion capture, in control engineering for observer design, and in
data-driven dynamics identification. The companion Taxonomy paper [@komarov2025], submitted
to the Journal of Computational Physics, provides the theoretical analysis underpinning the
method collection and benchmarks all included methods across diverse test signals. The
package is available on PyPI (`pip install pynumdiff`), documented at
[pynumdiff.readthedocs.io](https://pynumdiff.readthedocs.io/master/), and archived on
Zenodo [@pynumdiff_zenodo].


# AI Usage Disclosure

The draft of this paper was prepared with assistance from Claude Sonnet 4.6 (Anthropic).
This included integrating material from the repository, documentation, and related papers
into a coherent draft. The authors reviewed and edited all content and take full
responsibility for its accuracy.


# Acknowledgements

The authors thank Yuying Liu and Bingni W. Brunton for their contributions to the original
PyNumDiff package [@vanBreugel2022].


# References
