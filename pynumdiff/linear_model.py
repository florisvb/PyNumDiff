"""Fit a linear system model in a sliding window"""
from warnings import warn
import math, scipy
import numpy as np

from pynumdiff.finite_difference import finitediff
from pynumdiff.polynomial_fit import savgoldiff as _savgoldiff # patch through
from pynumdiff.polynomial_fit import polydiff as _polydiff # patch through
from pynumdiff.basis_fit import spectraldiff as _spectraldiff # patch through
from pynumdiff.utils import utility

try: import cvxpy
except ImportError: pass


def savgoldiff(*args, **kwargs): # pragma: no cover pylint: disable=missing-function-docstring
    warn("`savgoldiff` has moved to `polynomial_fit.savgoldiff` and will be removed from "
        "`linear_model` in a future release.", DeprecationWarning)
    return _savgoldiff(*args, **kwargs)

def polydiff(*args, **kwargs): # pragma: no cover pylint: disable=missing-function-docstring
    warn("`polydiff` has moved to `polynomial_fit.polydiff` and will be removed from "
        "`linear_model` in a future release.", DeprecationWarning)
    return _polydiff(*args, **kwargs)

def spectraldiff(*args, **kwargs): # pragma: no cover pylint: disable=missing-function-docstring
    warn("`spectraldiff` has moved to `basis_fit.spectraldiff` and will be removed from "
        "`linear_model` in a future release.", DeprecationWarning)
    return _spectraldiff(*args, **kwargs)


_PROBLEM_CACHE = {} # (order, window length) -> a parametrized CVXPY problem, so identically shaped windows reuse it

def lineardiff(x, dt, params=None, options=None, order=None, gamma=None, window_size=None,
    step_size=None, kernel='friedrichs', solver='CLARABEL', axis=0):
    """Slide a smoothing derivative function across data, with specified window size.

    :param np.array[float] x: data to differentiate. May be multidimensional; see :code:`axis`.
    :param float dt: step size
    :param list[int, float, int] params: (**deprecated**, prefer :code:`order`, :code:`gamma`, and :code:`window_size`)
    :param dict options: (**deprecated**, prefer :code:`sliding`, :code:`step_size`, :code:`kernel`, and :code:`solver`
            a dictionary consisting of {'sliding': (bool), 'step_size': (int), 'kernel_name': (str), 'solver': (str)}
    :param int>1 order: order of the polynomial
    :param float gamma: regularization term, in multiples of the data's own scale, so a given value means the same
            thing whatever the units. See #222
    :param int window_size: size of the sliding window (ignored if not sliding)
    :param int step_size: step size for sliding. Defaults to a fifth of :code:`window_size`, because what matters
            is the overlap ratio, not an absolute stride: a fifth costs a few percent of accuracy against a much finer
            stride while running several times faster, and strides past about half the window degrade badly
    :param str kernel: name of kernel to use for weighting and smoothing windows ('gaussian' or 'friedrichs')
    :param str solver: CVXPY solver to use, one of :code:`cvxpy.installed_solvers()`. CLARABEL converges reliably,
            but OSQP stalls on the badly-scaled subproblems short windows produce, returning half-converged iterates
    :param int axis: axis along which to differentiate (default 0)

    :return: - **x_hat** (np.array) -- estimated (smoothed) x
             - **dxdt_hat** (np.array) -- estimated derivative of x
    """
    if params is not None:
        warn("`params` and `options` parameters will be removed in a future version. Use `order`, "
            "`gamma`, and `window_size` instead.", DeprecationWarning)
        order, gamma = params[:2]
        if len(params) > 2: window_size = params[2]
        if options is not None:
            if 'sliding' in options and not options['sliding']: window_size = None
            if 'step_size' in options: step_size = options['step_size']
            if 'kernel_name' in options: kernel = options['kernel_name']
            if 'solver' in options: solver = options['solver']
    elif order is None or gamma is None or window_size is None:
        raise ValueError("`order`, `gamma`, and `window_size` must be given.")

    if np.any(np.isnan(x)): raise ValueError("`x` may not contain NaN. CVXPY cannot form a problem with missing data.")
    if not np.isscalar(dt): raise ValueError("`dt` must be a scalar. The integrals of x are accumulated at a constant step.")

    @np.errstate(invalid='ignore', over='ignore') # cvxpy#3503: building a sum atom reduces over uninitialized memory
    def _lineardiff(x, dt, order, gamma, solver=None): # just to read a shape, so it warns when that memory holds garbage
        """Estimate the parameters for a system xdot = Ax, and use that to calculate the derivative"""
        mean = np.mean(x)
        x = x - mean

        # Work in nondimensional time tau = t/T, so the window spans [0, 1] no matter how wide it is. Each row of the
        # matrix below is one more integration than the row beneath it, so in raw time the rows differ by powers of the
        # window duration T: at dt=0.01 and window_size=11 that leaves cond(integral_X) above 1e3 and entries around
        # 1e-5, small enough that a solver's absolute tolerances stop meaning anything. Integrating in tau instead
        # holds every row at O(1) and cond near 30 regardless of window_size. A comes back in units of 1/tau, so the
        # derivative it reconstructs is d/dtau and gets divided by T at the end to return to d/dt.
        T = len(x)*dt
        dtau = dt/T

        # Generate the matrix of integrals of x, then integrate it once more for the right hand side
        X = [x]
        for n in range(1, order):
            X.append(utility.integrate_dxdt_hat(X[-1], dtau))
        X = np.vstack(X[::-1])
        integral_X = np.hstack((np.zeros((X.shape[0], 1)), scipy.integrate.cumulative_trapezoid(X, axis=1)))*dtau

        # Powers of nondimensional time, the shape the integration constants take after `order` integrations. B[1:] is
        # its own derivative, since differentiating tau^k/k! gives tau^(k-1)/(k-1)!, which is the next row down.
        N = X.shape[1]
        tau = np.arange(N)/N # dtau is 1/N, so tau spans [0, 1) whatever the window's width in real time
        B = np.vstack([tau**(order-n-1)/math.factorial(order-n-1) for n in range(order)])

        # Solve X = A*integral_X + C*B for A and C. Canonicalizing costs ~85% of a solve here and depends only on the
        # shapes, so the problem is built once per (order, window length) and re-solved with new data, worth about 7x.
        # gamma is a Parameter too, which keeps a cached problem valid across a whole `optimize` sweep rather than only
        # within one call. All of which requires the problem to be DPP.
        if (order, N) not in _PROBLEM_CACHE:
            if len(_PROBLEM_CACHE) >= 64: _PROBLEM_CACHE.clear() # bound what an optimizer's window sweep accumulates
            A_v = cvxpy.Variable((order, order)); C_v = cvxpy.Variable((order, order))
            X_p = cvxpy.Parameter((order, N)); Xdot_p = cvxpy.Parameter((order, N)); g_p = cvxpy.Parameter(nonneg=True)
            _PROBLEM_CACHE[(order, N)] = (cvxpy.Problem(cvxpy.Minimize(
                cvxpy.sum_squares(Xdot_p - (A_v @ X_p + cvxpy.vstack([C_v[i, :] @ B for i in range(order)]))) +
                g_p*cvxpy.sum(cvxpy.abs(C_v)) + 1e-6*cvxpy.sum(cvxpy.abs(A_v)))), # gammaA is small and fixed
                A_v, C_v, X_p, Xdot_p, g_p)                                       # gammaC is the knob
        prob, A_v, C_v, X_p, Xdot_p, g_p = _PROBLEM_CACHE[(order, N)]
        X_p.value = integral_X; Xdot_p.value = X; g_p.value = gamma

        # Tighter than CLARABEL's defaults, because an l1 penalty's kinks make the argmin jump when a coordinate crosses
        # zero, and a loosely converged iterate turns a last-bit input difference into a visible one. Costs nothing
        # measurable on problems this small, and is what lets rescaled input reproduce rescaled output. See #222
        A = C = None
        try:
            prob.solve(solver=solver, **({'tol_gap_abs':1e-12, 'tol_gap_rel':1e-12, 'tol_feas':1e-12}
                                          if solver == 'CLARABEL' else {}))
            A, C = A_v.value, C_v.value
        except cvxpy.error.SolverError: pass
        if A is None or C is None: # The solve failed, which happens on a few percent of parameter combinations and would
            # otherwise abort a whole optimize() sweep. Fall back to the unregularized fit, the gamma -> 0 limit of the
            # same model: every row of X is separately linear in that row of A and C, so one lstsq recovers both.
            coeffs = np.linalg.lstsq(np.vstack((integral_X, B)).T, X.T, rcond=None)[0]
            A, C = coeffs[:order].T, coeffs[order:].T

        # Differentiating the fit gives Xdot = A*X + C*dB, whose bottom row is the derivative of the data itself
        Xdot = A@X + (C[:, :order-1] @ B[1:] if order > 1 else 0)
        dxdt_hat = np.ravel(Xdot[-1, :])/T # undo the time scaling

        x_hat = utility.integrate_dxdt_hat(dxdt_hat, dt)
        x_hat = x_hat + utility.estimate_integration_constant(x+mean, x_hat)

        return x_hat, dxdt_hat

    if window_size:
        if window_size % 2 == 0:
            window_size += 1
            warn("Kernel window size should be odd. Added 1 to length.")
        if step_size is None: step_size = max(1, window_size//5) # a ratio, so cost tracks window width instead of
            # ballooning when the search picks a wide window. Keeps step_size out of the optimizer's search space.
        if step_size > window_size:
            step_size = window_size
            warn("`step_size` wider than `window_size` would skip samples between windows, reduced to match `window_size`")
        kern = {'gaussian':utility.gaussian_kernel, 'friedrichs':utility.friedrichs_kernel}[kernel](window_size)

    x = np.asarray(x, dtype=float)
    x_work = np.moveaxis(x, axis, 0) # differentiation axis to front
    shape = x_work.shape             # remember it to restore the input's dimensionality
    x_flat = x_work.reshape(shape[0], -1) # rest of the dims flattened into columns
    x_hat = np.empty_like(x_flat); dxdt_hat = np.empty_like(x_flat)

    for i in range(x_flat.shape[1]):
        v = x_flat[:, i]
        # gamma weighs an l1 penalty against a squared residual, so on its own it carries the data's units and a
        # rescaled input silently gets a different fidelity/prior balance. Divide by one scale for the whole vector,
        # not per window, which would fit windows against different scales and crossfade incommensurate pieces.
        scale = scipy.stats.median_abs_deviation(v, scale='normal') # robust like tvrdiff's, so outliers can't inflate
        if scale == 0: x_hat[:, i] = v; dxdt_hat[:, i] = 0.; continue # quietly weaken gamma. Constant vector -> known 0
        v = v/scale

        if not window_size:
            xh, dh = _lineardiff(v, dt, order, gamma, solver)
        else: # Slide over overlapping windows in each direction, then crossfade the two passes, weighting each by how
            # far into its own pass a sample sits, so neither dominates near an edge.
            forward, _ = utility.slide_function(_lineardiff, v, dt, kern, order, gamma, stride=step_size, solver=solver)
            backward, _ = utility.slide_function(_lineardiff, v[::-1], dt, kern, order, gamma, stride=step_size, solver=solver)

            w = np.arange(1, len(forward)+1)[::-1]
            w = np.pad(w, [0, len(v)-len(w)], mode='constant')
            norm = np.sum(np.vstack((w, w[::-1])), axis=0)

            forward = np.pad(forward, [0, len(v)-len(forward)], mode='constant')
            backward = np.pad(backward[::-1], [len(v)-len(backward), 0], mode='constant')
            xh, dh = finitediff(forward*w/norm + backward*w[::-1]/norm, dt) # defaults to second order

        x_hat[:, i] = xh*scale; dxdt_hat[:, i] = dh*scale

    return (np.moveaxis(x_hat.reshape(shape), 0, axis),
            np.moveaxis(dxdt_hat.reshape(shape), 0, axis))
