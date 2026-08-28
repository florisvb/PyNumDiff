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

def _basis(order, N):
    """Powers of nondimensional time, the shape the integration constants take after being integrated `order` times."""
    tau = np.arange(N)/N # dtau is 1/N, so tau spans [0, 1) whatever the window's width in real time
    return np.vstack([tau**(order-n-1)/math.factorial(order-n-1) for n in range(order)])

@np.errstate(invalid='ignore', over='ignore') # cvxpy#3503: canonicalizing norm1/huber/tv builds sum atoms, which reduce over uninitialized
#  memory just to read off a shape, so they warn when it holds garbage. This wall can come down if they ever fix it upstream.
def _cached_problem(order, N):
    """Build, or reuse, the convex program for one window. Canonicalizing costs ~85% of a solve here, and depends only on
    the shapes, so hoisting it out of the window loop is worth about 7x. `gamma` is a Parameter too, which keeps a cached
    problem valid across a whole `optimize` sweep rather than only within one call. Requires the problem to be DPP."""
    if (order, N) not in _PROBLEM_CACHE:
        if len(_PROBLEM_CACHE) >= 64: _PROBLEM_CACHE.clear() # bound what an optimizer's window sweep can accumulate
        A = cvxpy.Variable((order, order)); C = cvxpy.Variable((order, order))
        X = cvxpy.Parameter((order, N)); Xdot = cvxpy.Parameter((order, N)); gamma = cvxpy.Parameter(nonneg=True)
        Csum = cvxpy.vstack([C[i, :] @ _basis(order, N) for i in range(order)])
        prob = cvxpy.Problem(cvxpy.Minimize(cvxpy.sum_squares(Xdot - (A @ X + Csum)) +
            gamma*cvxpy.sum(cvxpy.abs(C)) + 1e-6*cvxpy.sum(cvxpy.abs(A)))) # gammaA is small and fixed; gammaC is the knob
        _PROBLEM_CACHE[(order, N)] = (prob, A, C, X, Xdot, gamma)
    return _PROBLEM_CACHE[(order, N)]

@np.errstate(invalid='ignore', over='ignore') # see the note on _cached_problem
def _solve_for_A_and_C_given_X_and_Xdot(X, Xdot, order, gamma, solver='CLARABEL'):
    """Given state and the derivative, find the system evolution and measurement matrices."""
    prob, A, C, X_p, Xdot_p, gamma_p = _cached_problem(order, X.shape[1])
    X_p.value = X; Xdot_p.value = Xdot; gamma_p.value = gamma
    # Tighter than CLARABEL's defaults, because an l1 penalty's kinks make the argmin jump when a coordinate crosses
    # zero, and a loosely converged iterate turns a last-bit input difference into a visible one. Costs nothing
    # measurable on problems this small, and is what lets rescaled input reproduce rescaled output. See #222
    opts = {'tol_gap_abs':1e-12, 'tol_gap_rel':1e-12, 'tol_feas':1e-12} if solver == 'CLARABEL' else {}
    try:
        prob.solve(solver=solver, **opts)
        if A.value is not None and C.value is not None: return A.value, C.value
    except cvxpy.error.SolverError: pass
    # The solve failed, which happens on a few percent of parameter combinations and would otherwise abort a whole
    # optimize() sweep. Fall back to the unregularized fit, the gamma -> 0 limit of the same model: every row of Xdot
    # is separately linear in that row of A and C, so one lstsq against [X; basis] recovers both.
    coeffs = np.linalg.lstsq(np.vstack((X, _basis(order, X.shape[1]))).T, Xdot.T, rcond=None)[0]
    return coeffs[:order].T, coeffs[order:].T

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

    def _lineardiff(x, dt, order, gamma, solver=None):
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

        # Generate the matrix of integrals of x
        X = [x]
        for n in range(1, order):
            X.append(utility.integrate_dxdt_hat(X[-1], dtau))
        X = np.vstack(X[::-1])
        integral_Xdot = X
        integral_X = np.hstack((np.zeros((X.shape[0], 1)), scipy.integrate.cumulative_trapezoid(X, axis=1)))*dtau

        # Solve for A and the integration constants
        A, C = _solve_for_A_and_C_given_X_and_Xdot(integral_X, integral_Xdot, order, gamma, solver=solver)

        # Add the integration constants, differentiated once because this reconstructs Xdot rather than integral_Xdot
        Csum = C[:, :order-1] @ _basis(order-1, X.shape[1]) if order > 1 else 0

        # Use A and C to calculate the derivative, then undo the time scaling
        Xdot_reconstructed = A@X + Csum
        dxdt_hat = np.ravel(Xdot_reconstructed[-1, :])/T

        x_hat = utility.integrate_dxdt_hat(dxdt_hat, dt)
        x_hat = x_hat + utility.estimate_integration_constant(x+mean, x_hat)

        return x_hat, dxdt_hat

    def _slide(x, dt, kern):
        """Run _lineardiff over overlapping windows in both directions and crossfade the two passes together."""
        x_hat_forward, _ = utility.slide_function(_lineardiff, x, dt, kern, order, gamma, stride=step_size, solver=solver)
        x_hat_backward, _ = utility.slide_function(_lineardiff, x[::-1], dt, kern, order, gamma, stride=step_size, solver=solver)

        # weights
        w = np.arange(1, len(x_hat_forward)+1)[::-1]
        w = np.pad(w, [0, len(x)-len(w)], mode='constant')
        wfb = np.vstack((w, w[::-1]))
        norm = np.sum(wfb, axis=0)

        # orient and pad
        x_hat_forward = np.pad(x_hat_forward, [0, len(x)-len(x_hat_forward)], mode='constant')
        x_hat_backward = np.pad(x_hat_backward[::-1], [len(x)-len(x_hat_backward), 0], mode='constant')

        # merge
        x_hat = x_hat_forward*w/norm + x_hat_backward*w[::-1]/norm
        return finitediff(x_hat, dt) # defaults to second order

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
        scale = np.std(v)
        if scale == 0: x_hat[:, i] = v; dxdt_hat[:, i] = 0.; continue # a constant vector's derivative is known
        v = v/scale
        xh, dh = _slide(v, dt, kern) if window_size else _lineardiff(v, dt, order, gamma, solver)
        x_hat[:, i] = xh*scale; dxdt_hat[:, i] = dh*scale

    return (np.moveaxis(x_hat.reshape(shape), 0, axis),
            np.moveaxis(dxdt_hat.reshape(shape), 0, axis))
