"""Fit a linear system model in a sliding window"""
from warnings import warn
import math, scipy
import numpy as np

from pynumdiff.finite_difference import finitediff
from pynumdiff.utils import utility

try: import cvxpy
except ImportError: pass


_PROBLEM_CACHE = {} # (order, window length) -> a parametrized CVXPY problem, so identically shaped windows reuse it

def lineardiff(x, dt, order, gamma, window_size=None, stride=None, kernel='friedrichs', axis=0):
    """Fit a linear dynamical system to windows of the data, then differentiate that model.

    :param np.array[float] x: data to differentiate. May be multidimensional; see :code:`axis`.
    :param float dt: step size
    :param int>0 order: number of states in the linear system, equivalently how many times :code:`x` is integrated
    :param float gamma: regularization term, in multiples of the data's own scale, so a given value means the same
            thing whatever the units.
    :param int window_size: size of the sliding window, if not given no sliding
    :param int stride: step size for sliding. Defaults to :code:`window_size/5`, because what matters is overlap
            ratio, not an absolute stride: a fifth costs a few percent of accuracy against a much finer stride while
            running several times faster, and strides past about half the window degrade badly
    :param str kernel: name of kernel to use for weighting and smoothing windows ('gaussian' or 'friedrichs')
    :param int axis: axis along which to differentiate (default 0)

    :return: - **x_hat** (np.array) -- estimated (smoothed) x
             - **dxdt_hat** (np.array) -- estimated derivative of x
    """
    if np.any(np.isnan(x)): raise ValueError("`x` may not contain NaN. CVXPY cannot form a problem with missing data.")
    if not np.isscalar(dt): raise ValueError("`dt` must be a scalar. The integrals of x are accumulated at a constant step.")
    if window_size:
        if window_size % 2 == 0: window_size += 1; warn("Kernel window size should be odd. Added 1 to length.")
        if stride is None: stride = max(1, window_size//5) # Keeps stride out of the optimizer's search space.
        if stride > window_size: stride = window_size; warn("`stride` wider than `window_size`, reduced to match")
        kern = {'gaussian':utility.gaussian_kernel, 'friedrichs':utility.friedrichs_kernel}[kernel](window_size)

    @np.errstate(invalid='ignore', over='ignore') # cvxpy#3503: building a sum atom reduces over uninitialized memory
    def _lineardiff(x, dt, order, gamma): # just to read a shape, so it warns when that memory holds garbage
        """Fit X = A*integral_X + C*B, then differentiate it to Xdot = A*X + C*dB to get the derivative"""
        mean = np.mean(x)
        x = x - mean

        # Work in nondimensional time tau = t/T, so the normalized window spans tau \in [0, 1]. Each row of the
        # matrix below is one more integration than the row beneath it, so in raw time the rows differ by powers of the
        # window duration T. Integrating in tau instead holds cond number reasonable. A is fit in units of 1/tau, so the
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
                g_p*cvxpy.sum(cvxpy.abs(C_v)) + 1e-6*cvxpy.sum(cvxpy.abs(A_v)))), A_v, C_v, X_p, Xdot_p, g_p)
                # Smooth x has near-polynomial integrals, and B is polynomials, so A and C become interchangeable. 1e-6 on A's
                # norm enforces uniqueness while not significantly shrinking, which costs accuracy. See #223
        prob, A_v, C_v, X_p, Xdot_p, g_p = _PROBLEM_CACHE[(order, N)]
        X_p.value = integral_X; Xdot_p.value = X; g_p.value = gamma

        # Tighten CLARABEL's stop conditions, because its defaults of 1e-8 can cause failures against the equivariance
        # test (#222). Also no warm start, for reproducibility.
        try: prob.solve(solver=cvxpy.CLARABEL, warm_start=False, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
        except cvxpy.error.SolverError as e: # Convert so `optimize` scores the point badly and moves on
            raise np.linalg.LinAlgError(f"CVXPY failed to fit the linear model on a window of {N} samples at order "
                f"{order}, gamma {gamma}. Try a wider `window_size` or a lower `order`.") from e

        # Differentiating the fit gives Xdot = A*X + C*dB, whose bottom row is the derivative of the data itself
        Xdot = A_v.value @ X + (C_v.value[:, :order-1] @ B[1:] if order > 1 else 0)
        dxdt_hat = np.ravel(Xdot[-1, :])/T # undo the time scaling

        x_hat = utility.integrate_dxdt_hat(dxdt_hat, dt)
        x_hat += utility.estimate_integration_constant(x+mean, x_hat)

        return x_hat, dxdt_hat

    x_work = np.moveaxis(x, axis, 0); s = x_work.shape
    x_flat = x_work.reshape(s[0], -1) # big 2D matrix of all vecs we need to differentiate
    x_hat = np.empty(x_flat.shape, dtype=float); dxdt_hat = np.empty(x_flat.shape, dtype=float) # float explicitly, so inherited integer input type cannot silently truncate

    for i in range(x_flat.shape[1]):
        # gamma weighs an l1 penalty against a squared residual, so on its own it carries the data's units and a
        # rescaled input silently gets a different fidelity/prior balance. Divide by one scale for the whole vector,
        # not per window, which would fit windows against different scales and crossfade incommensurate pieces.
        scale = utility.robust_data_scale(x_flat[:, i]) # robust like tvrdiff's, so outliers can't inflate
        if scale == 0: x_hat[:, i] = x_flat[:, i]; dxdt_hat[:, i] = 0.; continue # constant vector -> known 0 deriv
        v = x_flat[:, i]/scale

        if not window_size:
            xh, dh = _lineardiff(v, dt, order, gamma)
        else: # Slide over overlapping windows in each direction, then crossfade the two passes to avoid bias
            forward, _ = utility.slide_function(_lineardiff, v, dt, kern, order, gamma, stride=stride)
            backward, _ = utility.slide_function(_lineardiff, v[::-1], dt, kern, order, gamma, stride=stride)

            w = np.arange(1, len(forward)+1)[::-1]
            w = np.pad(w, [0, len(v)-len(w)], mode='constant')
            norm = np.sum(np.vstack((w, w[::-1])), axis=0)

            forward = np.pad(forward, [0, len(v)-len(forward)], mode='constant')
            backward = np.pad(backward[::-1], [len(v)-len(backward), 0], mode='constant')
            xh, dh = finitediff(forward*w/norm + backward*w[::-1]/norm, dt) # defaults to second order

        x_hat[:, i] = xh*scale; dxdt_hat[:, i] = dh*scale

    return np.moveaxis(x_hat.reshape(s), 0, axis), np.moveaxis(dxdt_hat.reshape(s), 0, axis)
