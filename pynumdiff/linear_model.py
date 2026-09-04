"""Fit a linear system model in a sliding window"""
from warnings import warn
import math, scipy
import numpy as np

from pynumdiff.utils import utility

try: import cvxpy
except ImportError: pass


_PROBLEM_CACHE = {} # (order, window length) -> a parametrized CVXPY problem, so identically shaped windows reuse it

def lineardiff(x, dt, order, gamma, window_size=None, stride=None, kernel='friedrichs', axis=0):
    """Fit a linear dynamical system to windows of the data, then differentiate that model.

    :param np.array[float] x: data to differentiate. May be multidimensional; see :code:`axis`.
    :param float dt: step size
    :param int>1 order: order of the ODE fit, the number of states in the linear system, how many times :code:`x` is integrated.
    :param float gamma: regularization term, in multiples of the data's own scale, so a given value means the same
            thing whatever the units.
    :param int window_size: number of samples in the sliding window, or number of average step sizes to use as window
            width if irregular sampling; if not given, no sliding
    :param int stride: step size for sliding. Defaults to :code:`window_size/5`, because what matters is overlap
            ratio, not an absolute stride: a fifth costs a few percent of accuracy against a much finer stride while
            running several times faster, and strides past about half the window degrade badly
    :param str kernel: name of kernel to use for weighting and smoothing windows ('gaussian' or 'friedrichs')
    :param int axis: axis along which to differentiate (default 0)

    :return: - **x_hat** (np.array) -- estimated (smoothed) x
             - **dxdt_hat** (np.array) -- estimated derivative of x
    """
    if np.any(np.isnan(x)): raise ValueError("`x` may not contain NaN. CVXPY cannot form a problem with missing data.")
    if order < 2: raise ValueError("`order` must be at least 2. Order 1 is just xdot = a*x, solved by scalar exponential.")
    if not np.isscalar(dt):
        if len(dt) != x.shape[axis]: raise ValueError("If `dt` is given as sample locations, it must be as long as `x` along `axis`.")
        if np.any(np.diff(dt) <= 0): raise ValueError("`dt` must be strictly increasing when given as sample locations.")
    if window_size:
        if window_size < 2*order: # a and c hold `order` unknowns each, so need at least as many pieces of info to set up well-posed cost
            window_size = 2*order + 1 - (2*order)%2
            warn(f"`window_size` must be at least 2*order={2*order} to determine the fit. Widened to {window_size}.")
        if window_size % 2 == 0: window_size += 1; warn("Kernel window size should be odd. Added 1 to length.")
        if stride is None: stride = max(1, window_size//5) # Keeps stride out of the optimizer's search space.
        if stride > window_size: stride = window_size; warn("`stride` wider than `window_size`, reduced to match")
        kernel = {'gaussian':utility.gaussian_kernel, 'friedrichs':utility.friedrichs_kernel}[kernel]

    @np.errstate(invalid='ignore', over='ignore') # cvxpy#3503: building a sum atom reduces over uninitialized memory
    def _lineardiff(x, dt, order, gamma): # just to read a shape, so it warns when that memory holds garbage
        """Fit X = A*integral_X + C*B, then differentiate it to Xdot = A*X + C*dB to get the derivative"""
        mean = np.mean(x)
        xc = x - mean # not in-place `-=`: slide_function operates on views, so don't modify underlying array

        # Work in nondimensional time tau = t/T, so the normalized window spans tau \in [0, 1]. Each row of the
        # matrix below is one more integration than the row beneath it, so in raw time the rows differ by powers of the
        # window duration T. Integrating in tau instead holds cond number reasonable. A is fit in units of 1/tau, so the
        # derivative it reconstructs is d/dtau and gets divided by T at the end to return to d/dt.
        # With sample locations in hand, tau is read off them instead of assumed uniform; every step below is a
        # quadrature or a pointwise basis evaluation, so neither cares whether the samples are evenly spaced.
        equispaced = np.isscalar(dt)
        T = len(xc)*dt if equispaced else dt[-1] - dt[0]
        tau = np.arange(len(xc))/len(xc) if equispaced else (dt - dt[0])/T # dtau is 1/N when uniform, so tau spans [0, 1)
        step = dt/T if equispaced else tau # what the integrals below accumulate against

        # Generate the matrix of integrals of x, then integrate it once more for the right hand side
        X = [xc]
        for n in range(1, order):
            X.append(utility.integrate_dxdt_hat(X[-1], step))
        X = np.vstack(X[::-1])
        integral_X = np.hstack((np.zeros((X.shape[0], 1)), scipy.integrate.cumulative_trapezoid(X, axis=1)))*step \
            if equispaced else np.hstack((np.zeros((X.shape[0], 1)), scipy.integrate.cumulative_trapezoid(X, x=tau, axis=1)))

        # Powers of nondimensional time, the shape the integration constants take after `order` integrations. B[1:] is
        # its own derivative, since differentiating tau^k/k! gives tau^(k-1)/(k-1)!, which is the next row down.
        N = X.shape[1]
        B = np.vstack([tau**(order-n-1)/math.factorial(order-n-1) for n in range(order)])

        # Only the bottom row of X = A*integral_X + C*B carries dynamics. integral_X[i] is bitwise identical to X[i-1],
        # both being a cumulative_trapezoid of the row beneath, so the upper rows are tautologies that a shift matrix
        # solves at exactly zero residual with zero constants. The objective is separable by row and the derivative
        # below reads only the bottom row, so fitting the full A and C solves for 2*order^2 unknowns to use 2*order of
        # them. Fit that one row directly instead: x = a*integral_X + c*B. See #223
        # Canonicalizing costs ~85% of a solve here and depends only on the shapes, so the problem is built once per
        # (order, window length) and re-solved with new data. gamma is a Parameter too, which keeps a cached problem
        # valid across a whole `optimize` sweep rather than only within one call. All of which requires DPP.
        if (order, N) not in _PROBLEM_CACHE:
            if len(_PROBLEM_CACHE) >= 64: _PROBLEM_CACHE.clear() # bound what an optimizer's window sweep accumulates
            a_v = cvxpy.Variable(order); c_v = cvxpy.Variable(order)
            iX_p = cvxpy.Parameter((order, N)); x_p = cvxpy.Parameter(N); g_p = cvxpy.Parameter(nonneg=True)
            B_p = cvxpy.Parameter((order, N)) # a Parameter, not the array itself: with sample locations B depends on
                # where the samples fall, so baking it in would let a cached (order, N) problem carry the wrong basis
            _PROBLEM_CACHE[(order, N)] = (cvxpy.Problem(cvxpy.Minimize(
                cvxpy.sum_squares(x_p - (a_v @ iX_p + c_v @ B_p)) +
                g_p*cvxpy.sum(cvxpy.abs(c_v)) + 1e-6*cvxpy.sum(cvxpy.abs(a_v)))), a_v, c_v, iX_p, x_p, g_p, B_p)
                # Smooth x has near-polynomial integrals, and B is polynomials, so a and c become interchangeable. 1e-6 on a's
                # norm enforces uniqueness while not significantly shrinking, which costs accuracy. See #223
        prob, a_v, c_v, iX_p, x_p, g_p, B_p = _PROBLEM_CACHE[(order, N)]
        iX_p.value = integral_X; x_p.value = X[-1]; g_p.value = gamma; B_p.value = B

        # Tighten CLARABEL's stop conditions, because its defaults of 1e-8 can cause failures against the equivariance
        # test, see #222. Also no warm start, for reproducibility.
        try: prob.solve(solver=cvxpy.CLARABEL, warm_start=False, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
        except cvxpy.error.SolverError as e: # Convert so `optimize` scores the point badly and moves on
            raise np.linalg.LinAlgError(f"CVXPY failed to fit the linear model on a window of {N} samples at order "
                f"{order}, gamma {gamma}. Try a wider `window_size` or a lower `order`.") from e

        # Differentiating the fit gives the derivative of the data itself, xdot = a*X + c*dB
        dxdt_hat = (a_v.value @ X + (c_v.value[:order-1] @ B[1:] if order > 1 else 0))/T # undo the time scaling

        x_hat = utility.integrate_dxdt_hat(dxdt_hat, dt)
        x_hat += utility.estimate_integration_constant(xc, x_hat) + mean

        return x_hat, dxdt_hat

    x_move = np.moveaxis(x, axis, 0); s = x_move.shape
    x_flat = x_move.reshape(s[0], -1) # big 2D matrix of all vecs we need to differentiate
    x_hat = np.empty(x_flat.shape, dtype=float); dxdt_hat = np.empty(x_flat.shape, dtype=float) # float explicitly, so inherited integer input type cannot silently truncate

    for i in range(x_flat.shape[1]):
        # gamma weighs an l1 penalty against a squared residual, so on its own it carries the data's units and a
        # rescaled input silently gets a different fidelity/prior balance. Divide by one scale for the whole vector,
        # not per window, which would fit windows against different scales and crossfade incommensurate pieces.
        scale = utility.robust_data_scale(x_flat[:, i]) # robust like tvrdiff's, so outliers can't inflate
        if scale == 0: x_hat[:, i] = x_flat[:, i]; dxdt_hat[:, i] = 0.; continue # constant vector -> known 0 deriv
        v = x_flat[:, i]/scale

        xh, dh = _lineardiff(v, dt, order, gamma) if not window_size else \
            utility.slide_function(_lineardiff, v, dt, kernel, window_size, stride, min_samples=2*order, order=order, gamma=gamma)
        x_hat[:, i] = xh*scale; dxdt_hat[:, i] = dh*scale

    return np.moveaxis(x_hat.reshape(s), 0, axis), np.moveaxis(dxdt_hat.reshape(s), 0, axis)
