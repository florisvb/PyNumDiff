"""Fit a linear system model in a sliding window"""
from warnings import warn
import math, scipy
import numpy as np

from pynumdiff.utils import utility

try: import cvxpy
except ImportError: pass


_PROBLEM_CACHE = {} # (order, window length) -> a parametrized CVXPY problem, so identically shaped windows reuse it

def lineardiff(x, dt_or_t, order, gamma, window_size=None, stride=None, kernel='friedrichs', axis=0):
    """Fit a linear dynamical system to windows of the data, then differentiate that model.

    :param np.array[float] x: data to differentiate. May be multidimensional; see :code:`axis`.
    :param float or array[float] dt_or_t: This function supports variable step size. This parameter is either the constant
        :math:`\\Delta t` if given as a single float, or data locations if given as an array of same length as :code:`x`.
    :param int>0 order: order of the ODE fit, the number of states in the linear system, how many times :code:`x` is integrated.
    :param float gamma: regularization term, in multiples of the data's own scale, so a given value means the same
            thing whatever the units.
    :param int window_size: number of samples in the sliding window, or number of average step sizes to use as window
            width if irregular sampling; if not given, no sliding
    :param int stride: step size for sliding. Defaults to :code:`window_size//5`, which costs only a few percent of accuracy
            against a much finer stride while running reasonably fast; strides > half the window degrade performance badly
    :param str kernel: name of kernel to use for weighting and smoothing windows ('gaussian' or 'friedrichs')
    :param int axis: axis along which to differentiate (default 0)

    :return: - **x_hat** (np.array) -- estimated (smoothed) x
             - **dxdt_hat** (np.array) -- estimated derivative of x
    """
    if not np.isscalar(dt_or_t):
        if len(dt_or_t) != x.shape[axis]: raise ValueError("If `dt_or_t` is given as array-like, must have same length as `x`.")
        if np.any(np.diff(dt_or_t) <= 0): raise ValueError("`dt_or_t` must be strictly increasing, so integration has all positive widths.")
    if window_size:
        if window_size < 2*order: # a and c hold `order` unknowns each, so need at least as many pieces of info to set up well-posed cost
            window_size = 2*order + 1 - (2*order)%2
            warn(f"`window_size` must be at least 2*order={2*order} to determine the fit. Widened to {window_size}.")
        if window_size % 2 == 0: window_size += 1; warn("Kernel window size should be odd. Added 1 to length.")
        if stride is None: stride = max(1, window_size//5) # Keeps stride out of the optimizer's search space.
        if stride > window_size: stride = window_size; warn("`stride` wider than `window_size`, reduced to match")
        kernel = {'gaussian':utility.gaussian_kernel, 'friedrichs':utility.friedrichs_kernel}[kernel]

    @np.errstate(invalid='ignore', over='ignore') # cvxpy#3503: building a sum atom reduces over uninitialized memory
    # TODO fixed upstream by cvxpy#3512, merged to master 2026-09-06 but not in any release through v1.9.2;
    # when it ships, drop the line above and floor cvxpy there.
    def _lineardiff(x, dt_or_t, order, gamma): # just to read a shape, so it warns when that memory holds garbage
        """Fit x = a.T*integral_Y + c.T*B, then differentiate it to xdot = a.T*Y + c.T*dB to get the derivative"""
        obs = ~np.isnan(x) # Missing values drop out of the fit, and the fitted model imputes them back
        if obs.sum() < 2*order: raise ValueError(f"Window encountered with only {obs.sum()} non-NaN samples < {2*order} "
                f"needed to determine an order {order} fit. Widen `window_size` or lower `order`.")
        # Uncentered, Y integrals essentially carries a copy of B, so make a distinction. But at order 1 both B (a lone row
        # of 1s) and Y integrals are orthogonal to the centered signal, so the fit returns a = 0
        mu = np.nanmean(x) if order > 1 else 0
        y = x - mu # slide_function operates on views, so don't modify the underlying array

        # Work in tau = t/T \in [0, 1], because rows of integral_Y and B differ by powers of T in original units,
        # potentially wrecking conditioning. A then reconstructs d/dtau, and we can use chain rule to get back to d/dt. 
        equispaced = np.isscalar(dt_or_t)
        T = (len(y)-1)*dt_or_t if equispaced else dt_or_t[-1] - dt_or_t[0] # the window's span
        tau = np.arange(len(y))/(len(y)-1) if equispaced else (dt_or_t - dt_or_t[0])/T
        # Generate the matrix of integrals. Bridge gaps with interpolation, which matches trapezoid rule anyway.
        Y_integrals = [y if obs.all() else np.interp(tau, tau[obs], y[obs])]
        for n in range(order): Y_integrals.append(utility.integrate_dxdt_hat(Y_integrals[-1], tau)) # indifferent to spacing!
        Y_integrals = np.vstack(Y_integrals[::-1]) # fit and its derivative use mostly the same rows, differing just at ends

        # B holds integration constants after `order` integrations. \dot{B} = B[1:], since d/dtau tau^k/k! = tau^(k-1)/(k-1)!
        N = Y_integrals.shape[1]
        B = np.vstack([tau**(order-1-j)/math.factorial(order-1-j) for j in range(order)])

        # Canonicalizing costs ~85% of a solve and depends only on variable shapes, so build the problem once per
        # (order, len(y)) with DPP and re-solve with new data. Make gamma a Parameter so it can be filled with different values.
        if (order, N) not in _PROBLEM_CACHE:
            if len(_PROBLEM_CACHE) >= 64: _PROBLEM_CACHE.clear() # bound what an optimizer's window sweep accumulates
            a_v = cvxpy.Variable(order); c_v = cvxpy.Variable(order)
            iY_p = cvxpy.Parameter((order, N)); y_p = cvxpy.Parameter(N); g_p = cvxpy.Parameter(nonneg=True)
            B_p = cvxpy.Parameter((order, N)) # parameterized because variable sample locations or missing values
                # make the problem's sampling of B vary, so baking in could make cached (order, N) carry the wrong thing.
            _PROBLEM_CACHE[(order, N)] = (cvxpy.Problem(cvxpy.Minimize(
                cvxpy.sum_squares(y_p - (a_v @ iY_p + c_v @ B_p)) +
                g_p*cvxpy.sum(cvxpy.abs(c_v)) + 1e-6*cvxpy.sum(cvxpy.abs(a_v)))), a_v, c_v, iY_p, y_p, g_p, B_p)
                # Smooth x has near-polynomial integrals, and B is polynomials, so a and c become interchangeable. 1e-6 on
                # a's norm enforces uniqueness while not significantly biasing. See #223
        prob, a_v, c_v, iY_p, y_p, g_p, B_p = _PROBLEM_CACHE[(order, N)]
        # Zero out missing data locations with *obs, rather than indexing, because resid at missing locations becomes
        # (0 - (a*0 + c*0))^2 = 0, influencing the solve not a whit, as desired. Allows reuse of same-shape canonicalization.
        iY_p.value = Y_integrals[:-1]*obs; y_p.value = Y_integrals[-1]*obs; g_p.value = gamma; B_p.value = B*obs

        # Tighten CLARABEL's stop conditions, because its defaults of 1e-8 can cause failures against the equivariance
        # test, see #222. Also no warm start, for reproducibility.
        try: prob.solve(solver=cvxpy.CLARABEL, warm_start=False, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
        except cvxpy.error.SolverError as e: # Convert so `optimize` scores the point badly and moves on
            raise np.linalg.LinAlgError(f"CVXPY failed to fit the linear model on a window of {N} samples at order "
                f"{order}, gamma {gamma}. Try a wider `window_size` or a lower `order`.") from e

        x_hat = a_v.value @ Y_integrals[:-1] + c_v.value @ B + mu
        dxdt_hat = (a_v.value @ Y_integrals[1:] + c_v.value[:order-1] @ B[1:])/T # undo the time scaling

        return x_hat, dxdt_hat

    x_flat = np.moveaxis(x, axis, 0); s = x_flat.shape; x_flat = x_flat.reshape(s[0], -1) # big 2D matrix of all vecs to differentiate
    x_hat = np.empty(x_flat.shape); dxdt_hat = np.empty(x_flat.shape)

    for i in range(x_flat.shape[1]):
        # Normalizes the whole vector, not per window, for commeasurability and so gamma means the same regardless of data scale
        sigma = utility.robust_data_scale(x_flat[:, i]) # robust like tvrdiff's, so outliers can't inflate
        if sigma == 0: x_hat[:, i] = x_flat[:, i]; dxdt_hat[:, i] = 0.; continue # constant vector -> known 0 deriv
        y = x_flat[:, i]/sigma

        xh, dh = _lineardiff(y, dt_or_t, order, gamma) if not window_size else \
            utility.slide_function(_lineardiff, y, dt_or_t, kernel, window_size, stride, min_samples=2*order, order=order, gamma=gamma)
        x_hat[:, i] = xh*sigma; dxdt_hat[:, i] = dh*sigma

    return np.moveaxis(x_hat.reshape(s), 0, axis), np.moveaxis(dxdt_hat.reshape(s), 0, axis)
