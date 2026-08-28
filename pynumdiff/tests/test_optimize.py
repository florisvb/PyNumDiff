"""Unit tests of optimizer"""
import warnings
from functools import partial
from multiprocessing import Manager, Pool
import numpy as np
from pytest import warns

from ..smooth_finite_difference import butterdiff
from ..polynomial_fit import splinediff
from ..kalman_smooth import rtsdiff
from ..optimize import optimize, _objective_function
from ..utils.simulate import pi_cruise_control
from ..utils.evaluate import rmse


dt = 0.01
x, x_truth, dxdt_truth = pi_cruise_control(duration=2, noise_type='normal', noise_parameters=[0, 0.01], dt=dt)
cutoff_frequency = 3 # in Hz
tvgamma = np.exp(-1.6 * np.log(cutoff_frequency) - 0.71 * np.log(dt) - 5.1)


def test_parallel_same_as_serial():
    """Ensure running optimize across several processes returns the same result as running in a single process"""
    params_parallel, val_parallel = optimize(rtsdiff, x, dt, tvgamma=tvgamma, parallel=True)
    params_serial, val_serial = optimize(rtsdiff, x, dt, tvgamma=tvgamma, parallel=False)

    assert np.allclose(val_serial, val_parallel)
    assert np.allclose([params_serial[k] for k in params_serial], [params_parallel[k] for k in params_serial])


def test_targeting_rmse_vs_tvgamma_loss():
    """Ensure optimization properly targets different metrics"""
    params_rmse, val_rmse = optimize(splinediff, x, dt, dxdt_truth=dxdt_truth, parallel=False) # so coverage picks it up, because multiprocessing coverage is broken
    params_loss, val_loss = optimize(splinediff, x, dt, tvgamma=tvgamma)

    x_hat, dxdt_hat = splinediff(x, dt, **params_loss)
    loss_rmse = rmse(dxdt_truth, dxdt_hat)
    # This bound might break if using a different diff method or data series, but the point is they are ballpark similar.
    assert val_rmse <= loss_rmse < 2.5*val_rmse # Claude measures 1.4-2.7x across methods.


def test_search_space_updates_applied():
    """Ensure search space updates are used in optimization"""
    params2, _ = optimize(butterdiff, x, dt, search_space_updates={'filter_order':2}, tvgamma=tvgamma)
    params3, _ = optimize(butterdiff, x, dt, search_space_updates={'filter_order':3}, tvgamma=tvgamma)

    assert params2['filter_order'] == 2
    assert params3['filter_order'] == 3


def test_warning_filters_restored():
    """Ensure the UserWarning silencing inside optimize() doesn't leak out and mute the caller's warnings"""
    before = list(warnings.filters) # a copy, because filterwarnings() mutates this list in place
    optimize(butterdiff, x, dt, tvgamma=tvgamma, parallel=False, maxiter=1,
        search_space_updates={'filter_order':2, 'cutoff_freq':[0.1], 'num_iterations':1})

    assert warnings.filters == before


def test_search_space_with_no_dimensions():
    """Ensure pinning every parameter scores that lone point instead of handing scipy an empty one"""
    pinned = {'filter_order':2, 'cutoff_freq':0.1, 'num_iterations':1}
    with warns(UserWarning, match="Nothing to optimize"):
        opt_params, val = optimize(butterdiff, x, dt, dxdt_truth=dxdt_truth, parallel=False, search_space_updates=pinned)

    assert opt_params == pinned # nothing to search, so they should come back untouched
    assert np.isclose(val, rmse(dxdt_truth, butterdiff(x, dt, **pinned)[1]))


def test_categorical_only_search_space():
    """Ensure categoricals still get compared when there are no numerical dimensions left to search"""
    fixed = {'cutoff_freq':0.1, 'num_iterations':1}
    opt_params, val = optimize(butterdiff, x, dt, dxdt_truth=dxdt_truth, parallel=False,
        search_space_updates={'filter_order':{2, 3}, **fixed})

    scores = {order:rmse(dxdt_truth, butterdiff(x, dt, filter_order=order, **fixed)[1]) for order in [2, 3]}
    assert opt_params['filter_order'] == min(scores, key=scores.get) # the better of the two, not just either one
    assert np.isclose(val, min(scores.values()))


def test_cache_key_collides_across_processes():
    """The cache is shared between workers. To avoid duplicate work, a key must collide for the same parameters no matter which
    process built it. Getting this wrong is silent, because every answer stays correct, the sweep just redoes work it already has."""
    with Manager() as manager:
        cache = manager.dict()
        # a partial of a module-level function pickles fine, which is how optimize() ships jobs to its own workers
        evaluate = partial(_objective_function, func=splinediff, x=x, dt=dt, singleton_params={'num_iterations':1},
            categorical_params={'degree':3}, roundings={'s':float}, dxdt_truth=dxdt_truth, metric='rmse',
            tvgamma=None, padding=0, cache=cache, huberM=6)
        # each Pool is terminated before the next is built, so these two evaluations run in different processes
        with Pool(1) as pool: val1 = pool.map(evaluate, [[0.9]])[0]
        with Pool(1) as pool: val2 = pool.map(evaluate, [[0.9]])[0]

        assert val1 == val2 # same value
        assert len(cache) == 1 # the second process found the first one's entry rather than adding its own
