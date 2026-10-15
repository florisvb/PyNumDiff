"""Equivalent positive and negative axes must select the same signal vectors."""
import numpy as np
import pytest

from pynumdiff import splinediff, polydiff, rtsdiff, robustdiff, tvrdiff


@pytest.mark.parametrize('method,kwargs', [
    (splinediff, {'s': 0}),
    (polydiff, {'degree': 2, 'window_size': 7}),
    (rtsdiff, {'order': 2, 'log_qr_ratio': 3}),
    (robustdiff, {'order': 2, 'log_q': 3, 'log_r': 0}),
    (tvrdiff, {'order': 2, 'gamma': .1}),
])
@pytest.mark.parametrize('axis', [0, 1, 2])
def test_negative_signal_axes(method, kwargs, axis):
    """Check the same signals along both signs of each valid dimension."""
    t = np.linspace(0, 1, 15)
    x = np.stack([t**2, 2*t + 1, np.sin(t), np.cos(t)]).reshape(2, 2, -1)
    x = np.moveaxis(x, -1, axis)
    original = x.copy()
    expected = method(x, .1, axis=axis, **kwargs)
    actual = method(x, .1, axis=axis-x.ndim, **kwargs)
    for value, reference in zip(actual, expected):
        np.testing.assert_array_equal(value, reference)
    np.testing.assert_array_equal(x, original)
    for invalid in [-x.ndim-1, x.ndim]:
        with pytest.raises(IndexError):
            method(x, .1, axis=invalid, **kwargs)
