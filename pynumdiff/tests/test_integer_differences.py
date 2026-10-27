"""Integer samples must not overflow before floating derivative scaling."""
import numpy as np
import pytest

from pynumdiff import finitediff


@pytest.mark.parametrize('dtype', [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64])
@pytest.mark.parametrize('order', [1, 2, 4])
@pytest.mark.parametrize('num_iterations', [1, 3])
def test_integer_finite_differences(dtype, order, num_iterations):
    """Check analytic quadratic derivatives and equivalent floating samples."""
    # The exact quadratic and its derivative fit in the input dtype, while
    # weighted stencil intermediates can overflow even on unsigned inputs.
    t = np.arange(11)
    signal = np.stack([t**2, (10-t)**2]).astype(dtype)
    original = signal.copy()
    _, derivative = finitediff(signal, .25, order=order, num_iterations=num_iterations, axis=-1)
    expected = finitediff(signal.astype(float), .25, order=order,
                          num_iterations=num_iterations, axis=-1)
    np.testing.assert_array_equal(derivative, expected[1])
    if num_iterations == 1 and order > 1:
        np.testing.assert_array_equal(derivative, np.stack([8*t, -8*(10-t)]))
    np.testing.assert_array_equal(signal, original)
