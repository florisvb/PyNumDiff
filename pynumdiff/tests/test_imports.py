
import pytest
import importlib

cvxpy_dependent_modules = ["", ".total_variation_regularization", ".kalman_smooth",
                            ".linear_model", ".optimize"]

@pytest.mark.skipif(importlib.util.find_spec("cvxpy"), reason="cvxpy installed")
@pytest.mark.parametrize("module", cvxpy_dependent_modules)
def test_import_without_cvxpy_warns(module):
    with pytest.warns(UserWarning):
        importlib.import_module("pynumdiff" + module)
