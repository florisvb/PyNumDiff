"""Pytest configuration for pynumdiff tests. This is what enables the --plot and --bounds flags to work."""
from collections import defaultdict
from matplotlib import pyplot
import pytest

def pytest_addoption(parser):
    """Make some flags available to tests that take the :code:`request` argument"""
    parser.addoption("--plot", action="store_true", default=False) # whether to show plots
    parser.addoption("--bounds", action="store_true", default=False) # whether to print error bounds

@pytest.fixture(scope="session", autouse=True)
def store_plots(request):
    """This makes these plots available to tests that take the :code:`request` argument"""
    request.config.plots = defaultdict(lambda: pyplot.subplots(6, 2, figsize=(12,7))) # 6 is len(test_funcs_and_derivs)

def pytest_sessionfinish(session, exitstatus):
    """This is done at the end of the session, when tests are done running"""
    if not hasattr(session.config, 'plots'): return
    for method,(fig,axes) in session.config.plots.items():
        fig.legend(*axes[-1, -1].get_legend_handles_labels(), loc='lower left', ncol=2)
        fig.suptitle(method.__name__)
        fig.tight_layout()
    pyplot.show()
