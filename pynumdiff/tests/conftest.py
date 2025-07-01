"""Pytest configuration for pynumdiff tests"""
import pytest
from matplotlib import pyplot
from collections import defaultdict

def pytest_addoption(parser):
    parser.addoption("--plot", action="store_true", default=False) # whether to show plots
    parser.addoption("--bounds", action="store_true", default=False) # whether to print error bounds

@pytest.fixture(scope="session", autouse=True)
def store_plots(request):
    request.config.plots = defaultdict(lambda: pyplot.subplots(6, 2, figsize=(12,7))) # 6 is len(test_funcs_and_derivs)

def pytest_sessionfinish(session, exitstatus):
    if not hasattr(session.config, 'plots'): return
    for method,(fig,axes) in session.config.plots.items():
        fig.legend(*axes[-1, -1].get_legend_handles_labels(), loc='lower left', ncol=2)
        fig.suptitle(method.__name__)
        fig.tight_layout()
        pyplot.show()
