"""Import functions from the optimize module
"""
from ._optimize import optimize, suggest_method

__all__ = ['optimize', 'suggest_method'] # So these get treated as direct members of the module by sphinx
