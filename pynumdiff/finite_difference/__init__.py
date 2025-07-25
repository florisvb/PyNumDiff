"""This module implements some common finite difference schemes
"""
from ._finite_difference import first_order, second_order, fourth_order

__all__ = ['first_order', 'second_order', 'fourth_order'] # So these get treated as direct members of the module by sphinx
