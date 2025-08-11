"""This module implements some common finite difference schemes
"""
from ._finite_difference import finite_difference, first_order, second_order, fourth_order

__all__ = ['finite_difference', 'first_order', 'second_order', 'fourth_order'] # So these get treated as direct members of the module by sphinx
