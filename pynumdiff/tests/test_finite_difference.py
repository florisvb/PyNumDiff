"""
Unit tests for finite difference methods
"""
# pylint: skip-file
import numpy as np
from unittest import TestCase
from pynumdiff import first_order, second_order


class TestFD(TestCase):
    def test_first_order_1(self):
        x = np.array([1, 2, 3, 4, 5])
        dt = 0.01
        dxdt = np.array([100, 100, 100, 100, 100])
        _, dxdt_hat = first_order(x, dt)
        np.testing.assert_array_equal(dxdt_hat, dxdt)

    def test_first_order_2(self):
        x = np.array([8, 3, 14, 0, 9])
        dt = 0.01
        dxdt = np.array([-500, 300, -150, -250, 900])
        _, dxdt_hat = first_order(x, dt)
        np.testing.assert_array_equal(dxdt_hat, dxdt)

    def test_first_order_iterative(self):
        x = np.random.rand(100)
        dt = 0.01
        params = [100]
        _, dxdt_hat = first_order(x, dt, params, options={'iterate': True})
        assert x.shape == dxdt_hat.shape

    def test_second_order_1(self):
        x = np.array([1, 2, 3, 4, 5])
        dt = 0.01
        dxdt = np.array([100, 100, 100, 100, 100])
        _, dxdt_hat = second_order(x, dt)
        np.testing.assert_array_equal(dxdt_hat, dxdt)

    def test_second_order_2(self):
        x = np.array([8, 3, 14, 0, 9])
        dt = 0.01
        dxdt = np.array([-1300, 300, -150, -250, 2050])
        _, dxdt_hat = second_order(x, dt)
        np.testing.assert_array_equal(dxdt_hat, dxdt)
