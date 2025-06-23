"""Pytest configuration for pynumdiff tests"""
import pytest

def pytest_addoption(parser): parser.addoption("--plot", action="store_true", default=False) 