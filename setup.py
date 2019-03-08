import os
from setuptools import setup, find_packages

here = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(here, "README.md")) as f:
    long_description = f.read()

setup(
    name='pynumdiff',
    version='0.0.1',
    author='Floris van Breugel',
    author_email='floris@caltech.edu',
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
    ],
    include_package_data=True,
    license='BSD',
    test_requires=["pytest", "scipy", "numpy"],
    description='Taking derivatives of noisy data',
    long_description=long_description,
    long_description_content_type="text/markdown"
)