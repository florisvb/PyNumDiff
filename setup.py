from distutils.core import setup

setup(
    name='PyNumDiff',
    version='0.0.1',
    author='Floris van Breugel',
    author_email='fvanbreugel@unr.edu',
    packages = ['pynumdiff'],
    license='BSD',
    description='Numerical derivatives of noisy data',
    long_description=open('README.md').read(),
)