import os
from setuptools import setup, find_packages
PACKAGES = find_packages()

# Get version and release info, which is all stored in pynumdiff/__version__.py
ver_file = os.path.join('pynumdiff', '__version__.py')
with open(ver_file) as f:
    exec(f.read())

opts = dict(name="pynumdiff",
            maintainer="Floris van Breugel",
            maintainer_email="fvanbreugel@unr.edu",
            description="Numerical derivatives of noisy data",
            long_description="Numerical derivatives of noisy data",
            license="MIT License",
            url="https://github.com/florisvb/PyNumDiff",
            download_url="https://github.com/florisvb/PyNumDiff/archive/0.0.3.tar.gz",
            classifiers=CLASSIFIERS,
            author="Floris van Breugel",
            author_email="fvanbreugel@unr.edu",
            platforms=PLATFORMS,
            version="0.0.3",
            packages=PACKAGES,
            install_requires=["numpy",
                              "matplotlib",
                              "scipy"])


if __name__ == '__main__':
    setup(**opts)
