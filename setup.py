import sys
from distutils.core import setup

import numpy
from setuptools import find_packages

"""
To re-setup: python setup.py bdist_wheel

pip install -r requirements.txt --process-dependency-links

To test on test pypi:
    
    twine upload --repository testpypi dist/*
    
"""

PACKAGE_NAME = "eegio"
VERSION = "0.1"
DESCRIPTION = "An io package for eeg data and time-series transformations done on it."
URL = "https://github.com/adam2392/eegio"
MINIMUM_PYTHON_VERSION = 3, 6  # Minimum of Python 3.4
REQUIRED_PACKAGES = [
    "numpy>=1.14.5",
    "scipy>=1.1.0",
    "scikit-learn>=0.19.2",
    "pandas>=0.23.4",
    "mne>=0.18.2",
    "pyEDFlib == 0.1.14",
    "xlrd >= 1.2.0",
]
CLASSIFICATION_OF_PACKAGE = [
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    "Development Status :: 3 - Alpha",
    # Indicate who your project is intended for
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Medical Science Apps.",
    "Topic :: Scientific/Engineering",
    "Topic :: Utilities",
    # Pick your license as you wish (should match "license" above)
    "License :: OSI Approved :: GNU General Public License (GPL)",
    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: Implementation",
    "Natural Language :: English",
]


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


check_python_version()

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    author="Adam Li",
    long_description=open("README.md").read(),
    url=URL,
    license="GNU General Public License (GPL)",
    keywords="EEG, epilepsy, research tools, IO tools",
    packages=find_packages(include=[], exclude=[]),
    project_urls={
        "Documentation": "https://github.com/adam2392/eegio/docs/",
        "Source": URL,
        "Tracker": "https://github.com/adam2392/eegio/issues",
    },
    include_dirs=[numpy.get_include()],
    dependency_links=["git+https://github.com/adam2392/eegio#egg=eegio"],
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    classifiers=CLASSIFICATION_OF_PACKAGE,
    zip_ok=False,
)
