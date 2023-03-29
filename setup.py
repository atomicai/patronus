"""Setups simple package."""

import pkg_resources
import setuptools
from setuptools import setup

extras_require = dict(
    format=["isort==5.12.0", "black==23.1.0"],
    test=[
        "pytest",
        "pytest-sugar",  # For nicer look and feel
        "pytest-faker",  # For faker generator fixture
        # For running only subset of tests for changed files
        # Currently, testmon doesn't seem to work with xdist.
        # https://github.com/tarpas/pytest-testmon/issues/42
        "pytest-testmon==1.1.0",
        "pytest-custom-exit-code",  # For `--suppress-no-test-exit-code` option
    ],
)
extras_require["dev"] = sum((extras_require[k] for k in ["format", "test"]), [])
extras_require["all"] = sum(extras_require.values(), [])

setup(
    name="patronus",
    version="1.0.0",
    install_requires=list(map(str, pkg_resources.parse_requirements(open("requirements.txt")))),
    extras_require=extras_require,
    python_requires=">=3.9",
    packages=setuptools.find_packages(),
    include_package_data=True,
)
