"""Installation script for the ``moma_lab`` Python package."""

from setuptools import find_packages, setup

setup(
    name="moma_lab",
    version="0.1.0",
    description="Focused Unitree Go2W locomotion extension for Isaac Lab",
    packages=find_packages(),
    include_package_data=True,
    package_data={"moma_lab": ["data/**/*"]},
    install_requires=["gymnasium"],
    python_requires=">=3.10",
    zip_safe=False,
)

