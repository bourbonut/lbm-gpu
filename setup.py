# !/usr/bin/python3
from setuptools import setup

setup(
    name="lbm-gpu",
    python_requires=">=3.8.5",
    install_requires=[
        "cmapy>=0.6.6",
        "opencv-python>=4.5.5.64",
        "numpy>=1.21.6",
        "numba>=0.55.1",
        "cupy-cuda116>=10.4.0",
    ],
    license="GNU LGPL v3",
    author="Benjamin Bourbon",
    author_email="ben.bourbon06@gmail.com",
    description="The Lattice Boltzmann Method on GPU",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/bourbonut/lbm-gpu",
)
