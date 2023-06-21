#!/usr/bin/env python
import os

import numpy
import pkg_resources
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

exts = [
    Extension(
        name="diff_ttsg.utils.monotonic_align.core",
        sources=["diff_ttsg/utils/monotonic_align/core.pyx"],
    )
]

setup(
    name="diff_ttsg",
    version="0.0.1",
    description="Denoising probabilistic integrated speech and gesture synthesis",
    author="Shivam Mehta",
    author_email="shivam.mehta25@gmail.com",
    url="https://shivammehta25.github.io/Diff-TTSG/",
    install_requires=[
            str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )   
        ],
    include_dirs=[numpy.get_include()],
    packages=find_packages(exclude=["tests", "tests/*", "examples", "examples/*"]),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
        ]
    },
    ext_modules=cythonize(exts, language_level=3),
)
