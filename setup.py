#!/usr/bin/env python
import os, sys
import shutil
import datetime

from setuptools import setup, find_packages
from setuptools.command.install import install

# readme = open('README.md').read()
readme = """
# OFA²: Train one network, Search once, Deploy in many scenarios [[arXiv]](https://arxiv.org/abs/2303.13683)
```BibTex
@misc{ito2023ofa2,
      title={OFA²: A Multi-Objective Perspective for the Once-for-All Neural Architecture Search},
      author={Rafael C. Ito and Fernando J. Von Zuben},
      year={2023},
      eprint={2303.13683},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
}
```

## Check our [GitHub](https://github.com/ito-rafael/once-for-all-2) for more details.
"""
VERSION = "0.1.0"

requirements = [
    "torch",
]

# print(VERSION)

setup(
    # Metadata
    name="ofa2",
    version=VERSION,
    author="Rafael Ito",
    author_email="ito.rafael@gmail.com",
    url="https://github.com/ito-rafael/once-for-all-2",
    description="Train one network, Search once, Deploy in many scenarios.",
    long_description=readme,
    long_description_content_type="text/markdown",
    license="MIT",
    # Package info
    packages=find_packages(exclude=("*test*",)),
    #
    zip_safe=True,
    install_requires=requirements,
    # Classifiers
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
