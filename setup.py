#!/usr/bin/env python
# encoding: utf-8

import os
from setuptools import setup


def read(fname):
    """Utility function to read the README file."""
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read()


setup(
    name="nphelper",
    description="nphelper - convenient numpy helper functions",
    long_description=read("README.rst"),
    author="Stefan Otte",
    url="https://github.com/sotte/nphelper",
    download_url="https://github.com/sotte/nphelper",
    author_email="stefan.otte@gmail.com",
    version="0.0.5",
    install_requires=["numpy", ],
    packages=["nphelper"],
    license="MIT"
)
