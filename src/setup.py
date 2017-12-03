#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:19:41 2017

@author: matthewxfz
"""

from distutils.core import setup
from Cython.Build import cythonize
import numpy


setup(
    ext_modules = cythonize("util.pyx"),
    include_dirs=[numpy.get_include()]
)