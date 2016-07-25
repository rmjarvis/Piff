# Copyright (c) 2016 by Mike Jarvis and the other collaborators on GitHub at
# https://github.com/rmjarvis/Piff  All rights reserved.
#
# Piff is free software: Redistribution and use in source and binary forms
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.

"""
Piff: PSFs in the Full FOV

https://github.com/rmjarvis/Piff

Copyright (c) 2016 by Mike Jarvis and the other collaborators on GitHub at
https://github.com/rmjarvis/Piff  All rights reserved.

Piff is free software: Redistribution and use in source and binary forms
with or without modification, are permitted provided that the following
conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the disclaimer given in the accompanying LICENSE
   file.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer and/or other materials
   provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

# The version is stored in _version.py as recommended here:
# http://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package
from _version import __version__, __version_info__

# Also let piff.version show the version.
version = __version__

# We don't have any C functions, but once we do, I recommend using cffi to
# wrap them.  This is the entire code we need to get C functions into Python.
if False:
    import os,cffi,glob
    # Set module level attributes for the include directory and the library file name.
    include_dir = os.path.join(os.path.dirname(__file__),'include')
    lib_file = os.path.join(os.path.dirname(__file__),'_piff.so')

    # Load the C functions with cffi
    _ffi = cffi.FFI()
    # Put the C prototype of any functions that we want wrapped into header
    # files named *_C.h.  Then this reads them, parses the prototypes and
    # puts python callable versions into the _lib object.
    for file_name in glob.glob(os.path.join(include_dir,'*_C.h')):
        _ffi.cdef(open(file_name).read())
    _lib = _ffi.dlopen(lib_file)
    # Now piff._lib will have Python versions of all our C functions.


# Import things from the other files that we want in the piff namespace
from .config import piffify, setup_logger, read_config

# Models
# Class names here match what they are called in the config file
from .model import Model
from .pixelgrid import PixelGrid, Lanczos, Bilinear
from .gaussian_model import Gaussian

# Interpolators
# Class names here match what they are called in the config file
from .interp import Interp
from .mean_interp import Mean
from .polynomial_interp import Polynomial, polynomial_types
from .basis_interp import BasisInterp, BasisPolynomial

# Outliers
# Outlier handlers are named BlahOutliers where Blah is what they are called in teh config file
from .outliers import Outliers, ChisqOutliers, MedOutliers

# Inputs
# Input handlers are named InputBlah where Blah is what they are called in the config file
from .input import Input, InputFiles
from .star import Star, StarData, StarFit

# Outputs
# Output handlers are named OutputBlah where Blah is what they are called in the config file
from .output import Output, OutputFile

# PSF
# PSF classes are named BlahPSF where Blah is what they are called in the config file
from .psf import PSF, read
from .simplepsf import SimplePSF
from .singlechip import SingleChipPSF
from .multiexp import SingleExpPSF

# Stats
# Stats classes are named BlahStats where Blah is what they are called in the config file
from .stats import Stats, RhoStats, ShapeHistogramsStats, ShapeScatterStats
from .twod_stats import TwoDHistStats

# Util -- leave these in the piff.util namespace
from . import util
