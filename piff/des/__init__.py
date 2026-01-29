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
Some specialized features that specific to the DES survey and the DECam camera.
"""

from .decam_wavefront import DECamWavefront
from .decaminfo import DECamInfo

# The following is a workaround to fix an astropy backwards incompatibility that
# they are (quite reasonably imo) unwilling to fix.
# cf. https://github.com/astropy/astropy/issues/17416#issuecomment-2489073378
# They added an attribute, which breaks the pickled object we read in from Y6
# wcs objects in the Piff output file.  So if we get one of these, we need
# to fix the attribute to work with new astropy code.

def fix_y6(wcs_dict):
    try:
        for chipnum, wcs in wcs_dict.items():
            for e in wcs._wcs.pmap.elements:
                for d in getattr(e, 'tweak_data', []):
                    if hasattr(d, 'array'):
                        d.array._tbsize = d.array.nbytes
        return True
    except Exception as e:  # pragma: no cover
        return False
