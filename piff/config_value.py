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
.. module:: config_value
"""

import galsim
import os


def _GenerateImageHeaderValue(config, base, value_type):
    """Return a value read from the current image FITS header.
    """
    req = { 'key': str }
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req)
    key = kwargs['key']

    if '_current_image' not in base:
        raise ValueError("ImageHeaderValue requires base['_current_image'] to be set.")

    header = base['_current_image'].header
    if header is None:
        raise ValueError('Current image has no header information. Required for ImageHeaderValue.')
    if key not in header:
        raise KeyError("Key %s not found in FITS header" % key)

    return header[key], safe


def _GenerateASDFValue(config, base, value_type):
    """Read scalar metadata from an ASDF file using a dotted path.
    """
    import asdf
    from .util import get_asdf_value

    req = { 'file_name': str, 'path': str }
    opt = { 'dir': str }
    kwargs, safe = galsim.config.GetAllParams(config, base, req=req, opt=opt)
    file_name = kwargs['file_name']
    if 'dir' in kwargs:
        file_name = os.path.join(kwargs['dir'], file_name)
    with asdf.open(file_name) as af:
        value = get_asdf_value(af.tree, kwargs['path'])
    return value, safe


galsim.config.RegisterValueType('ImageHeaderValue', _GenerateImageHeaderValue,
                                [float, int, bool, str, None])
galsim.config.RegisterValueType('ASDFValue', _GenerateASDFValue,
                                [float, int, bool, str, None])
