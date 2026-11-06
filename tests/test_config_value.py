# Copyright (c) 2026 by Mike Jarvis and the other collaborators on GitHub at
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

import os
import fitsio
import galsim
import numpy as np
import pytest
import piff  # Register Piff's GalSim config value types.

from piff_test_helper import timer


@timer
def test_image_header_value():
    """Test FITS header values and the current-image requirements of ImageHeaderValue."""
    # Header values retain their numeric and string types through config parsing.
    file_name = os.path.join('output', 'config_value.fits')
    header = {'SKYLEVEL': 15.0, 'DETECTOR': 'WFI01'}
    fitsio.write(file_name, np.zeros((2, 2)), header=header, clobber=True)
    image = galsim.fits.read(file_name, read_header=True)
    base = {'_current_image': image}
    for key, value_type in [('SKYLEVEL', float), ('DETECTOR', str)]:
        config = {'value': {'type': 'ImageHeaderValue', 'key': key}}
        value = galsim.config.ParseValue(config, 'value', base, value_type)[0]
        assert value == header[key]

    # A missing header key raises KeyError.
    config = {'value': {'type': 'ImageHeaderValue', 'key': 'MISSING'}}
    np.testing.assert_raises(KeyError, galsim.config.ParseValue, config, 'value', base, None)

    # A current image is required even if the requested keyword is valid.
    config = {'value': {'type': 'ImageHeaderValue', 'key': 'SKYLEVEL'}}
    np.testing.assert_raises(ValueError, galsim.config.ParseValue, config, 'value', {}, None)

    # An image without a FITS header cannot supply ImageHeaderValue.
    image.header = None
    with np.testing.assert_raises(ValueError) as e:
        galsim.config.ParseValue(config, 'value', base, None)
    assert 'Current image has no header information' in str(e.exception)


@timer
def test_asdf_value():
    """Test independent ASDFValue lookup of scalar metadata and filename expressions."""
    # The ASDF tree contains scalar metadata without any science-image requirements.
    asdf = pytest.importorskip('asdf')
    file_name = os.path.join('output', 'config_value.asdf')
    metadata = {'sky': 15.0, 'detector': 'WFI01'}
    with asdf.AsdfFile({'metadata': metadata}) as af:
        af.write_to(file_name)

    # A dotted path reads a scalar value without a current input image.
    config = {
        'sky': {
            'type': 'ASDFValue',
            'file_name': file_name,
            'path': 'metadata.sky',
        },
    }
    assert galsim.config.ParseValue(config, 'sky', config, float)[0] == metadata['sky']

    # Directory and filename config references resolve to the selected ASDF file.
    config = {
        'detector': {
            'type': 'ASDFValue',
            'dir': 'output',
            'file_name': '@asdf_name',
            'path': 'metadata.detector',
        },
        'asdf_name': os.path.basename(file_name),
    }
    assert galsim.config.ParseValue(config, 'detector', config, str)[0] == metadata['detector']


if __name__ == '__main__':
    test_image_header_value()
    test_asdf_value()
