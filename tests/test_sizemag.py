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

import numpy as np
import piff
import os

from piff_test_helper import timer

@timer
def test_sizemag_plot():
    """Check a size-magnitude plot.
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(2)
    else:
        logger = piff.config.setup_logger(log_file='output/test_sizemag_plot.log')

    config = piff.config.read_config('sizemag.yaml')
    file_name = os.path.join('output', config['output']['stats'][0]['file_name'])

    # Some modifications to speed it up a bit.
    config['select'] = {
        'type': 'Properties',
        'where': '(CLASS_STAR > 0.9) & (MAG_AUTO < 13)',
        'hsm_size_reject': 4,
        'min_snr': 50,
    }
    config['psf']['interp'] = {'type': 'Mean'}
    config['psf']['outliers']['nsigma'] = 10
    del config['output']['stats'][1:]

    # Run via piffify
    piff.piffify(config, logger)
    assert os.path.isfile(file_name)

    # repeat with plotify function
    os.remove(file_name)
    piff.plotify(config, logger)
    assert os.path.isfile(file_name)

@timer
def test_smallbright():
    """Test the SmallBright selection algorithm.
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(2)
    else:
        logger = None

    config = piff.config.read_config('sizemag.yaml')
    config['select'] = {
        'type': 'SmallBright'
    }

    objects, _, _ = piff.Input.process(config['input'], logger=logger)
    stars = piff.Select.process(config['select'], objects, logger=logger)

    # This does a pretty decent job actually.  Finds 94 stars.
    print('nstars = ',len(stars))
    assert len(stars) == 94

    # They are all ones that CLASS_STAR also identified as stars.
    class_star = np.array([s['CLASS_STAR'] for s in stars])
    print('class_star = ',class_star)
    print('min class_star = ',np.min(class_star))
    assert np.all(class_star > 0.95)

    mag_auto = np.array([s['MAG_AUTO'] for s in stars])
    print('mag_auto = ',mag_auto)
    print('max mag_auto = ',np.max(mag_auto))
    assert np.max(mag_auto) < 16

    # Sizes are all pretty similar (by construction of the algorithm)
    sizes = [s.hsm[3] for s in stars]
    print('mean size = ',np.mean(sizes))
    print('median size = ',np.median(sizes))
    print('min/max size = ',np.min(sizes),np.max(sizes))
    assert (np.max(sizes)-np.min(sizes)) / np.median(sizes) < 0.1

    # Try some different parameter values.
    config['select'] = {
        'type': 'SmallBright',
        'bright_fraction': 0.1,
        'small_fraction': 0.5,
        'locus_fraction': 0.8,
        'max_spread': 0.05,
    }
    stars = piff.Select.process(config['select'], objects, logger=logger)

    # Fewer stars since limited to brighter subset
    print('nstars = ',len(stars))
    assert len(stars) == 51

    # But still finds all high confidence stars
    class_star = np.array([s['CLASS_STAR'] for s in stars])
    print('class_star = ',class_star)
    print('min class_star = ',np.min(class_star))
    assert np.all(class_star > 0.95)

    # And sizes are now restricted to max_spread = 0.05
    sizes = [s.hsm[3] for s in stars]
    print('mean size = ',np.mean(sizes))
    print('median size = ',np.median(sizes))
    print('min/max size = ',np.min(sizes),np.max(sizes))
    assert (np.max(sizes)-np.min(sizes)) / np.median(sizes) < 0.05

    # Error to have other parameters
    config['select'] = {
        'type': 'SmallBright',
        'bright_frac': 0.1,
        'small_frac': 0.5,
    }
    with np.testing.assert_raises(ValueError):
        piff.Select.process(config['select'], objects, logger=logger)

    # But ok to have parameters that the base class will handle.
    config['select'] = {
        'type': 'SmallBright',
        'locus_fraction': 0.8,
        'hsm_size_reject': 4,
        'min_snr': 50,
    }
    piff.Select.process(config['select'], objects, logger=logger)


@timer
def test_sizemag():
    """Test the SizeMag selection algorithm.
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(2)
    else:
        logger = None

    config = piff.config.read_config('sizemag.yaml')
    config['select'] = {
        'type': 'SizeMag'
    }

    objects, _, _ = piff.Input.process(config['input'], logger=logger)
    stars = piff.Select.process(config['select'], objects, logger=logger)

    # This finds more stars than the simple SmallBright selector found.
    print('nstars = ',len(stars))
    assert len(stars) == 135

    # A few of these have lower CLASS_STAR values, but still most are > 0.95
    class_star = np.array([s['CLASS_STAR'] for s in stars])
    print('class_star = ',class_star)
    print('min class_star = ',np.min(class_star))
    print('N class_star > 0.95 = ',np.sum(class_star > 0.95))
    assert np.sum(class_star > 0.95) == 133

    # This goes a bit fainter than the SmallBright selector did (which is kind of the point).
    mag_auto = np.array([s['MAG_AUTO'] for s in stars])
    print('mag_auto = ',mag_auto)
    print('max mag_auto = ',np.max(mag_auto))
    assert np.max(mag_auto) > 16
    assert np.max(mag_auto) < 17

    # Sizes are all pretty similar (but not exactly by construction anymore)
    sizes = [s.hsm[3] for s in stars]
    print('mean size = ',np.mean(sizes))
    print('median size = ',np.median(sizes))
    print('min/max size = ',np.min(sizes),np.max(sizes))
    assert (np.max(sizes)-np.min(sizes)) / np.median(sizes) < 0.1

    # Try some different parameter values.
    config['select'] = {
        'type': 'SizeMag',
        'initial_select': {
            'type': 'Properties',
            'where': '(CLASS_STAR > 0.9) & (MAG_AUTO < 16)',
        },
        'purity' : 0,
        'num_iter' : 2,
        'fit_order': 0,
    }
    stars = piff.Select.process(config['select'], objects, logger=logger)

    # A bit fewer, but not really consequentially different.
    print('nstars = ',len(stars))
    assert len(stars) == 124

    class_star = np.array([s['CLASS_STAR'] for s in stars])
    print('class_star = ',class_star)
    print('min class_star = ',np.min(class_star))
    print('N class_star > 0.95 = ',np.sum(class_star > 0.95))
    assert np.sum(class_star > 0.95) == 122

    sizes = [s.hsm[3] for s in stars]
    print('mean size = ',np.mean(sizes))
    print('median size = ',np.median(sizes))
    print('min/max size = ',np.min(sizes),np.max(sizes))
    assert (np.max(sizes)-np.min(sizes)) / np.median(sizes) < 0.1

    # Make sure it doesn't crap out if all the input objects are stars
    config['input'] = {
        'dir': 'input',
        'image_file_name': 'DECam_00241238_01.fits.fz',
        'image_hdu': 1,
        'badpix_hdu': 2,
        'weight_hdu': 3,

        'cat_file_name' : 'DECam_00241238_01_psfcat_tb_maxmag_17.0_magcut_3.0_findstars.fits',
        'cat_hdu' : 2,
        'x_col' : 'XWIN_IMAGE',
        'y_col' : 'YWIN_IMAGE',
        'sky_col' : 'BACKGROUND',

        'stamp_size': 25,
    }
    config['select'] = {
        'type': 'SizeMag',
        'initial_select': {
            # Use all the input objects
        }
    }
    objects, _, _ = piff.Input.process(config['input'], logger=logger)
    stars = piff.Select.process(config['select'], objects, logger=logger)

    # Error to have other parameters
    config['select'] = {
        'type': 'SizeMag',
        'order': 3,
    }
    with np.testing.assert_raises(ValueError):
        piff.Select.process(config['select'], objects, logger=logger)

    # But ok to have parameters that the base class will handle.
    config['select'] = {
        'type': 'SizeMag',
        'purity': 0,
        'hsm_size_reject': 4,
        'min_snr': 50,
    }
    piff.Select.process(config['select'], objects, logger=logger)

if __name__ == '__main__':
    test_sizemag_plot()
    test_smallbright()
    test_sizemag()
