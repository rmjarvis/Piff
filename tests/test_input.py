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

from __future__ import print_function
import os
import galsim
import numpy as np
import fitsio
import piff

from piff_test_helper import get_script_name, timer, CaptureLog


def setup():
    """Make sure the images and catalogs that we'll use throughout this module are done first.
    """
    gs_config = galsim.config.ReadConfig(os.path.join('input','make_input.yaml'))[0]
    galsim.config.Process(gs_config)

    # Add some header values to the first one.
    # Also add some alternate weight and badpix maps to enable some edge-case tests
    image_file = os.path.join('input','test_input_image_00.fits')
    with fitsio.FITS(image_file, 'rw') as f:
        f[0].write_key('SKYLEVEL', 200, 'sky level')
        f[0].write_key('GAIN_A', 2.0, 'gain')
        f[0].write_key('RA', '06:00:00', 'telescope ra')
        f[0].write_key('DEC', '-30:00:00', 'telescope dec')
        wt = f[1].read().copy()
        wt[:,:] = 0
        f.write(wt) # hdu = 3
        wt[:,:] = -1.
        f.write(wt) # hdu = 4
        bp = f[2].read().copy()
        bp = bp.astype(np.int32)
        bp[:,:] = 32768
        f.write(bp) # hdu = 5
        bp[:,:] = -32768
        f.write(bp) # hdu = 6
        bp[:,:] = 16
        f.write(bp) # hdu = 7
        wt[:,:] = 1.
        wt[::2,:] = 0   # Even cols
        f.write(wt) # hdu = 8
        bp[:,:] = 0
        bp[1::2,:] = 16 # Odd cols
        f.write(bp) # hdu = 9


@timer
def test_basic():
    """Test the (usual) basic kind of input field without too many bells and whistles.
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(log_file=os.path.join('output','test_input.log'))

    dir = 'input'
    image_file = 'test_input_image_00.fits'
    cat_file = 'test_input_cat_00.fits'

    # Simple with one image, cat
    config = {
                'dir' : dir,
                'image_file_name' : image_file,
                'cat_file_name': cat_file
             }
    input = piff.InputFiles(config, logger=logger)
    assert len(input.image_pos) == 1
    assert len(input.image_pos[0]) == 100

    # Can omit the dir and just inlcude it in the file names
    config = {
                'image_file_name' : os.path.join(dir,image_file),
                'cat_file_name': os.path.join(dir,cat_file)
             }
    input = piff.InputFiles(config, logger=logger)
    assert len(input.image_pos) == 1
    assert len(input.image_pos[0]) == 100

    # 3 images in a list
    image_files = [ 'test_input_image_%02d.fits'%k for k in range(3) ]
    cat_files = [ 'test_input_cat_%02d.fits'%k for k in range(3) ]
    config = {
                'dir' : dir,
                'image_file_name' : image_files,
                'cat_file_name': cat_files
             }
    input = piff.InputFiles(config, logger=logger)
    assert len(input.image_pos) == 3
    np.testing.assert_array_equal([len(p) for p in input.image_pos], 100)

    # 3 images using glob
    image_files = 'test_input_image_*.fits'
    cat_files = 'test_input_cat_*.fits'
    config = {
                'dir' : dir,
                'image_file_name' : image_files,
                'cat_file_name': cat_files
             }
    input = piff.InputFiles(config, logger=logger)
    assert len(input.image_pos) == 3
    np.testing.assert_array_equal([len(p) for p in input.image_pos], 100)

    # Can limit the number of stars
    config['nstars'] = 37
    input = piff.InputFiles(config, logger=logger)
    assert len(input.image_pos) == 3
    np.testing.assert_array_equal([len(p) for p in input.image_pos], 37)


@timer
def test_invalid():
    """Test invalid configuration elements
    """

    image_file = 'test_input_image_00.fits'
    cat_file = 'test_input_cat_00.fits'

    # Leaving off either image_file_name or cat_file_name is an error
    config = { 'image_file_name' : os.path.join('input',image_file) }
    np.testing.assert_raises(AttributeError, piff.InputFiles, config)
    config = { 'cat_file_name': os.path.join('input',cat_file) }
    np.testing.assert_raises(AttributeError, piff.InputFiles, config)

    # Invalid values for image or cat name
    config = { 'image_file_name' : os.path.join('input',image_file), 'cat_file_name' : 17 }
    np.testing.assert_raises(ValueError, piff.InputFiles, config)
    config = { 'cat_file_name': os.path.join('input',cat_file), 'image_file_name' : 17 }
    np.testing.assert_raises(ValueError, piff.InputFiles, config)

    # Using the wrong name (either as a glob or not) should raise an appropriate exception
    config = { 'dir' : 'input', 'image_file_name' : 'x'+image_file, 'cat_file_name' : cat_file }
    np.testing.assert_raises(ValueError, piff.InputFiles, config)
    config = { 'dir' : 'input', 'image_file_name' : 'x*'+image_file, 'cat_file_name' : cat_file }
    np.testing.assert_raises(ValueError, piff.InputFiles, config)
    config = { 'dir' : 'input', 'image_file_name' : image_file, 'cat_file_name' : 'x'+cat_file }
    np.testing.assert_raises(ValueError, piff.InputFiles, config)
    config = { 'dir' : 'input', 'image_file_name' : image_file, 'cat_file_name' : 'x*'+cat_file }
    np.testing.assert_raises(ValueError, piff.InputFiles, config)

    # nimages given but wrong
    image_files = 'input/test_input_image_*.fits'
    cat_files = 'input/test_input_cat_*.fits'
    image_files_1 = { 'type': 'Eval', 'str': '"input/test_input_image_01.fits"' }
    cat_files_1 = { 'type': 'Eval', 'str': '"input/test_input_cat_01.fits"' }
    config = { 'nimages' : 2, 'image_file_name' : image_files, 'cat_file_name': cat_files }
    np.testing.assert_raises(ValueError, piff.InputFiles, config)
    config = { 'nimages' : 2, 'image_file_name' : image_files, 'cat_file_name': cat_files_1 }
    np.testing.assert_raises(ValueError, piff.InputFiles, config)
    config = { 'nimages' : 2, 'image_file_name' : image_files_1, 'cat_file_name': cat_files }
    np.testing.assert_raises(ValueError, piff.InputFiles, config)
    config = { 'nimages' : 0, 'image_file_name' : image_files, 'cat_file_name': cat_files }
    np.testing.assert_raises(ValueError, piff.InputFiles, config)
    config = { 'nimages' : -1, 'image_file_name' : image_files, 'cat_file_name': cat_files }
    np.testing.assert_raises(ValueError, piff.InputFiles, config)

    # No images in list
    config = { 'dir' : 'input', 'image_file_name' : [], 'cat_file_name' : cat_file }
    np.testing.assert_raises(ValueError, piff.InputFiles, config)
    config = { 'dir' : 'input', 'image_file_name' : image_file, 'cat_file_name' : [] }
    np.testing.assert_raises(ValueError, piff.InputFiles, config)


@timer
def test_cols():
    """Test the various allowed column specifications
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(log_file=os.path.join('output','test_input.log'))

    # Specifiable columns are: x, y, flag, use, sky, gain.  (We'll do flag, use below.)
    config = {
                'dir' : 'input',
                'image_file_name' : 'test_input_image_00.fits',
                'cat_file_name' : 'test_input_cat_00.fits',
                'x_col' : 'x',
                'y_col' : 'y',
                'sky_col' : 'sky',
                'gain_col' : 'gain',
             }
    input = piff.InputFiles(config, logger=logger)
    assert len(input.image_pos) == 1
    assert len(input.sky) == 1
    assert len(input.gain) == 1
    assert len(input.image_pos[0]) == 100
    assert len(input.sky[0]) == 100
    assert len(input.gain[0]) == 100

    # sky and gain are constant (although they don't have to be of course)
    sky = input.sky[0][0]
    gain = input.gain[0][0]
    np.testing.assert_array_equal(input.sky[0], sky)
    np.testing.assert_array_equal(input.gain[0], gain)

    # sky and gain can also be given as float values for the whole catalog
    config = {
                'dir' : 'input',
                'image_file_name' : 'test_input_image_00.fits',
                'cat_file_name' : 'test_input_cat_00.fits',
                'sky' : sky,
                'gain' : gain,
             }
    input = piff.InputFiles(config, logger=logger)
    assert len(input.sky) == 1
    assert len(input.gain) == 1
    assert len(input.sky[0]) == 100
    assert len(input.gain[0]) == 100
    # These aren't precisely equal because we go through a str value, which truncates it.
    # We could hack this to keep it exact, but it's probably not worth it and it's easier to
    # enable both str and float by reading it as str and then trying the float conversion to see
    # it if works.  Anyway, that's why this is only decimal=9.
    np.testing.assert_almost_equal(input.sky[0], sky, decimal=9)
    np.testing.assert_almost_equal(input.gain[0], gain, decimal=9)

    # sky and gain can also be given as str values, which mean look in the FITS header.
    config = {
                'dir' : 'input',
                'image_file_name' : 'test_input_image_00.fits',
                'cat_file_name' : 'test_input_cat_00.fits',
                'sky' : 'SKYLEVEL',
                'gain' : 'GAIN_A',
             }
    input = piff.InputFiles(config, logger=logger)
    assert len(input.sky) == 1
    assert len(input.gain) == 1
    assert len(input.sky[0]) == 100
    assert len(input.gain[0]) == 100
    np.testing.assert_array_equal(input.sky[0], 200)
    np.testing.assert_array_equal(input.gain[0], 2.0)

    # Using flag will skip flagged columns.  Here every 5th item is flagged.
    config = {
                'dir' : 'input',
                'image_file_name' : 'test_input_image_00.fits',
                'cat_file_name' : 'test_input_cat_00.fits',
                'flag_col' : 'flag',
             }
    input = piff.InputFiles(config, logger=logger)
    assert len(input.image_pos) == 1
    assert len(input.image_pos[0]) == 80

    # Similarly the use columns will skip anything with use == 0 (every 7th item here)
    config = {
                'dir' : 'input',
                'image_file_name' : 'test_input_image_00.fits',
                'cat_file_name' : 'test_input_cat_00.fits',
                'use_col' : 'use',
             }
    input = piff.InputFiles(config, logger=logger)
    assert len(input.image_pos) == 1
    assert len(input.image_pos[0]) == 85

    # Can do both
    config = {
                'dir' : 'input',
                'image_file_name' : 'test_input_image_00.fits',
                'cat_file_name' : 'test_input_cat_00.fits',
                'use_col' : 'use',
                'flag_col' : 'flag',
             }
    input = piff.InputFiles(config, logger=logger)
    assert len(input.image_pos) == 1
    assert len(input.image_pos[0]) == 68

    # Check invalid column names
    base_config = {
                'dir' : 'input',
                'image_file_name' : 'test_input_image_00.fits',
                'cat_file_name' : 'test_input_cat_00.fits', }
    np.testing.assert_raises(ValueError, piff.InputFiles, dict(x_col='xx', **base_config))
    np.testing.assert_raises(ValueError, piff.InputFiles, dict(y_col='xx', **base_config))
    np.testing.assert_raises(ValueError, piff.InputFiles, dict(sky_col='xx', **base_config))
    np.testing.assert_raises(ValueError, piff.InputFiles, dict(gain_col='xx', **base_config))
    np.testing.assert_raises(ValueError, piff.InputFiles, dict(flag_col='xx', **base_config))
    np.testing.assert_raises(ValueError, piff.InputFiles, dict(use_col='xx', **base_config))

    # Can't give duplicate sky, gain
    np.testing.assert_raises(ValueError, piff.InputFiles, dict(sky_col='sky', sky=3, **base_config))
    np.testing.assert_raises(ValueError, piff.InputFiles, dict(gain_col='gain', gain=3, **base_config))

    # Invalid header keys
    np.testing.assert_raises(KeyError, piff.InputFiles, dict(sky='sky', **base_config))
    np.testing.assert_raises(KeyError, piff.InputFiles, dict(gain='gain', **base_config))


@timer
def test_chipnum():
    """Test the ability to renumber the chipnums
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(log_file=os.path.join('output','test_input.log'))

    # First, the default is to just use the index in the image list
    dir = 'input'
    image_files = 'test_input_image_*.fits'
    cat_files = 'test_input_cat_*.fits'
    config = {
                'dir' : dir,
                'image_file_name' : image_files,
                'cat_file_name': cat_files
             }
    input = piff.InputFiles(config, logger=logger)
    assert input.chipnums == list(range(3))

    # Now make the chipnums something else
    config = {
                'dir' : dir,
                'image_file_name' : image_files,
                'cat_file_name': cat_files,
                'chipnum' : [ 5, 6, 7 ]
             }
    input = piff.InputFiles(config, logger=logger)
    assert input.chipnums == [ i+5 for i in range(3) ]

    # Use the GalSim Eval capability to get the index + 1
    config = {
                'dir' : dir,
                'image_file_name' : image_files,
                'cat_file_name': cat_files,
                'chipnum' : '$image_num + 1'
             }
    input = piff.InputFiles(config, logger=logger)
    assert input.chipnums == [ i+1 for i in range(3) ]

    # Or parse it from the image_file_name
    config = {
                'dir' : dir,
                'image_file_name' : image_files,
                'cat_file_name': cat_files,
                'chipnum' : {
                    'type' : 'Eval',
                    'str' : "image_file_name.split('_')[-1].split('.')[0]",
                    'simage_file_name' : '@input.image_file_name'
                }
             }
    input = piff.InputFiles(config, logger=logger)
    assert input.chipnums == list(range(3))

@timer
def test_weight():
    """Test the weight map and bad pixel masks
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(log_file=os.path.join('output','test_input.log'))

    # If no weight or badpix is specified, the weights are all equal.
    config = {
                'image_file_name' : 'input/test_input_image_00.fits',
                'cat_file_name' : 'input/test_input_cat_00.fits',
             }
    input = piff.InputFiles(config, logger=logger)
    assert len(input.image_pos) == 1
    assert len(input.image_pos[0]) == 100
    assert len(input.images) == 1
    assert input.images[0].array.shape == (1024, 1024)
    assert len(input.weight) == 1
    assert input.weight[0].array.shape == (1024, 1024)
    np.testing.assert_array_equal(input.weight[0].array, 1.0)

    # The default weight and badpix masks that GalSim makes don't do any masking, so this
    # is the almost the same as above, but the weight value is 1/sky.
    config = {
                'image_file_name' : 'input/test_input_image_00.fits',
                'cat_file_name' : 'input/test_input_cat_00.fits',
                'weight_hdu' : 1,
                'badpix_hdu' : 2,
                'sky_col' : 'sky',  # Used to determine what the value of weight should be
                'gain_col' : 'gain',
             }
    input = piff.InputFiles(config, logger=logger)
    assert len(input.image_pos) == 1
    assert len(input.image_pos[0]) == 100
    assert len(input.images) == 1
    assert input.images[0].array.shape == (1024, 1024)
    assert len(input.weight) == 1
    assert input.weight[0].array.shape == (1024, 1024)
    sky = input.sky[0][0]
    gain = input.gain[0][0]
    scale = input.images[0].scale
    read_noise = 10
    expected_noise = sky / gain + read_noise**2 / gain**2
    np.testing.assert_almost_equal(input.weight[0].array, expected_noise**-1)

    # Can set the noise by hand
    config = {
                'image_file_name' : 'input/test_input_image_00.fits',
                'cat_file_name' : 'input/test_input_cat_00.fits',
                'noise' : 32,
             }
    input = piff.InputFiles(config, logger=logger)
    assert len(input.weight) == 1
    assert input.weight[0].array.shape == (1024, 1024)
    np.testing.assert_almost_equal(input.weight[0].array, 32.**-1)

    # Some old versions of fitsio had a bug where the badpix mask could be offset by 32768.
    # We move them back to 0
    config = {
                'image_file_name' : 'input/test_input_image_00.fits',
                'cat_file_name' : 'input/test_input_cat_00.fits',
                'weight_hdu' : 1,
                'badpix_hdu' : 5,
             }
    input = piff.InputFiles(config, logger=logger)
    assert len(input.weight) == 1
    assert input.weight[0].array.shape == (1024, 1024)
    np.testing.assert_almost_equal(input.weight[0].array, expected_noise**-1)

    config['badpix_hdu'] = 6
    input = piff.InputFiles(config, logger=logger)
    assert len(input.weight) == 1
    assert input.weight[0].array.shape == (1024, 1024)
    np.testing.assert_almost_equal(input.weight[0].array, expected_noise**-1)

    # Various ways to get all weight values == 0 (which will emit a logger message, but isn't
    # an error).
    config['weight_hdu'] = 1
    config['badpix_hdu'] = 7  # badpix > 0
    input = piff.InputFiles(config, logger=logger)
    assert len(input.weight) == 1
    assert input.weight[0].array.shape == (1024, 1024)
    np.testing.assert_almost_equal(input.weight[0].array, 0.)

    config['weight_hdu'] = 3  # wt = 0
    config['badpix_hdu'] = 2
    input = piff.InputFiles(config, logger=logger)
    assert len(input.weight) == 1
    assert input.weight[0].array.shape == (1024, 1024)
    np.testing.assert_almost_equal(input.weight[0].array, 0.)

    config['weight_hdu'] = 8  # Even cols are = 0
    config['badpix_hdu'] = 9  # Odd cols are > 0
    input = piff.InputFiles(config, logger=logger)
    assert len(input.weight) == 1
    assert input.weight[0].array.shape == (1024, 1024)
    np.testing.assert_almost_equal(input.weight[0].array, 0.)

    # Negative valued weights are invalid
    config['weight_hdu'] = 4
    with CaptureLog() as cl:
        piff.InputFiles(config, logger=cl.logger)
    assert 'Warning: weight map has invalid negative-valued pixels.' in cl.output


@timer
def test_stars():
    """Test the input.makeStars function
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(log_file=os.path.join('output','test_input.log'))

    dir = 'input'
    image_file = 'test_input_image_00.fits'
    cat_file = 'test_input_cat_00.fits'

    # Turn off two defaults for now (max_snr=100 and use_partial=False)
    config = {
                'dir' : dir,
                'image_file_name' : image_file,
                'cat_file_name' : cat_file,
                'weight_hdu' : 1,
                'sky_col' : 'sky',
                'gain_col' : 'gain',
                'max_snr' : 0,
                'use_partial' : True,
             }
    input = piff.InputFiles(config, logger=logger)
    stars = input.makeStars(logger=logger)
    assert len(stars) == 100
    chipnum_list = [ star['chipnum'] for star in stars ]
    gain_list = [ star['gain'] for star in stars ]
    snr_list = [ star['snr'] for star in stars ]
    snr_list2 = [ input.calculateSNR(star.data.image, star.data.orig_weight) for star in stars ]
    print('snr = ', np.min(snr_list), np.max(snr_list))
    np.testing.assert_array_equal(chipnum_list, 0)
    np.testing.assert_array_equal(gain_list, gain_list[0])
    np.testing.assert_almost_equal(snr_list, snr_list2, decimal=5)
    assert np.min(snr_list) < 4.
    assert np.max(snr_list) > 200.

    # max_snr increases the noise to achieve a maximum snr
    config['max_snr'] = 120
    input = piff.InputFiles(config, logger=logger)
    stars = input.makeStars(logger=logger)
    assert len(stars) == 100
    snr_list = [ star['snr'] for star in stars ]
    snr_list2 = [ input.calculateSNR(star.data.image, star.data.orig_weight) for star in stars ]
    print('snr = ', np.min(snr_list), np.max(snr_list))
    np.testing.assert_almost_equal(snr_list, snr_list2, decimal=5)
    assert np.min(snr_list) < 4.
    assert np.max(snr_list) == 120.

    # The default is max_snr == 100
    del config['max_snr']
    input = piff.InputFiles(config, logger=logger)
    stars = input.makeStars(logger=logger)
    assert len(stars) == 100
    snr_list = np.array([ star['snr'] for star in stars ])
    snr_list2 = [ input.calculateSNR(star.data.image, star.data.orig_weight) for star in stars ]
    print('snr = ', np.min(snr_list), np.max(snr_list))
    np.testing.assert_almost_equal(snr_list, snr_list2, decimal=5)
    assert np.min(snr_list) < 4.
    assert np.max(snr_list) == 100.

    # min_snr removes stars with a snr < min_snr
    config['min_snr'] = 20
    input = piff.InputFiles(config, logger=logger)
    stars = input.makeStars(logger=logger)
    print('len should be ',len(snr_list[snr_list >= 20]))
    print('actual len is ',len(stars))
    assert len(stars) == len(snr_list[snr_list >= 20])
    assert len(stars) == 91  # hard-coded for this case, just to make sure
    snr_list = np.array([ star['snr'] for star in stars ])
    snr_list2 = [ input.calculateSNR(star.data.image, star.data.orig_weight) for star in stars ]
    print('snr = ', np.min(snr_list), np.max(snr_list))
    np.testing.assert_almost_equal(snr_list, snr_list2, decimal=5)
    assert np.min(snr_list) >= 20.
    assert np.max(snr_list) == 100.

    # use_partial=False will skip any stars that are partially off the edge of the image
    config['use_partial'] = False
    input = piff.InputFiles(config, logger=logger)
    stars = input.makeStars(logger=logger)
    assert len(stars) == 89  # skipped 2 additional stars

    # use_partial=False is the default
    del config['use_partial']
    input = piff.InputFiles(config, logger=logger)
    stars = input.makeStars(logger=logger)
    assert len(stars) == 89  # skipped 2 additional stars

    # alt_x and alt_y also include some object completely off the image, which are always skipped.
    # (Don't do the min_snr anymore, since most of these stamps don't actually have any signal.)
    config['x_col'] = 'alt_x'
    config['y_col'] = 'alt_y'
    del config['min_snr']
    input = piff.InputFiles(config, logger=logger)
    stars = input.makeStars(logger=logger)
    assert len(stars) == 37

    # Also skip objects which are all weight=0
    config['weight_hdu'] = 3
    input = piff.InputFiles(config, logger=logger)
    stars = input.makeStars(logger=logger)
    assert len(stars) == 0

    # But not ones that are only partially weight=0
    config['weight_hdu'] = 8
    input = piff.InputFiles(config, logger=logger)
    stars = input.makeStars(logger=logger)
    assert len(stars) == 37


@timer
def test_pointing():
    """Test the input.setPointing function
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(log_file=os.path.join('output','test_input.log'))

    dir = 'input'
    image_file = 'test_input_image_00.fits'
    cat_file = 'test_input_cat_00.fits'

    # First, with no ra, dec, pointing is None
    config = {
                'dir' : dir,
                'image_file_name' : image_file,
                'cat_file_name' : cat_file,
             }
    input = piff.InputFiles(config, logger=logger)
    assert input.pointing is None

    # Explicit ra, dec as floats
    config['ra'] = 6.0
    config['dec'] = -30.0
    input = piff.InputFiles(config, logger=logger)
    np.testing.assert_almost_equal(input.pointing.ra.rad(), np.pi/2.)
    np.testing.assert_almost_equal(input.pointing.dec.rad(), -np.pi/6.)

    # Also ok as ints in this case
    config['ra'] = 6
    config['dec'] = -30
    input = piff.InputFiles(config, logger=logger)
    np.testing.assert_almost_equal(input.pointing.ra.rad(), np.pi/2.)
    np.testing.assert_almost_equal(input.pointing.dec.rad(), -np.pi/6.)

    # Strings as hh:mm:ss or dd:mm:ss
    config['ra'] = '06:00:00'
    config['dec'] = '-30:00:00'
    input = piff.InputFiles(config, logger=logger)
    np.testing.assert_almost_equal(input.pointing.ra.rad(), np.pi/2.)
    np.testing.assert_almost_equal(input.pointing.dec.rad(), -np.pi/6.)

    # Strings as keys into FITS header
    config['ra'] = 'RA'
    config['dec'] = 'DEC'
    input = piff.InputFiles(config, logger=logger)
    np.testing.assert_almost_equal(input.pointing.ra.rad(), np.pi/2.)
    np.testing.assert_almost_equal(input.pointing.dec.rad(), -np.pi/6.)

    # If multiple files, use the first one.
    config['image_file_name'] = 'test_input_image_*.fits'
    config['cat_file_name'] = 'test_input_cat_*.fits'
    input = piff.InputFiles(config, logger=logger)
    np.testing.assert_almost_equal(input.pointing.ra.rad(), np.pi/2.)
    np.testing.assert_almost_equal(input.pointing.dec.rad(), -np.pi/6.)

    # Check invalid ra,dec values
    base_config = { 'dir' : dir, 'image_file_name' : image_file, 'cat_file_name' : cat_file }
    np.testing.assert_raises(ValueError, piff.InputFiles,
                             dict(ra=0, dec='00:00:00' , **base_config))
    np.testing.assert_raises(ValueError, piff.InputFiles,
                             dict(ra='00:00:00', dec=0, **base_config))
    np.testing.assert_raises(ValueError, piff.InputFiles,
                             dict(ra=0*galsim.degrees, dec=0*galsim.radians, **base_config))
    np.testing.assert_raises(KeyError, piff.InputFiles,
                             dict(ra='bad_ra', dec='bad_dec', **base_config))
    np.testing.assert_raises(ValueError, piff.InputFiles,
                             dict(ra=0, **base_config))
    np.testing.assert_raises(ValueError, piff.InputFiles,
                             dict(dec=0, **base_config))

    # If image has celestial wcs, and no ra, dec specified then it will compute it for you
    config = {
                'dir' : dir,
                'image_file_name' : 'DECam_00241238_01.fits.fz',
                'cat_file_name' : 'DECam_00241238_01_cat.fits',
                'cat_hdu' : 2,
                'x_col' : 'XWIN_IMAGE',
                'y_col' : 'YWIN_IMAGE',
             }
    input = piff.InputFiles(config, logger=logger)
    np.testing.assert_almost_equal(input.pointing.ra / galsim.hours, 4.063, decimal=3)
    np.testing.assert_almost_equal(input.pointing.dec / galsim.degrees, -51.471, decimal=3)

    # Similar, but not quite equal to teh TELRA, TELDEC, which is at center of exposure.
    config['ra'] = 'TELRA'
    config['dec'] = 'TELDEC'
    input = piff.InputFiles(config, logger=logger)
    np.testing.assert_almost_equal(input.pointing.ra / galsim.hours, 4.097, decimal=3)
    np.testing.assert_almost_equal(input.pointing.dec / galsim.degrees, -52.375, decimal=3)

    # We only have the one celestial wcs image in the repo, but with multiple ones, it will
    # average over all images.
    config = {
                'dir' : dir,
                'image_file_name' : ['DECam_00241238_01.fits.fz', 'DECam_00241238_01.fits.fz'],
                'cat_file_name' : ['DECam_00241238_01_cat.fits', 'DECam_00241238_01_cat.fits'],
                'cat_hdu' : 2,
                'x_col' : 'XWIN_IMAGE',
                'y_col' : 'YWIN_IMAGE',
             }
    input = piff.InputFiles(config, logger=logger)
    np.testing.assert_almost_equal(input.pointing.ra / galsim.hours, 4.063, decimal=3)
    np.testing.assert_almost_equal(input.pointing.dec / galsim.degrees, -51.471, decimal=3)



if __name__ == '__main__':
    setup()
    test_basic()
    test_invalid()
    test_cols()
    test_chipnum()
    test_weight()
    test_stars()
    test_pointing()
