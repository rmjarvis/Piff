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

@timer
def setup():
    """Make sure the images and catalogs that we'll use throughout this module are done first.
    """
    gs_config = galsim.config.ReadConfig(os.path.join('input','make_input.yaml'))[0]
    galsim.config.BuildFiles(2, galsim.config.CopyConfig(gs_config))

    # For the third file, add in a real wcs and ra, dec to the output catalog.
    gs_config['image']['wcs'] = {
        'type': 'Tan',
        'dudx': 0.27,
        'dudy': 0.01,
        'dvdx': -0.02,
        'dvdy': 0.26,
        'ra': '6 hours',
        'dec': '-30 degrees',
    }
    gs_config['output']['truth']['columns']['ra'] = '$wcs.toWorld(image_pos).ra / galsim.hours'
    gs_config['output']['truth']['columns']['dec'] = '$wcs.toWorld(image_pos).dec / galsim.degrees'
    galsim.config.BuildFiles(1, galsim.config.CopyConfig(gs_config), file_num=2)

    cat_file_name = os.path.join('input', 'test_input_cat_00.fits')
    data = fitsio.read(cat_file_name)
    sky = np.mean(data['sky'])
    gain = np.mean(data['gain'])
    print('sky, gain = ',sky,gain)
    satur = 1890  # Exactly 1 star has a pixel > 1890, so pretend this is the saturation level.

    # Write a sky image file
    sky_im = galsim.Image(1024,1024, init_value=sky)
    sky_im.write(os.path.join('input', 'test_input_sky_00.fits.fz'))

    # Add two color columns to first catalog file
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(234)
    gr_color = rng1.uniform(0.,1.,100)
    rz_color = rng2.uniform(0.,1.,100)
    print('gr_color, rz_color = ', gr_color[:10], rz_color[:10])
    with fitsio.FITS(cat_file_name, 'rw') as f:
        f[1].insert_column('gr_color', gr_color)
        f[1].insert_column('rz_color', rz_color)

    # Add some header values to the first one.
    # Also add some alternate weight and badpix maps to enable some edge-case tests
    image_file = os.path.join('input','test_input_image_00.fits')
    with fitsio.FITS(image_file, 'rw') as f:
        f[0].write_key('SKYLEVEL', sky, 'sky level')
        f[0].write_key('GAIN_A', gain, 'gain')
        f[0].write_key('SATURAT', satur, 'saturation level')
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
        wt = f[1].read().copy()
        var = 1/wt + (f[0].read() - sky) / gain
        f.write(var) # hdu = 10


@timer
def test_basic():
    """Test the (usual) basic kind of input field without too many bells and whistles.
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = None

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
    assert input.nimages == 1
    _, _, image_pos, _ = input.getRawImageData(0)
    assert len(image_pos) == 100

    # Can omit the dir and just include it in the file names
    config = {
                'image_file_name' : os.path.join(dir,image_file),
                'cat_file_name': os.path.join(dir,cat_file)
             }
    input = piff.InputFiles(config, logger=logger)
    assert input.nimages == 1
    _, _, image_pos, _ = input.getRawImageData(0)
    assert len(image_pos) == 100

    # 3 images in a list
    image_files = [ 'test_input_image_%02d.fits'%k for k in range(3) ]
    cat_files = [ 'test_input_cat_%02d.fits'%k for k in range(3) ]
    config = {
                'dir' : dir,
                'image_file_name' : image_files,
                'cat_file_name': cat_files
             }
    input = piff.InputFiles(config, logger=logger)
    assert input.nimages == 3
    for i in range(3):
        _, _, image_pos, _ = input.getRawImageData(i)
        assert len(image_pos) == 100

    # Again without dir.
    image_files = [ 'input/test_input_image_%02d.fits'%k for k in range(3) ]
    cat_files = [ 'input/test_input_cat_%02d.fits'%k for k in range(3) ]
    config = {
                'image_file_name' : image_files,
                'cat_file_name': cat_files
             }
    input = piff.InputFiles(config, logger=logger)
    assert input.nimages == 3
    for i in range(3):
        _, _, image_pos, _ = input.getRawImageData(i)
        assert len(image_pos) == 100

    # 3 images using glob
    image_files = 'test_input_image_*.fits'
    cat_files = 'test_input_cat_*.fits'
    config = {
                'dir' : dir,
                'image_file_name' : image_files,
                'cat_file_name': cat_files
             }
    input = piff.InputFiles(config, logger=logger)
    assert input.nimages == 3
    for i in range(3):
        _, _, image_pos, _ = input.getRawImageData(i)
        assert len(image_pos) == 100

    # Can limit the number of stars
    config['nstars'] = 37
    input = piff.InputFiles(config, logger=logger)
    assert input.nimages == 3
    for i in range(3):
        _, _, image_pos, _ = input.getRawImageData(i)
        assert len(image_pos) == 37

    # Can limit stars differently on each chip
    config['nstars'] = '$0 if @image_num == 1 else 20 if @image_num == 2 else 40'
    input = piff.InputFiles(config, logger=logger)
    assert input.nimages == 3
    for i in range(3):
        _, _, image_pos, _ = input.getRawImageData(i)
        if i == 0:
            assert len(image_pos) == 40
        elif i == 1:
            assert len(image_pos) == 0
        else:
            assert len(image_pos) == 20

    # Semi-gratuitous use of reserve_frac when one image has no stars for coverage
    # This functionality changed location to the Select class, but keep this test nonetheless.
    select_config = {'reserve_frac': 0.2}
    config['use_partial'] = True
    config['ra'] = 6.0
    config['dec'] = -30.0
    input = piff.InputFiles(config, logger=logger)
    assert input.nimages == 3
    stars = input.makeStars(logger=logger)
    select = piff.FlagSelect(select_config, logger=logger)
    select.reserveStars(stars, logger=logger)
    print('len stars = ',len(stars))
    assert len(stars) == 60
    assert len([s for s in stars if s.chipnum == 0]) == 40
    assert len([s for s in stars if s.chipnum == 1]) == 0
    assert len([s for s in stars if s.chipnum == 2]) == 20
    reserve_stars = [s for s in stars if s.is_reserve]
    assert len(reserve_stars) == 12
    assert len([s for s in reserve_stars if s['chipnum'] == 0]) == 8
    assert len([s for s in reserve_stars if s['chipnum'] == 1]) == 0
    assert len([s for s in reserve_stars if s['chipnum'] == 2]) == 4

    # If no stars, raise error
    # (normally because all stars have errors, but easier to just limit to 0 to test this.)
    config['nstars'] = 0
    with np.testing.assert_raises(RuntimeError):
        input = piff.Input.process(config, logger=logger)


@timer
def test_invalid():
    """Test invalid configuration elements
    """

    image_file = 'test_input_image_00.fits'
    cat_file = 'test_input_cat_00.fits'

    # Leaving off either image_file_name or cat_file_name is an error
    config = { 'image_file_name' : os.path.join('input',image_file) }
    np.testing.assert_raises(TypeError, piff.InputFiles, config)
    config = { 'cat_file_name': os.path.join('input',cat_file) }
    np.testing.assert_raises(TypeError, piff.InputFiles, config)

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

    # Missing nimages (required when using dict)
    config = { 'image_file_name' : image_files_1, 'cat_file_name': cat_files }
    np.testing.assert_raises(TypeError, piff.InputFiles, config)


@timer
def test_cols():
    """Test the various allowed column specifications
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = None

    # Specifiable columns are: x, y, flag, use, sky, gain.  (We'll do flag, use below.)
    config = {
                'dir' : 'input',
                'image_file_name' : 'test_input_image_02.fits',
                'cat_file_name' : 'test_input_cat_02.fits',
                'x_col' : 'x',
                'y_col' : 'y',
             }
    input = piff.InputFiles(config, logger=logger)
    assert input.nimages == 1
    _, _, image_pos, _ = input.getRawImageData(0)
    assert len(image_pos) == 100

    # Can do ra, dec instead of x, y
    config = {
                'dir' : 'input',
                'image_file_name' : 'test_input_image_02.fits',
                'cat_file_name' : 'test_input_cat_02.fits',
                'ra_col' : 'ra',
                'dec_col' : 'dec',
                'ra_units' : 'hours',
                'dec_units' : 'degrees',
             }
    input2 = piff.InputFiles(config, logger=logger)
    assert input2.nimages == 1
    _, _, image_pos2, _ = input2.getRawImageData(0)
    print('input.image_pos = ',image_pos)
    print('input2.image_pos = ',image_pos2)
    assert len(image_pos2) == 100
    x1 = [pos.x for pos in image_pos]
    x2 = [pos.x for pos in image_pos2]
    y1 = [pos.y for pos in image_pos]
    y2 = [pos.y for pos in image_pos2]
    np.testing.assert_allclose(x2, x1)
    np.testing.assert_allclose(y2, y1)

    # Back to first file, where we also have header values for things.
    cat_file_name = os.path.join('input', 'test_input_cat_00.fits')
    data = fitsio.read(cat_file_name)
    sky = np.mean(data['sky'])
    gain = np.mean(data['gain'])
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
    assert input.nimages == 1
    _, _, image_pos, props = input.getRawImageData(0)
    sky_list = props['sky']
    gain_list = props['gain']
    assert len(image_pos) == 100
    assert len(sky_list) == 100
    assert len(gain_list) == 100
    # sky and gain are constant (although they don't have to be of course)
    np.testing.assert_array_equal(sky_list, sky)
    np.testing.assert_array_equal(gain_list, gain)

    # sky and gain can also be given as float values for the whole catalog
    config = {
                'dir' : 'input',
                'image_file_name' : 'test_input_image_00.fits',
                'cat_file_name' : 'test_input_cat_00.fits',
                'sky' : sky,
                'gain' : gain,
             }
    input = piff.InputFiles(config, logger=logger)
    _, _, image_pos, props = input.getRawImageData(0)
    sky_list = props['sky']
    gain_list = props['gain']
    assert len(image_pos) == 100
    assert len(sky_list) == 100
    assert len(gain_list) == 100
    # These aren't precisely equal because we go through a str value, which truncates it.
    # We could hack this to keep it exact, but it's probably not worth it and it's easier to
    # enable both str and float by reading it as str and then trying the float conversion to see
    # it if works.  Anyway, that's why this is only decimal=9.
    np.testing.assert_almost_equal(sky_list, sky, decimal=9)
    np.testing.assert_almost_equal(gain_list, gain, decimal=9)

    # sky and gain can also be given as str values, which mean look in the FITS header.
    config = {
                'dir' : 'input',
                'image_file_name' : 'test_input_image_00.fits',
                'cat_file_name' : 'test_input_cat_00.fits',
                'sky' : 'SKYLEVEL',
                'gain' : 'GAIN_A',
             }
    input = piff.InputFiles(config, logger=logger)
    _, _, image_pos, props = input.getRawImageData(0)
    sky_list = props['sky']
    gain_list = props['gain']
    assert len(image_pos) == 100
    assert len(sky_list) == 100
    assert len(gain_list) == 100
    np.testing.assert_almost_equal(sky_list, sky)
    np.testing.assert_almost_equal(gain_list, gain)

    # including satur will skip stars that are over the given saturation value.
    # (It won't skip them here, just when building the stars list.)
    config = {
                'dir' : 'input',
                'image_file_name' : 'test_input_image_00.fits',
                'cat_file_name' : 'test_input_cat_00.fits',
                'sky' : 'SKYLEVEL',
                'gain' : 'GAIN_A',
                'satur' : 1890,
             }
    input = piff.InputFiles(config, logger=logger)
    _, _, image_pos, props = input.getRawImageData(0)
    satur = props['satur'][0]
    assert satur == 1890
    assert len(image_pos) == 100

    config = {
                'dir' : 'input',
                'image_file_name' : 'test_input_image_00.fits',
                'cat_file_name' : 'test_input_cat_00.fits',
                'sky' : 'SKYLEVEL',
                'gain' : 'GAIN_A',
                'satur' : 'SATURAT',
             }
    input = piff.InputFiles(config, logger=logger)
    _, _, image_pos, props = input.getRawImageData(0)
    satur = props['satur'][0]
    assert satur == 1890
    assert len(image_pos) == 100

    # Using flag will skip flagged columns.  Here every 5th item is flagged.
    config = {
                'dir' : 'input',
                'image_file_name' : 'test_input_image_00.fits',
                'cat_file_name' : 'test_input_cat_00.fits',
                'flag_col' : 'flag',
                'skip_flag' : 4
             }
    input = piff.InputFiles(config, logger=logger)
    _, _, image_pos, _ = input.getRawImageData(0)
    assert input.nimages == 1
    print('len = ',len(image_pos))
    assert len(image_pos) == 80

    # Similarly the use columns will skip anything with use == 0 (every 7th item here)
    config = {
                'dir' : 'input',
                'image_file_name' : 'test_input_image_00.fits',
                'cat_file_name' : 'test_input_cat_00.fits',
                'flag_col' : 'flag',
                'use_flag' : 1
             }
    input = piff.InputFiles(config, logger=logger)
    _, _, image_pos, _ = input.getRawImageData(0)
    print('len = ',len(image_pos))
    assert len(image_pos) == 85

    # Can do both
    config = {
                'dir' : 'input',
                'image_file_name' : 'test_input_image_00.fits',
                'cat_file_name' : 'test_input_cat_00.fits',
                'flag_col' : 'flag',
                'skip_flag' : '$2**2',
                'use_flag' : '$2**0',
             }
    input = piff.InputFiles(config, logger=logger)
    _, _, image_pos, _ = input.getRawImageData(0)
    print('len = ',len(image_pos))
    assert len(image_pos) == 68

    # If no skip_flag is specified, it skips all != 0.
    config = {
                'dir' : 'input',
                'image_file_name' : 'test_input_image_00.fits',
                'cat_file_name' : 'test_input_cat_00.fits',
                'flag_col' : 'flag',
             }
    input = piff.InputFiles(config, logger=logger)
    _, _, image_pos, _ = input.getRawImageData(0)
    print('len = ',len(image_pos))
    assert len(image_pos) == 12

    # Check property_cols gets set for stars' props_dict correctly
    cat_file_name = os.path.join('input', 'test_input_cat_00.fits')
    data = fitsio.read(cat_file_name)
    gr_color = data['gr_color']
    rz_color = data['rz_color']
    base_config = {
                'dir' : 'input',
                'image_file_name' : 'test_input_image_00.fits',
                'cat_file_name' : 'test_input_cat_00.fits',
             }
    input = piff.InputFiles(dict(property_cols=['gr_color'], **base_config), logger=logger)
    _, _, _, props_dict = input.getRawImageData(0)
    assert len(props_dict['gr_color']) == 100
    np.testing.assert_array_equal(props_dict['gr_color'], gr_color)
    input = piff.InputFiles(dict(property_cols=['gr_color', 'rz_color'], **base_config),
                            logger=logger)
    _, _, _, props_dict = input.getRawImageData(0)
    print(props_dict)
    assert len(props_dict['gr_color']) == 100
    assert len(props_dict['rz_color']) == 100
    np.testing.assert_array_equal(props_dict['gr_color'], gr_color)
    np.testing.assert_array_equal(props_dict['rz_color'], rz_color)

    # Check invalid column names
    base_config = {
                'dir' : 'input',
                'image_file_name' : 'test_input_image_00.fits',
                'cat_file_name' : 'test_input_cat_00.fits', }
    input = piff.InputFiles(dict(x_col='xx', **base_config))
    np.testing.assert_raises(ValueError, input.getRawImageData, 0)
    input = piff.InputFiles(dict(x_col='xx', y_col='y', **base_config))
    np.testing.assert_raises(ValueError, input.getRawImageData, 0)
    input = piff.InputFiles(dict(x_col='x', y_col='xx', **base_config))
    np.testing.assert_raises(ValueError, input.getRawImageData, 0)
    input = piff.InputFiles(dict(ra_col='xx', **base_config))
    np.testing.assert_raises(ValueError, input.getRawImageData, 0)
    input = piff.InputFiles(dict(ra_col='xx', dec_col='y', **base_config))
    np.testing.assert_raises(ValueError, input.getRawImageData, 0)
    input = piff.InputFiles(dict(ra_col='x', dec_col='xx', **base_config))
    np.testing.assert_raises(ValueError, input.getRawImageData, 0)
    input = piff.InputFiles(dict(sky_col='xx', **base_config))
    np.testing.assert_raises(ValueError, input.getRawImageData, 0)
    input = piff.InputFiles(dict(gain_col='xx', **base_config))
    np.testing.assert_raises(ValueError, input.getRawImageData, 0)
    input = piff.InputFiles(dict(flag_col='xx', **base_config))
    np.testing.assert_raises(ValueError, input.getRawImageData, 0)
    input = piff.InputFiles(dict(property_cols='invalid_string', **base_config))
    np.testing.assert_raises(ValueError, input.getRawImageData, 0)
    input = piff.InputFiles(dict(property_cols=['gr_color','invalid_col'], **base_config))
    np.testing.assert_raises(ValueError, input.getRawImageData, 0)

    # skip_flag, use_flag need to be integers
    np.testing.assert_raises(ValueError, piff.InputFiles,
                             dict(flag_col='flag', skip_flag='xx', **base_config))
    np.testing.assert_raises(ValueError, piff.InputFiles,
                             dict(flag_col='flag', use_flag='xx', **base_config))

    # Can't give duplicate sky, gain
    np.testing.assert_raises(ValueError, piff.InputFiles, dict(sky_col='sky', sky=3, **base_config))
    np.testing.assert_raises(ValueError, piff.InputFiles, dict(gain_col='gain', gain=3, **base_config))

    # Invalid header keys
    input = piff.InputFiles(dict(sky='sky', **base_config))
    np.testing.assert_raises(KeyError, input.getRawImageData, 0)
    input = piff.InputFiles(dict(gain='gain', **base_config))
    np.testing.assert_raises(KeyError, input.getRawImageData, 0)
    input = piff.InputFiles(dict(satur='satur', **base_config))
    np.testing.assert_raises(KeyError, input.getRawImageData, 0)


@timer
def test_flag_select():
    """Test the simple flag selection type
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = None

    # This repeats some of the tests in test_cols for using flags, but in that context
    # the flags were technically selecting which objects to consider, and then the select
    # field would select all of them as stars.  We didn't actually run the select above,
    # so do that here, just to confirm that step in the process.
    # Using flag will skip flagged columns.  Here every 5th item is flagged.
    config = {
        'input': {
                'dir' : 'input',
                'image_file_name' : 'test_input_image_00.fits',
                'cat_file_name' : 'test_input_cat_00.fits',
                'use_partial': True,
                'flag_col' : 'flag',
                'skip_flag' : 4
        },
        'select': {}
    }
    input = piff.InputFiles(config['input'], logger=logger)
    select = piff.FlagSelect(config['select'], logger=logger)
    stars = input.makeStars(logger=logger)
    stars = select.selectStars(stars, logger=logger)
    assert len(stars) == 80

    # However, we could do this by having input return all rows as objects, and then have
    # the select field pick out the subset with flag!=4.
    config['input']['skip_flag'] = 0
    config['select'] = {
            'flag_name' : 'flag',
            'skip_flag' : 4
    }
    input = piff.InputFiles(config['input'], logger=logger)
    select = piff.FlagSelect(config['select'], logger=logger)
    stars = input.makeStars(logger=logger)
    stars = select.selectStars(stars, logger=logger)
    assert len(stars) == 80

    # We could also read the flag column in as an extra property, rather than tell input that it
    # is a flag.  Then we don't have to set skip_flag in the input field.
    config['input'] = {
            'dir' : 'input',
            'image_file_name': 'test_input_image_00.fits',
            'cat_file_name': 'test_input_cat_00.fits',
            'use_partial': True,
            'property_cols': ['flag']
    }
    input = piff.InputFiles(config['input'], logger=logger)
    select = piff.FlagSelect(config['select'], logger=logger)
    stars = input.makeStars(logger=logger)
    stars = select.selectStars(stars, logger=logger)
    assert len(stars) == 80

    # Similarly the use columns will skip anything with use == 0 (every 7th item here)
    config['select'] = {
            'flag_name' : 'flag',
            'use_flag' : 1
    }
    input = piff.InputFiles(config['input'], logger=logger)
    select = piff.FlagSelect(config['select'], logger=logger)
    stars = input.makeStars(logger=logger)
    stars = select.selectStars(stars, logger=logger)
    assert len(stars) == 85

    # Can do both
    config['select'] = {
            'flag_name' : 'flag',
            'skip_flag' : 4,
            'use_flag' : 1,
    }
    input = piff.InputFiles(config['input'], logger=logger)
    select = piff.FlagSelect(config['select'], logger=logger)
    stars = input.makeStars(logger=logger)
    stars = select.selectStars(stars, logger=logger)
    assert len(stars) == 68

    # Can also have the input do some flag selection and then the select field do more.
    config = {
        'input': {
                'type': 'Files',  # This is not required, but it is allowed.
                'dir' : 'input',
                'image_file_name' : 'test_input_image_00.fits',
                'cat_file_name' : 'test_input_cat_00.fits',
                'use_partial': True,
                'flag_col' : 'flag',
                'skip_flag' : 4
        },
        'select': {
                'type': 'Flag',  # This is not required, but it is allowed.
                'flag_name' : 'flag',
                'use_flag' : 1,
        }
    }
    input = piff.InputFiles(config['input'], logger=logger)
    select = piff.FlagSelect(config['select'], logger=logger)
    stars = input.makeStars(logger=logger)
    stars = select.selectStars(stars, logger=logger)
    assert len(stars) == 68

    # If no stars are selected, then the process function will raise an error.
    config['select']['use_flag'] = 4  # same as skip_flag, so none will be selected.
    input = piff.InputFiles(config['input'])
    select = piff.FlagSelect(config['select'])
    stars1 = input.makeStars()
    stars = select.selectStars(stars)
    assert len(stars) == 0
    with np.testing.assert_raises(RuntimeError):
        piff.Select.process(config['select'], stars1)

    # Raises at different place if all stars are rejected.
    config['select']['use_flag'] = 1
    config['select']['reject_where'] = 'True'
    with np.testing.assert_raises(RuntimeError):
        piff.Select.process(config['select'], stars1)
    del config['select']['reject_where']

    # Base class selectStars function is not implemented.
    select = piff.Select(config['select'])
    with np.testing.assert_raises(NotImplementedError):
        select.selectStars(stars1)

    # Error if flag_name is not in the property list
    config['select']['flag_name'] = 'invalid'
    select = piff.FlagSelect(config['select'])
    with np.testing.assert_raises(ValueError):
        select.selectStars(stars1, logger=logger)

    # Even if flag_name is valid, if the input didn't save it, it's won't work.
    del config['input']['flag_col']
    del config['input']['skip_flag']
    config['select']['flag_name'] = 'flag'
    input = piff.InputFiles(config['input'])
    select = piff.FlagSelect(config['select'])
    stars1 = input.makeStars()
    with np.testing.assert_raises(ValueError):
        select.selectStars(stars1, logger=logger)

@timer
def test_properties_select():
    """Test the Properties selection type
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = None

    # The Properties select type is more general than flag.  You can pretty much do any
    # calculation you want with any number of properties.  But here we keep things pretty
    # simple, doing basically the same thing as we did above for the flag selection.

    # This is equivalent to skip_flag = 4
    config = {
        'input': {
                'dir' : 'input',
                'image_file_name' : 'test_input_image_00.fits',
                'cat_file_name' : 'test_input_cat_00.fits',
                'use_partial': True,
                'flag_col' : 'flag',
                'skip_flag' : 0  # This instructs it not to skip anything yet.
        },
        'select': {
                'type': 'Properties',
                'where': 'flag & 4 == 0'
        }
    }
    config = galsim.config.CleanConfig(config)
    input = piff.InputFiles(config['input'], logger=logger)
    select = piff.PropertiesSelect(config['select'], logger=logger)
    objects = input.makeStars(logger=logger)
    stars = select.selectStars(objects, logger=logger)
    assert len(stars) == 80

    # We could also read the flag column in as an extra property, rather than tell input that it
    # is a flag.  Then we don't have to set skip_flag in the input field.
    config['input'] = {
            'dir' : 'input',
            'image_file_name': 'test_input_image_00.fits',
            'cat_file_name': 'test_input_cat_00.fits',
            'use_partial': True,
            'property_cols': ['flag']
    }
    config = galsim.config.CleanConfig(config)
    input = piff.InputFiles(config['input'], logger=logger)
    select = piff.PropertiesSelect(config['select'], logger=logger)
    objects = input.makeStars(logger=logger)
    stars = select.selectStars(objects, logger=logger)
    assert len(stars) == 80

    # This is equivalent to use_flag = 1
    config['select']['where'] = 'flag & 1 != 0'
    config = galsim.config.CleanConfig(config)
    select = piff.PropertiesSelect(config['select'], logger=logger)
    stars = select.selectStars(objects, logger=logger)
    assert len(stars) == 85

    # This is equivalent to use_flag = 1, skip_flag = 4
    config['select']['where'] = '(flag & 4 == 0) & (flag & 1 != 0)'
    config = galsim.config.CleanConfig(config)
    select = piff.PropertiesSelect(config['select'], logger=logger)
    stars = select.selectStars(objects, logger=logger)
    assert len(stars) == 68

    # If eval string doesn't work with numpy arrays, then it does the slower method.
    config['select']['where'] = '(flag & 4 == 0) and (flag & 1 != 0)'
    config = galsim.config.CleanConfig(config)
    select = piff.PropertiesSelect(config['select'], logger=logger)
    stars = select.selectStars(objects, logger=logger)
    assert len(stars) == 68

    # This is gratuitous here, but can use np, numpy, math modules if desired.
    config['select']['where'] = 'np.array(flag) & int(math.sqrt(16)) == numpy.zeros_like(flag)'
    config = galsim.config.CleanConfig(config)
    select = piff.PropertiesSelect(config['select'], logger=logger)
    stars = select.selectStars(objects, logger=logger)
    assert len(stars) == 80

    # Error if where isn't given
    del config['select']['where']
    config = galsim.config.CleanConfig(config)
    with np.testing.assert_raises(ValueError):
       piff.PropertiesSelect(config['select'], logger=logger)

    # Also if it uses invalid properties (Note: capitalization is respected.)
    config['select']['where'] = 'FLAG & 4 == 0'
    config = galsim.config.CleanConfig(config)
    select = piff.PropertiesSelect(config['select'], logger=logger)
    with np.testing.assert_raises(NameError):
        stars = select.selectStars(objects, logger=logger)

    # Finally, the reject option reject_where is basically the converse of this,
    # which may be easier in some use cases.
    config['select'] = {
        'type': 'Properties',
        'where': 'flag & 1 != 0',
        'reject_where': 'flag & 4 != 0'
    }
    select = piff.PropertiesSelect(config['select'], logger=logger)
    stars = select.selectStars(objects, logger=logger)
    stars = select.rejectStars(stars, logger=logger)
    assert len(stars) == 68

    # Can use reject_where for other types besides Properties
    config['select'] = {
        'type': 'Flag',
        'reject_where': '(flag & 4 != 0) | (flag & 1 == 0)'
    }
    select = piff.FlagSelect(config['select'], logger=logger)
    stars = select.selectStars(objects, logger=logger)
    stars = select.rejectStars(stars, logger=logger)
    assert len(stars) == 68


@timer
def test_boolarray():
    """Test the ability to use a flag_col that is really a boolean array rather than ints.
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = None

    cat_file_name = os.path.join('input', 'test_input_cat_00.fits')
    data = fitsio.read(cat_file_name)
    print('flag = ',data['flag'])
    new_flag = np.empty((len(data), 50), dtype=bool)
    for bit in range(50):
        new_flag[:,bit] = data['flag'] & 2**bit != 0
    print('new_flag = ',new_flag)
    # Write out the catalog to a file
    print('dtype = ',new_flag.dtype)
    dtype = [ ('x','f8'), ('y','f8'), ('flag', bool, 50) ]
    new_data = np.empty(len(data), dtype=dtype)
    new_data['x'] = data['x']
    new_data['y'] = data['y']
    new_data['flag'] = new_flag
    new_cat_file_name = os.path.join('input','test_input_boolarray.fits')
    fitsio.write(new_cat_file_name, new_data, clobber=True)

    # Specifiable columns are: x, y, flag, use, sky, gain.  (We'll do flag, use below.)
    config = {
                'dir' : 'input',
                'image_file_name' : 'test_input_image_00.fits',
                'cat_file_name' : 'test_input_boolarray.fits',
                'x_col' : 'x',
                'y_col' : 'y',
                'flag_col' : 'flag',
                'skip_flag' : '$2**1 + 2**2 + 2**39'
             }
    input = piff.InputFiles(config, logger=logger)
    assert input.nimages == 1
    _, _, image_pos, _ = input.getRawImageData(0)
    print('len = ',len(image_pos))
    assert len(image_pos) == 80


@timer
def test_chipnum():
    """Test the ability to renumber the chipnums
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = None

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
        logger = None

    # If no weight or badpix is specified, the weights are all equal.
    config = {
                'image_file_name' : 'input/test_input_image_00.fits',
                'cat_file_name' : 'input/test_input_cat_00.fits',
             }
    input = piff.InputFiles(config, logger=logger)
    assert input.nimages == 1
    image, weight, image_pos, _ = input.getRawImageData(0)
    assert len(image_pos) == 100
    assert image.array.shape == (1024, 1024)
    assert weight.array.shape == (1024, 1024)
    np.testing.assert_array_equal(weight.array, 1.0)

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
    assert input.nimages == 1
    image, weight, image_pos, extra_props = input.getRawImageData(0)
    assert len(image_pos) == 100
    assert image.array.shape == (1024, 1024)
    assert weight.array.shape == (1024, 1024)
    sky = extra_props['sky'][0]
    gain = extra_props['gain'][0]
    read_noise = 10
    expected_noise = sky / gain + read_noise**2 / gain**2
    np.testing.assert_almost_equal(weight.array, expected_noise**-1)

    # Can set the noise by hand
    config = {
                'image_file_name' : 'input/test_input_image_00.fits',
                'cat_file_name' : 'input/test_input_cat_00.fits',
                'noise' : 32,
             }
    input = piff.InputFiles(config, logger=logger)
    assert input.nimages == 1
    _, weight, _, _ = input.getRawImageData(0)
    assert weight.array.shape == (1024, 1024)
    np.testing.assert_almost_equal(weight.array, 32.**-1)

    # Some old versions of fitsio had a bug where the badpix mask could be offset by 32768.
    # We move them back to 0
    config = {
                'image_file_name' : 'input/test_input_image_00.fits',
                'cat_file_name' : 'input/test_input_cat_00.fits',
                'weight_hdu' : 1,
                'badpix_hdu' : 5,
             }
    input = piff.InputFiles(config, logger=logger)
    assert input.nimages == 1
    _, weight, _, _ = input.getRawImageData(0)
    assert weight.array.shape == (1024, 1024)
    np.testing.assert_almost_equal(weight.array, expected_noise**-1)

    config['badpix_hdu'] = 6
    input = piff.InputFiles(config, logger=logger)
    assert input.nimages == 1
    _, weight, _, _ = input.getRawImageData(0)
    assert weight.array.shape == (1024, 1024)
    np.testing.assert_almost_equal(weight.array, expected_noise**-1)

    # Various ways to get all weight values == 0 (which will emit a logger message, but isn't
    # an error).
    config['weight_hdu'] = 1
    config['badpix_hdu'] = 7  # badpix > 0
    input = piff.InputFiles(config, logger=logger)
    assert input.nimages == 1
    _, weight, _, _ = input.getRawImageData(0)
    assert weight.array.shape == (1024, 1024)
    np.testing.assert_almost_equal(weight.array, 0.)

    config['weight_hdu'] = 3  # wt = 0
    config['badpix_hdu'] = 2
    input = piff.InputFiles(config, logger=logger)
    assert input.nimages == 1
    _, weight, _, _ = input.getRawImageData(0)
    assert weight.array.shape == (1024, 1024)
    np.testing.assert_almost_equal(weight.array, 0.)

    config['weight_hdu'] = 8  # Even cols are = 0
    config['badpix_hdu'] = 9  # Odd cols are > 0
    input = piff.InputFiles(config, logger=logger)
    assert input.nimages == 1
    _, weight, _, _ = input.getRawImageData(0)
    assert weight.array.shape == (1024, 1024)
    np.testing.assert_almost_equal(weight.array, 0.)

    # Negative valued weights are invalid
    config['weight_hdu'] = 4
    input = piff.InputFiles(config)
    with CaptureLog() as cl:
        _, weight, _, _  = input.getRawImageData(0, logger=cl.logger)
    assert 'Warning: weight map has invalid negative-valued pixels.' in cl.output


@timer
def test_lsst_weight():
    """Test the way LSSTDM stores the weight (as a variance including signal)
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = None

    # First with a sky level.  This isn't actually how LSST calexps are made (they are sky
    # subtracted), but it is how the input images are made.
    config = {
                'image_file_name' : 'input/test_input_image_00.fits',
                'cat_file_name' : 'input/test_input_cat_00.fits',
                'weight_hdu' : 10,
                'sky' : 'SKYLEVEL',
                'gain' : 'GAIN_A',
                'invert_weight' : True,
                'remove_signal_from_weight' : True,
             }
    input = piff.InputFiles(config, logger=logger)
    assert input.nimages == 1
    image, weight, image_pos, props = input.getRawImageData(0)
    assert len(image_pos) == 100
    assert image.array.shape == (1024, 1024)
    assert weight.array.shape == (1024, 1024)
    gain = props['gain'][0]
    sky = props['sky'][0]
    read_noise = 10
    expected_noise = sky / gain + read_noise**2 / gain**2
    print('expected noise = ',expected_noise)
    print('var = ',1./weight.array)
    np.testing.assert_allclose(weight.array, expected_noise**-1, rtol=1.e-6)

    # If the gain is not given, it can determine it automatically.
    config = {
                'image_file_name' : 'input/test_input_image_00.fits',
                'cat_file_name' : 'input/test_input_cat_00.fits',
                'weight_hdu' : 10,
                'sky' : 'SKYLEVEL',
                'invert_weight' : True,
                'remove_signal_from_weight' : True,
             }
    input = piff.InputFiles(config, logger=logger)
    assert input.nimages == 1
    image, weight, image_pos, props  = input.getRawImageData(0)
    np.testing.assert_allclose(props['gain'], gain, rtol=1.e-6)
    np.testing.assert_allclose(weight.array, expected_noise**-1, rtol=1.e-6)

    # Now pretend that the sky is part of the signal, so the input can match how we would
    # do this when running on calexps.
    config = {
                'image_file_name' : 'input/test_input_image_00.fits',
                'cat_file_name' : 'input/test_input_cat_00.fits',
                'weight_hdu' : 10,
                'gain' : 'GAIN_A',
                'invert_weight' : True,
                'remove_signal_from_weight' : True,
             }
    input = piff.InputFiles(config, logger=logger)
    assert input.nimages == 1
    image, weight, image_pos, props = input.getRawImageData(0)
    assert len(image_pos) == 100
    assert image.array.shape == (1024, 1024)
    assert weight.array.shape == (1024, 1024)
    gain = props['gain'][0]
    read_noise = 10
    expected_noise = read_noise**2 / gain**2
    print('expected noise = ',expected_noise)
    print('var = ',1./weight.array)
    np.testing.assert_allclose(weight.array, expected_noise**-1, rtol=1.e-5)


@timer
def test_stars():
    """Test the input.makeStars function
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = None

    dir = 'input'
    image_file = 'test_input_image_00.fits'
    cat_file = 'test_input_cat_00.fits'

    # Turn off two defaults for now (max_snr=100 and use_partial=False)
    config = {
        'input' : {
                'dir' : dir,
                'image_file_name' : image_file,
                'cat_file_name' : cat_file,
                'weight_hdu' : 1,
                'sky_col' : 'sky',
                'gain_col' : 'gain',
                'use_partial' : True,
             },
        'select': {
                'max_snr' : 0,
        }
    }
    input = piff.InputFiles(config['input'], logger=logger)
    select = piff.FlagSelect(config['select'], logger=logger)
    stars = input.makeStars(logger=logger)
    stars = select.rejectStars(stars, logger=logger)
    assert len(stars) == 100
    chipnum_list = [ star['chipnum'] for star in stars ]
    gain_list = [ star['gain'] for star in stars ]
    snr_list = [ star['snr'] for star in stars ]
    snr_list2 = [ piff.util.calculateSNR(star.data.image, star.data.orig_weight) for star in stars ]
    print('snr = ', np.min(snr_list), np.max(snr_list))
    np.testing.assert_array_equal(chipnum_list, 0)
    np.testing.assert_array_equal(gain_list, gain_list[0])
    np.testing.assert_almost_equal(snr_list, snr_list2, decimal=5)
    print('min_snr = ',np.min(snr_list))
    print('max_snr = ',np.max(snr_list))
    assert np.min(snr_list) < 30.
    assert np.max(snr_list) > 600.

    # max_snr increases the noise to achieve a maximum snr
    config['select']['max_snr'] = 120
    select = piff.FlagSelect(config['select'], logger=logger)
    stars = input.makeStars(logger=logger)
    stars = select.rejectStars(stars, logger=logger)
    assert len(stars) == 100
    snr_list = [ star['snr'] for star in stars ]
    print('snr = ', np.min(snr_list), np.max(snr_list))
    assert np.min(snr_list) < 30.
    assert np.max(snr_list) == 120.
    snr_list2 = [ piff.util.calculateSNR(star.data.image, star.data.orig_weight) for star in stars ]
    snr_list = np.array(snr_list)
    snr_list2 = np.array(snr_list2)
    lo = np.where(snr_list < 120)
    hi = np.where(snr_list == 120)
    # Uncorrected stars still have the same snr
    np.testing.assert_almost_equal(snr_list[lo], snr_list2[lo], decimal=5)
    # Corrected ones come out a little lower than the target.
    assert np.all(snr_list2[hi] <= 120.)
    assert np.all(snr_list2[hi] > 110.)

    # The default is max_snr == 100
    del config['select']['max_snr']
    select = piff.FlagSelect(config['select'], logger=logger)
    stars = input.makeStars(logger=logger)
    stars = select.rejectStars(stars, logger=logger)
    assert len(stars) == 100
    snr_list = np.array([ star['snr'] for star in stars ])
    print('snr = ', np.min(snr_list), np.max(snr_list))
    assert np.min(snr_list) < 30.
    assert np.max(snr_list) == 100.
    snr_list2 = [ piff.util.calculateSNR(star.data.image, star.data.orig_weight) for star in stars ]
    snr_list = np.array(snr_list)
    snr_list2 = np.array(snr_list2)
    lo = np.where(snr_list < 100)
    hi = np.where(snr_list == 100)
    np.testing.assert_almost_equal(snr_list[lo], snr_list2[lo], decimal=5)
    assert np.all(snr_list2[hi] <= 100.)

    # min_snr removes stars with a snr < min_snr
    config['select']['min_snr'] = 50
    select = piff.FlagSelect(config['select'], logger=logger)
    stars = input.makeStars(logger=logger)
    stars = select.rejectStars(stars, logger=logger)
    print('len should be ',len(snr_list[snr_list >= 50]))
    print('actual len is ',len(stars))
    assert len(stars) == len(snr_list[snr_list >= 50])
    assert len(stars) == 94  # hard-coded for this case, just to make sure
    snr_list = np.array([ star['snr'] for star in stars ])
    print('snr = ', np.min(snr_list), np.max(snr_list))
    assert np.min(snr_list) >= 50.
    assert np.max(snr_list) == 100.
    snr_list2 = [ piff.util.calculateSNR(star.data.image, star.data.orig_weight) for star in stars ]
    snr_list = np.array(snr_list)
    snr_list2 = np.array(snr_list2)
    lo = np.where(snr_list < 100)
    hi = np.where(snr_list == 100)
    np.testing.assert_almost_equal(snr_list[lo], snr_list2[lo], decimal=5)
    assert np.all(snr_list2[hi] <= 100.)

    # use_partial=False will skip any stars that are partially off the edge of the image
    config['input']['use_partial'] = False
    input = piff.InputFiles(config['input'], logger=logger)
    stars = input.makeStars(logger=logger)
    stars = select.rejectStars(stars, logger=logger)
    print('new len is ',len(stars))
    assert len(stars) == 90  # skipped 4 additional stars

    # use_partial=False is the default
    del config['input']['use_partial']
    input = piff.InputFiles(config['input'], logger=logger)
    stars = input.makeStars(logger=logger)
    stars = select.rejectStars(stars, logger=logger)
    assert len(stars) == 90

    # Setting satur will skip any stars with a pixel above that value.
    # Here there is 1 star with a pixel > 1890
    config['input']['satur'] = 'SATURAT'
    input = piff.InputFiles(config['input'], logger=logger)
    stars = input.makeStars(logger=logger)
    stars = select.rejectStars(stars, logger=logger)
    assert len(stars) == 89
    # 3 more stars have pixels > 1850
    config['input']['satur'] = 1850
    input = piff.InputFiles(config['input'], logger=logger)
    stars = input.makeStars(logger=logger)
    stars = select.rejectStars(stars, logger=logger)
    assert len(stars) == 86
    del config['input']['satur']

    # maxpixel_cut is almost the same thing, but figures out an equivalent flux cut and uses
    # that to avoid imparting a size selection bias.
    # For this set, it pulls in a few more to reject.
    config['select']['max_pixel_cut'] = 1850
    input = piff.InputFiles(config['input'], logger=logger)
    select = piff.FlagSelect(config['select'], logger=logger)
    stars = input.makeStars(logger=logger)
    stars = select.rejectStars(stars, logger=logger)
    assert len(stars) == 79

    # Gratuitous coverage test.  If all objects have snr < 40, then max_pixel_cut doesn't
    # remove anything, since it only considers stars with snr > 40.
    config['select']['max_snr'] = 30
    input = piff.InputFiles(config['input'], logger=logger)
    select = piff.FlagSelect(config['select'], logger=logger)
    stars = input.makeStars(logger=logger)
    stars = select.rejectStars(stars, logger=logger)
    assert len(stars) == 90
    del config['select']['max_snr']
    del config['select']['max_pixel_cut']

    # hsm_size_reject=True rejects a few of these.  But mostly objects with neighbors.
    config['select']['hsm_size_reject'] = True
    input = piff.InputFiles(config['input'], logger=logger)
    select = piff.FlagSelect(config['select'], logger=logger)
    stars = input.makeStars(logger=logger)
    stars = select.rejectStars(stars, logger=logger)
    assert len(stars) == 88

    # hsm_size_reject can also be a float.  (True is equivalent to 10.)
    config['select']['hsm_size_reject'] = 100.
    select = piff.FlagSelect(config['select'], logger=logger)
    stars = input.makeStars(logger=logger)
    stars = select.rejectStars(stars, logger=logger)
    assert len(stars) == 90
    config['select']['hsm_size_reject'] = 3.
    select = piff.FlagSelect(config['select'], logger=logger)
    stars = input.makeStars(logger=logger)
    stars = select.rejectStars(stars, logger=logger)
    assert len(stars) == 85
    config['select']['hsm_size_reject'] = 10.
    select = piff.FlagSelect(config['select'], logger=logger)
    stars = input.makeStars(logger=logger)
    stars = select.rejectStars(stars, logger=logger)
    assert len(stars) == 88
    del config['select']['hsm_size_reject']

    # alt_x and alt_y also include some object completely off the image, which are always skipped.
    # (Don't do the min_snr anymore, since most of these stamps don't actually have any signal.)
    config['input']['x_col'] = 'alt_x'
    config['input']['y_col'] = 'alt_y'
    del config['select']['min_snr']
    input = piff.InputFiles(config['input'], logger=logger)
    select = piff.FlagSelect(config['select'], logger=logger)
    stars = input.makeStars(logger=logger)
    stars = select.rejectStars(stars, logger=logger)
    print('new len is ',len(stars))
    assert len(stars) == 36

    # Also skip objects which are all weight=0
    config['input']['weight_hdu'] = 3
    input = piff.InputFiles(config['input'], logger=logger)
    stars = input.makeStars(logger=logger)
    stars = select.rejectStars(stars, logger=logger)
    print('new len is ',len(stars))
    assert len(stars) == 0

    # But not ones that are only partially weight=0
    config['input']['weight_hdu'] = 8
    input = piff.InputFiles(config['input'], logger=logger)
    stars = input.makeStars(logger=logger)
    stars = select.rejectStars(stars, logger=logger)
    print('new len is ',len(stars))
    assert len(stars) == 36

    # Check the masked pixel cut
    # This is a bit artificial, b/c 512 / 1024 of the pixels are masked in the test case
    del config['input']['x_col']
    del config['input']['y_col']
    config['input']['weight_hdu'] = 8
    config['select']['max_mask_pixels'] = 513
    input = piff.InputFiles(config['input'], logger=logger)
    select = piff.FlagSelect(config['select'], logger=logger)
    stars = input.makeStars(logger=logger)
    stars = select.rejectStars(stars, logger=logger)
    print('new len is ',len(stars))
    assert len(stars) == 95

    config['select']['max_mask_pixels'] = 500
    select = piff.FlagSelect(config['select'], logger=logger)
    stars = input.makeStars(logger=logger)
    stars = select.rejectStars(stars, logger=logger)
    print('new len is ',len(stars))
    assert len(stars) == 0

    # Check the edge fraction cut
    # with use_partial=True to make sure it catches edge case
    del config['select']['max_mask_pixels']
    config['select']['max_edge_frac'] = 0.25
    config['input']['use_partial'] = True
    input = piff.InputFiles(config['input'], logger=logger)
    select = piff.FlagSelect(config['select'], logger=logger)
    stars = input.makeStars(logger=logger)
    stars = select.rejectStars(stars, logger=logger)
    print('new len is ',len(stars))
    assert len(stars) == 91

    # Check that negative snr flux yields 0, not an error (from sqrt(neg))
    # Negative flux is actually ok, since it gets squared, but if an image has negative weights
    # (which would be weird of course), then it could get to negative flux = wI^2.
    star0 = stars[0]
    star0.data.orig_weight *= -1.
    snr0 = piff.util.calculateSNR(star0.data.image, star0.data.orig_weight)
    assert snr0 == 0.


@timer
def test_pointing():
    """Test the input.setPointing function
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = None

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
    np.testing.assert_almost_equal(input.pointing.ra.rad, np.pi/2.)
    np.testing.assert_almost_equal(input.pointing.dec.rad, -np.pi/6.)

    # Also ok as ints in this case
    config['ra'] = 6
    config['dec'] = -30
    input = piff.InputFiles(config, logger=logger)
    np.testing.assert_almost_equal(input.pointing.ra.rad, np.pi/2.)
    np.testing.assert_almost_equal(input.pointing.dec.rad, -np.pi/6.)

    # Strings as hh:mm:ss or dd:mm:ss
    config['ra'] = '06:00:00'
    config['dec'] = '-30:00:00'
    input = piff.InputFiles(config, logger=logger)
    np.testing.assert_almost_equal(input.pointing.ra.rad, np.pi/2.)
    np.testing.assert_almost_equal(input.pointing.dec.rad, -np.pi/6.)

    # Strings as keys into FITS header
    config['ra'] = 'RA'
    config['dec'] = 'DEC'
    input = piff.InputFiles(config, logger=logger)
    np.testing.assert_almost_equal(input.pointing.ra.rad, np.pi/2.)
    np.testing.assert_almost_equal(input.pointing.dec.rad, -np.pi/6.)

    # If multiple files, use the first one.
    config['image_file_name'] = 'test_input_image_*.fits'
    config['cat_file_name'] = 'test_input_cat_*.fits'
    input = piff.InputFiles(config, logger=logger)
    np.testing.assert_almost_equal(input.pointing.ra.rad, np.pi/2.)
    np.testing.assert_almost_equal(input.pointing.dec.rad, -np.pi/6.)

    # Check invalid ra,dec values
    base_config = { 'dir' : dir, 'image_file_name' : image_file, 'cat_file_name' : cat_file }
    np.testing.assert_raises(ValueError, piff.InputFiles,
                             dict(ra=0, dec='00:00:00' , **base_config))
    np.testing.assert_raises(ValueError, piff.InputFiles,
                             dict(ra='00:00:00', dec=0, **base_config))
    np.testing.assert_raises(ValueError, piff.InputFiles,
                             dict(ra=0*galsim.degrees, dec=0*galsim.radians, **base_config))
    np.testing.assert_raises(KeyError, piff.InputFiles,
                             dict(ra='bad_ra', dec='DEC', **base_config))
    np.testing.assert_raises(KeyError, piff.InputFiles,
                             dict(ra='RA', dec='bad_dec', **base_config))
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

@timer
def test_sky():
    """Test the different ways to specify a sky to subtract.
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = None

    # First get the image without any sky subtraction.
    image_file = 'input/test_input_image_00.fits'
    cat_file = 'input/test_input_cat_00.fits'
    sky_file = 'input/test_input_sky_00.fits.fz'
    config = {
                'image_file_name' : image_file,
                'cat_file_name' : cat_file,
                'use_partial' : True,
             }
    input = piff.InputFiles(config, logger=logger)
    raw_image, _, _, _ = input.getRawImageData(0)
    raw_stars = input.makeStars(logger=logger)
    assert raw_image.array.shape == (1024, 1024)
    assert len(raw_stars) == 100

    # With a sky_col, it will subtract that value from each star's image
    config = {
                'image_file_name' : image_file,
                'cat_file_name' : cat_file,
                'use_partial' : True,
                'sky_col' : 'sky',
             }
    input = piff.InputFiles(config, logger=logger)
    image, _, _, extra_props = input.getRawImageData(0)
    stars = input.makeStars(logger=logger)
    np.testing.assert_allclose(image.array, raw_image.array)
    assert len(extra_props['sky']) == 100
    for i in range(len(stars)):
        assert stars[i]['sky'] == extra_props['sky'][i]
        np.testing.assert_allclose(stars[i].image.array, raw_stars[i].image.array - stars[i]['sky'])

    # If sky is given as a constant value, then that number is used.
    config = {
                'image_file_name' : image_file,
                'cat_file_name' : cat_file,
                'use_partial' : True,
                'sky' : 7,
             }
    input = piff.InputFiles(config, logger=logger)
    image, _, _, extra_props = input.getRawImageData(0)
    stars = input.makeStars(logger=logger)
    np.testing.assert_allclose(image.array, raw_image.array)
    for i in range(len(stars)):
        assert extra_props['sky'][i] == 7
        np.testing.assert_allclose(stars[i].image.array, raw_stars[i].image.array - 7)

    # If sky is given as a string, it is a header key word.
    config = {
                'image_file_name' : image_file,
                'cat_file_name' : cat_file,
                'use_partial' : True,
                'sky' : 'SKYLEVEL',
             }
    input = piff.InputFiles(config, logger=logger)
    image, _, _, extra_props = input.getRawImageData(0)
    stars = input.makeStars(logger=logger)
    np.testing.assert_allclose(image.array, raw_image.array)
    with fitsio.FITS(image_file, 'r') as f:
        sky = f[0].read_header()['SKYLEVEL']
    for i in range(len(stars)):
        assert extra_props['sky'][i] == sky
        np.testing.assert_allclose(stars[i].image.array, raw_stars[i].image.array - sky)

    # If sky_file_name is given then it is an image file to subtract
    # In this case, we made it so this image is a constant with values equal to SKYLEVEL from
    # the previous test, so keep the same value for sky.
    config = {
                'image_file_name' : image_file,
                'cat_file_name' : cat_file,
                'use_partial' : True,
                'sky_file_name' : sky_file,
             }
    input = piff.InputFiles(config, logger=logger)
    image, _, _, extra_props = input.getRawImageData(0)
    stars = input.makeStars(logger=logger)
    np.testing.assert_allclose(image.array, raw_image.array - sky)
    assert 'sky' not in extra_props  # No sky property in this case.
    for i in range(len(stars)):
        np.testing.assert_allclose(stars[i].image.array, raw_stars[i].image.array - sky)

    # Can provide an explicit hdu, although the default works fine in this case.
    config['sky_hdu'] = 1
    input = piff.InputFiles(config, logger=logger)
    image, _, _, extra_props = input.getRawImageData(0)
    stars = input.makeStars(logger=logger)
    np.testing.assert_allclose(image.array, raw_image.array - sky)
    assert 'sky' not in extra_props
    for i in range(len(stars)):
        np.testing.assert_allclose(stars[i].image.array, raw_stars[i].image.array - sky)

    # It can also use a provided dir
    config = {
                'dir' : 'input',
                'image_file_name' : image_file[6:],
                'cat_file_name' : cat_file[6:],
                'use_partial' : True,
                'sky_file_name' : sky_file[6:],
             }
    input = piff.InputFiles(config, logger=logger)
    image, _, _, extra_props = input.getRawImageData(0)
    stars = input.makeStars(logger=logger)
    np.testing.assert_allclose(image.array, raw_image.array - sky)
    assert 'sky' not in extra_props
    for i in range(len(stars)):
        np.testing.assert_allclose(stars[i].image.array, raw_stars[i].image.array - sky)

    # As with the other file names, sky_file_name can be parsed as a dict.
    config = {
                'image_file_name' : image_file,
                'cat_file_name' : cat_file,
                'use_partial' : True,
                'sky_file_name' : { 'type': 'List', 'items': [sky_file], },
             }
    input = piff.InputFiles(config, logger=logger)
    image, _, _, extra_props = input.getRawImageData(0)
    stars = input.makeStars(logger=logger)
    np.testing.assert_allclose(image.array, raw_image.array - sky)
    assert 'sky' not in extra_props
    for i in range(len(stars)):
        np.testing.assert_allclose(stars[i].image.array, raw_stars[i].image.array - sky)

    # Error if sky_file doesn't exist
    config['sky_file_name'] = sky_file[-2]
    with np.testing.assert_raises(ValueError):
        piff.InputFiles(config, logger=logger)

    # Also if not either a string or dict.
    config['sky_file_name'] = 77
    with np.testing.assert_raises(ValueError):
        piff.InputFiles(config, logger=logger)


if __name__ == '__main__':
    setup()
    test_basic()
    test_invalid()
    test_cols()
    test_boolarray()
    test_chipnum()
    test_weight()
    test_lsst_weight()
    test_stars()
    test_pointing()
