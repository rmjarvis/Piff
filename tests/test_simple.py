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
import galsim
import numpy as np
import piff
import os
import sys
import subprocess
import yaml
import fitsio
import copy

from piff_test_helper import get_script_name, timer, CaptureLog


@timer
def test_Gaussian():
    """This is about the simplest possible model I could think of.  It just uses the
    HSM adaptive moments routine to measure the moments, and then it models the
    PSF as a Gaussian.
    """

    # Here is the true PSF
    sigma = 1.3
    g1 = 0.23
    g2 = -0.17
    psf = galsim.Gaussian(sigma=sigma).shear(g1=g1, g2=g2)

    # Draw the PSF onto an image.  Let's go ahead and give it a non-trivial WCS.
    wcs = galsim.JacobianWCS(0.26, 0.05, -0.08, -0.29)
    image = galsim.Image(64,64, wcs=wcs)
    # This is only going to come out right if we (unphysically) don't convolve by the pixel.
    psf.drawImage(image, method='no_pixel')

    # Make a StarData instance for this image
    stardata = piff.StarData(image, image.true_center)
    star = piff.Star(stardata, None)

    # Fit the model from the image
    model = piff.Gaussian(include_pixel=False)
    star = model.initialize(star)
    fit = model.fit(star).fit

    print('True sigma = ',sigma,', model sigma = ',fit.params[0])
    print('True g1 = ',g1,', model g1 = ',fit.params[1])
    print('True g2 = ',g2,', model g2 = ',fit.params[2])

    # This test is pretty accurate, since we didn't add any noise and didn't convolve by
    # the pixel, so the image is very accurately a sheared Gaussian.
    true_params = [ sigma, g1, g2 ]
    np.testing.assert_almost_equal(fit.params[0], sigma, decimal=7)
    np.testing.assert_almost_equal(fit.params[1], g1, decimal=7)
    np.testing.assert_almost_equal(fit.params[2], g2, decimal=7)
    np.testing.assert_almost_equal(fit.params, true_params, decimal=7)

    # Now test running it via the config parser
    config = {
        'model' : {
            'type' : 'Gaussian',
            'include_pixel': False
        }
    }
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(log_file='output/test_Gaussian.log')
    model = piff.Model.process(config['model'], logger)
    fit = model.fit(star).fit

    # Same tests.
    np.testing.assert_almost_equal(fit.params[0], sigma, decimal=7)
    np.testing.assert_almost_equal(fit.params[1], g1, decimal=7)
    np.testing.assert_almost_equal(fit.params[2], g2, decimal=7)
    np.testing.assert_almost_equal(fit.params, true_params, decimal=7)


@timer
def test_Mean():
    """For the interpolation, the simplest possible model is just a mean value, which barely
    even qualifies as doing any kind of interpolating.  But it tests the basic glue software.
    """
    # Make a list of parameter vectors to "interpolate"
    np_rng = np.random.RandomState(1234)
    nstars = 100
    vectors = [ np_rng.random_sample(10) for i in range(nstars) ]
    mean = np.mean(vectors, axis=0)
    print('mean = ',mean)

    # Make some dummy StarData objects to use.  The only thing we really need is the properties,
    # although for the Mean interpolator, even this is ignored.
    target_data = [
            piff.Star.makeTarget(x=np_rng.random_sample()*2048, y=np_rng.random_sample()*2048).data
            for i in range(nstars) ]
    fit = [ piff.StarFit(v) for v in vectors ]
    stars = [ piff.Star(d, f) for d,f in zip(target_data,fit) ]

    # Use the piff.Mean interpolator
    interp = piff.Mean()
    interp.solve(stars)

    print('True mean = ',mean)
    print('Interp mean = ',interp.mean)

    # This should be exactly equal, since we did the same calculation.  But use almost_equal
    # anyway, just in case we decide to do something slightly different, but equivalent.
    np.testing.assert_almost_equal(mean, interp.mean)

    # Now test running it via the config parser
    config = {
        'interp' : {
            'type' : 'Mean'
        }
    }
    logger = piff.config.setup_logger()
    interp = piff.Interp.process(config['interp'], logger)
    interp.solve(stars)
    np.testing.assert_almost_equal(mean, interp.mean)


@timer
def test_single_image():
    """Test the simple case of one image and one catalog.
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(log_file='output/test_single_image.log')

    # Make the image
    image = galsim.Image(2048, 2048, scale=0.26)

    # Where to put the stars.  Include some flagged and not used locations.
    x_list = [ 123.12, 345.98, 567.25, 1094.94, 924.15, 1532.74, 1743.11, 888.39, 1033.29, 1409.31 ]
    y_list = [ 345.43, 567.45, 1094.32, 924.29, 1532.92, 1743.83, 888.83, 1033.19, 1409.20, 123.11 ]
    flag_list = [ 1, 1, 13, 1, 1, 4, 1, 1, 0, 1 ]

    # Draw a Gaussian PSF at each location on the image.
    sigma = 1.3
    g1 = 0.23
    g2 = -0.17
    psf = galsim.Gaussian(sigma=sigma).shear(g1=g1, g2=g2)
    for x,y,flag in zip(x_list, y_list, flag_list):
        bounds = galsim.BoundsI(int(x-31), int(x+32), int(y-31), int(y+32))
        offset = galsim.PositionD( x-int(x)-0.5 , y-int(y)-0.5 )
        psf.drawImage(image=image[bounds], method='no_pixel', offset=offset)
        # corrupt the ones that are marked as flagged
        if flag & 4:
            print('corrupting star at ',x,y)
            ar = image[bounds].array
            im_max = np.max(ar) * 0.2
            ar[ar > im_max] = im_max
    image.addNoise(galsim.GaussianNoise(rng=galsim.BaseDeviate(1234), sigma=1e-6))

    # Write out the image to a file
    image_file = os.path.join('output','simple_image.fits')
    image.write(image_file)

    # Write out the catalog to a file
    dtype = [ ('x','f8'), ('y','f8'), ('flag','i2') ]
    data = np.empty(len(x_list), dtype=dtype)
    data['x'] = x_list
    data['y'] = y_list
    data['flag'] = flag_list
    cat_file = os.path.join('output','simple_cat.fits')
    fitsio.write(cat_file, data, clobber=True)

    # Use InputFiles to read these back in
    config = { 'image_file_name' : image_file,
               'cat_file_name': cat_file }
    input = piff.InputFiles(config, logger=logger)
    assert input.image_file_name == [ image_file ]
    assert input.cat_file_name == [ cat_file ]

    # Check image
    assert input.nimages == 1
    image1, _, image_pos, _ = input.getRawImageData(0)
    np.testing.assert_equal(image1.array, image.array)

    # Check catalog
    np.testing.assert_equal([pos.x for pos in image_pos], x_list)
    np.testing.assert_equal([pos.y for pos in image_pos], y_list)

    # Repeat, using flag columns this time.
    config = { 'image_file_name' : image_file,
               'cat_file_name': cat_file,
               'flag_col': 'flag',
               'use_flag': '1',
               'skip_flag': '4',
               'stamp_size': 48 }
    input = piff.InputFiles(config, logger=logger)
    assert input.nimages == 1
    _, _, image_pos, _ = input.getRawImageData(0)
    assert len(image_pos) == 7

    # Make star data
    orig_stars = input.makeStars()
    assert len(orig_stars) == 7
    assert orig_stars[0].image.array.shape == (48,48)

    # Process the star data
    # can only compare to truth if include_pixel=False
    model = piff.Gaussian(fastfit=True, include_pixel=False)
    interp = piff.Mean()
    fitted_stars = [ model.fit(model.initialize(star)) for star in orig_stars ]
    interp.solve(fitted_stars)
    print('mean = ',interp.mean)

    # Check that the interpolation is what it should be
    # Any position would work here.
    chipnum = 0
    x = 1024
    y = 123
    orig_wcs = input.getWCS()[chipnum]
    orig_pointing = input.getPointing()
    image_pos = galsim.PositionD(x,y)
    world_pos = piff.StarData.calculateFieldPos(image_pos, orig_wcs, orig_pointing)
    u,v = world_pos.x, world_pos.y
    stamp_size = config['stamp_size']

    target = piff.Star.makeTarget(x=x, y=y, u=u, v=v, wcs=orig_wcs, stamp_size=stamp_size,
                                  pointing=orig_pointing)
    true_params = [ sigma, g1, g2 ]
    test_star = interp.interpolate(target)
    np.testing.assert_almost_equal(test_star.fit.params, true_params, decimal=4)

    # Check default values of options
    psf = piff.SimplePSF(model, interp)
    assert psf.chisq_thresh == 0.1
    assert psf.max_iter == 30
    assert psf.outliers == None
    assert psf.interp_property_names == ('u','v')

    # Now test running it via the config parser
    psf_file = os.path.join('output','simple_psf.fits')
    config = {
        'input' : {
            'image_file_name' : image_file,
            'cat_file_name' : cat_file,
            'flag_col' : 'flag',
            'use_flag' : 1,
            'skip_flag' : 4,
            'stamp_size' : stamp_size
        },
        'psf' : {
            'model' : { 'type' : 'Gaussian',
                        'fastfit': True,
                        'include_pixel': False},
            'interp' : { 'type' : 'Mean' },
            'max_iter' : 10,
            'chisq_thresh' : 0.2,
        },
        'output' : { 'file_name' : psf_file },
    }
    orig_stars, wcs, pointing = piff.Input.process(config['input'], logger)

    # Use a SimplePSF to process the stars data this time.
    interp = piff.Mean()
    psf = piff.SimplePSF(model, interp, max_iter=10, chisq_thresh=0.2)
    assert psf.chisq_thresh == 0.2
    assert psf.max_iter == 10

    # Error if input has no stars
    with np.testing.assert_raises(RuntimeError):
        psf.fit([], wcs, pointing, logger=logger)

    # Do the fit
    psf.fit(orig_stars, wcs, pointing, logger=logger)
    test_star = psf.interp.interpolate(target)
    np.testing.assert_almost_equal(test_star.fit.params, true_params, decimal=4)

    # test that drawStar and drawStarList work
    test_star = psf.drawStar(target)
    test_star_list = psf.drawStarList([target])[0]
    np.testing.assert_equal(test_star.fit.params, test_star_list.fit.params)
    np.testing.assert_equal(test_star.image.array, test_star_list.image.array)

    # test copy_image property of drawStar and draw
    for draw in [psf.drawStar, psf.model.draw]:
        target_star_copy = psf.interp.interpolate(piff.Star(target.data.copy(), target.fit.copy()))
        # interp is so that when we do psf.model.draw we have fit.params to work with

        test_star_copy = draw(target_star_copy, copy_image=True)
        test_star_nocopy = draw(target_star_copy, copy_image=False)
        # if we modify target_star_copy, then test_star_nocopy should be modified,
        # but not test_star_copy
        target_star_copy.image.array[0,0] = 23456
        assert test_star_nocopy.image.array[0,0] == target_star_copy.image.array[0,0]
        assert test_star_copy.image.array[0,0] != target_star_copy.image.array[0,0]
        # however the other pixels SHOULD still be all the same value
        assert test_star_nocopy.image.array[1,1] == target_star_copy.image.array[1,1]
        assert test_star_copy.image.array[1,1] == target_star_copy.image.array[1,1]

        test_star_center = draw(test_star_copy, copy_image=True, center=(x+1,y+1))
        np.testing.assert_almost_equal(test_star_center.image.array[1:,1:],
                                       test_star_copy.image.array[:-1,:-1])

    # test that draw works
    test_image = psf.draw(x=target['x'], y=target['y'], stamp_size=config['input']['stamp_size'],
                          flux=target.fit.flux, offset=target.fit.center)
    # this image should be the same values as test_star
    assert test_image == test_star.image
    # test that draw does not copy the image
    image_ref = psf.draw(x=target['x'], y=target['y'], stamp_size=config['input']['stamp_size'],
                         flux=target.fit.flux, offset=target.fit.center, image=test_image)
    image_ref.array[0,0] = 123456789
    assert test_image.array[0,0] == image_ref.array[0,0]
    assert test_star.image.array[0,0] != test_image.array[0,0]
    assert test_star.image.array[1,1] == test_image.array[1,1]

    # Round trip to a file
    psf.write(psf_file, logger)
    psf2 = piff.read(psf_file, logger)
    assert type(psf2.model) is piff.Gaussian
    assert type(psf2.interp) is piff.Mean
    assert psf2.chisq == psf.chisq
    assert psf2.last_delta_chisq == psf.last_delta_chisq
    assert psf2.chisq_thresh == psf.chisq_thresh
    assert psf2.max_iter == psf.max_iter
    assert psf2.dof == psf.dof
    assert psf2.nremoved == psf.nremoved
    test_star = psf2.interp.interpolate(target)
    np.testing.assert_almost_equal(test_star.fit.params, true_params, decimal=4)

    # Do the whole thing with the config parser
    os.remove(psf_file)

    piff.piffify(config, logger)
    psf3 = piff.read(psf_file)
    assert type(psf3.model) is piff.Gaussian
    assert type(psf3.interp) is piff.Mean
    assert psf3.chisq == psf.chisq
    assert psf3.last_delta_chisq == psf.last_delta_chisq
    assert psf3.chisq_thresh == psf.chisq_thresh
    assert psf3.max_iter == psf.max_iter
    assert psf3.dof == psf.dof
    assert psf3.nremoved == psf.nremoved
    test_star = psf3.interp.interpolate(target)
    np.testing.assert_almost_equal(test_star.fit.params, true_params, decimal=4)

    # Test using the piffify executable
    os.remove(psf_file)
    # This would be simpler as a direct assignment, but this once, test the way you would set
    # this from the command line, which would call parse_variables.
    piff.config.parse_variables(config, ['verbose=0'], logger=logger)
    #config['verbose'] = 0
    with open('simple.yaml','w') as f:
        f.write(yaml.dump(config, default_flow_style=False))
    config2 = piff.config.read_config('simple.yaml')
    assert config == config2
    piffify_exe = get_script_name('piffify')
    p = subprocess.Popen( [piffify_exe, 'simple.yaml'] )
    p.communicate()
    psf4 = piff.read(psf_file)
    assert type(psf4.model) is piff.Gaussian
    assert type(psf4.interp) is piff.Mean
    assert psf4.chisq == psf.chisq
    assert psf4.last_delta_chisq == psf.last_delta_chisq
    assert psf4.chisq_thresh == psf.chisq_thresh
    assert psf4.max_iter == psf.max_iter
    assert psf4.dof == psf.dof
    assert psf4.nremoved == psf.nremoved
    test_star = psf4.interp.interpolate(target)
    np.testing.assert_almost_equal(test_star.fit.params, true_params, decimal=4)

    # With very low max_iter, we hit the warning about non-convergence
    config['psf']['max_iter'] = 1
    with CaptureLog(level=1) as cl:
        piff.piffify(config, cl.logger)
    assert 'PSF fit did not converge' in cl.output

@timer
def test_invalid_config():
    # Test a few invalid uses of the config parsing.
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(log_file='output/test_invalid_config.log')
    image_file = os.path.join('output','simple_image.fits')
    cat_file = os.path.join('output','simple_cat.fits')
    psf_file = os.path.join('output','simple_psf.fits')
    config = {
        'input' : {
            'image_file_name' : image_file,
            'cat_file_name' : cat_file,
            'flag_col' : 'flag',
            'use_flag' : 1,
            'skip_flag' : 4,
        },
        'psf' : {
            'model' : { 'type' : 'Gaussian',
                        'fastfit': True,
                        'include_pixel': False},
            'interp' : { 'type' : 'Mean' },
            'max_iter' : 10,
            'chisq_thresh' : 0.2,
        },
        'output' : { 'file_name' : psf_file },
    }
    # Invalid variable specification
    with np.testing.assert_raises(ValueError):
        piff.parse_variables(config, ['verbose:0'], logger=logger)
    # process needs both input and psf
    with np.testing.assert_raises(ValueError):
        piff.process(config={'input':config['input']}, logger=logger)
    with np.testing.assert_raises(ValueError):
        piff.process(config={'psf':config['psf']}, logger=logger)
    # piffify also needs output
    with np.testing.assert_raises(ValueError):
        piff.piffify(config={'input':config['input']}, logger=logger)
    with np.testing.assert_raises(ValueError):
        piff.piffify(config={'psf':config['psf']}, logger=logger)
    with np.testing.assert_raises(ValueError):
        piff.piffify(config={'input':config['input'], 'psf':config['psf']}, logger=logger)
    # plotify doesn't need psf, but needs a 'file_name' in output
    with np.testing.assert_raises(ValueError):
        piff.plotify(config={'input':config['input']}, logger=logger)
    with np.testing.assert_raises(ValueError):
        piff.plotify(config={'input':config['input'], 'output':{}}, logger=logger)

    # Error if missing either model or interp
    config2 = copy.deepcopy(config)
    del config2['psf']['model']
    with np.testing.assert_raises(ValueError):
        piff.piffify(config2, logger)
    config2['psf']['model'] = config['psf']['model']
    del config2['psf']['interp']
    with np.testing.assert_raises(ValueError):
        piff.piffify(config2, logger)


@timer
def test_reserve():
    """Test the reserve_frac option.
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(log_file='output/test_single_reserve.log')

    # Make the image
    image = galsim.Image(2048, 2048, scale=0.26)

    # Where to put the stars.
    x_list = [ 123.12, 345.98, 567.25, 1094.94, 924.15, 1532.74, 1743.11, 888.39, 1033.29, 1409.31 ]
    y_list = [ 345.43, 567.45, 1094.32, 924.29, 1532.92, 1743.83, 888.83, 1033.19, 1409.20, 123.11 ]

    # Draw a Gaussian PSF at each location on the image.
    sigma = 1.3
    g1 = 0.23
    g2 = -0.17
    true_params = [ sigma, g1, g2 ]
    psf = galsim.Gaussian(sigma=sigma).shear(g1=g1, g2=g2)
    for x,y in zip(x_list, y_list):
        bounds = galsim.BoundsI(int(x-31), int(x+32), int(y-31), int(y+32))
        psf.drawImage(image[bounds], center=galsim.PositionD(x,y), method='no_pixel')
    image.addNoise(galsim.GaussianNoise(rng=galsim.BaseDeviate(1234), sigma=1e-6))

    # Write out the image to a file
    image_file = os.path.join('output','test_simple_reserve_image.fits')
    image.write(image_file)

    # Write out the catalog to a file
    dtype = [ ('x','f8'), ('y','f8') ]
    data = np.empty(len(x_list), dtype=dtype)
    data['x'] = x_list
    data['y'] = y_list
    cat_file = os.path.join('output','test_simple_reserve_cat.fits')
    fitsio.write(cat_file, data, clobber=True)

    psf_file = os.path.join('output','test_simple_reserve_psf.fits')
    config = {
        'input' : {
            'image_file_name' : image_file,
            'cat_file_name' : cat_file,
            'stamp_size' : 32,
        },
        'select' : {
            'reserve_frac' : 0.2,
        },
        'psf' : {
            'model' : { 'type' : 'Gaussian',
                        'fastfit': True,
                        'include_pixel': False},
            'interp' : { 'type' : 'Mean' },
        },
        'output' : { 'file_name' : psf_file },
    }

    piff.piffify(config, logger)
    psf = piff.read(psf_file)
    assert type(psf.model) is piff.Gaussian
    assert type(psf.interp) is piff.Mean
    print('chisq = ',psf.chisq)
    print('dof = ',psf.dof)
    nreserve = len([s for s in psf.stars if s.is_reserve])
    ntot = len(psf.stars)
    print('reserve = %s/%s'%(nreserve,ntot))
    assert nreserve == 2
    assert ntot == 10
    print('dof =? ',(32*32 - 3) * (ntot-nreserve))
    assert psf.dof == (32*32 - 3) * (ntot-nreserve)
    for star in psf.stars:
        # Fits should be good for both reserve and non-reserve stars
        np.testing.assert_almost_equal(star.fit.params, true_params, decimal=4)

    # Check the old API of putting reserve_frac in input
    config['input']['reserve_frac'] = 0.2
    del config['select']
    with CaptureLog() as cl:
        piff.piffify(config, cl.logger)
    assert "WARNING: Items ['reserve_frac'] should now be in the 'select' field" in cl.output
    psf = piff.read(psf_file)
    assert type(psf.model) is piff.Gaussian
    assert type(psf.interp) is piff.Mean
    nreserve = len([s for s in psf.stars if s.is_reserve])
    ntot = len(psf.stars)
    assert nreserve == 2
    assert ntot == 10


@timer
def test_model():
    """Test Model base class
    """
    # type is required
    config = { 'include_pixel': False }
    with np.testing.assert_raises(ValueError):
        model = piff.Model.process(config)

    # Can't do much with a base Model class
    model = piff.Model()
    np.testing.assert_raises(NotImplementedError, model.initialize, None)
    np.testing.assert_raises(NotImplementedError, model.fit, None)

    # Invalid to read a type that isn't a piff.Model type.
    # Mock this by pretending that Gaussian is the only subclass of Model.
    if sys.version_info < (3,): return  # mock only available on python 3
    from unittest import mock
    filename = os.path.join('input','D00240560_r_c01_r2362p01_piff.fits')
    with mock.patch('piff.util.get_all_subclasses', return_value=[piff.Gaussian]):
        with fitsio.FITS(filename,'r') as f:
            np.testing.assert_raises(ValueError, piff.Model.read, f, extname='psf_model')


@timer
def test_interp():
    """Test Interp base class
    """
    # type is required
    config = { 'order' : 0, }
    with np.testing.assert_raises(ValueError):
        interp = piff.Interp.process(config)

    # Can't do much with a base Interp class
    interp = piff.Interp()
    np.testing.assert_raises(NotImplementedError, interp.solve, None)
    np.testing.assert_raises(NotImplementedError, interp.interpolate, None)
    np.testing.assert_raises(NotImplementedError, interp.interpolateList, [None])

    filename1 = os.path.join('output','test_interp.fits')
    with fitsio.FITS(filename1,'rw',clobber=True) as f:
        np.testing.assert_raises(NotImplementedError, interp._finish_write, f, extname='interp')
    filename2 = os.path.join('input','D00240560_r_c01_r2362p01_piff.fits')
    with fitsio.FITS(filename2,'r') as f:
        np.testing.assert_raises(NotImplementedError, interp._finish_read, f, extname='interp')

    # Invalid to read a type that isn't a piff.Interp type.
    # Mock this by pretending that Mean is the only subclass of Interp.
    if sys.version_info < (3,): return  # mock only available on python 3
    from unittest import mock
    with mock.patch('piff.util.get_all_subclasses', return_value=[piff.Mean]):
        with fitsio.FITS(filename2,'r') as f:
            np.testing.assert_raises(ValueError, piff.Interp.read, f, extname='psf_interp')


@timer
def test_psf():
    """Test PSF base class
    """
    # Need a dummy star for the below calls
    image = galsim.Image(1,1,scale=1)
    stardata = piff.StarData(image, image.true_center)
    star = piff.Star(stardata, None)

    # Can't do much with a base PSF class
    psf = piff.PSF()
    np.testing.assert_raises(NotImplementedError, psf.parseKwargs, None)
    np.testing.assert_raises(NotImplementedError, psf.interpolateStar, star)
    np.testing.assert_raises(NotImplementedError, psf.interpolateStarList, [star])
    np.testing.assert_raises(NotImplementedError, psf.drawStar, star)
    np.testing.assert_raises(NotImplementedError, psf.drawStarList, [star])
    np.testing.assert_raises(NotImplementedError, psf._drawStar, star)
    np.testing.assert_raises(NotImplementedError, psf._getProfile, star)

    # Invalid to read a type that isn't a piff.PSF type.
    # Mock this by pretending that SingleChip is the only subclass of PSF.
    if sys.version_info < (3,): return  # mock only available on python 3
    from unittest import mock
    filename = os.path.join('input','D00240560_r_c01_r2362p01_piff.fits')
    with mock.patch('piff.util.get_all_subclasses', return_value=[piff.SingleChipPSF]):
        np.testing.assert_raises(ValueError, piff.PSF.read, filename)

@timer
def test_load_images():
    """Test the load_images function
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(log_file='output/test_load_image2.log')

    # Same setup as test_single_image, but without flags
    image = galsim.Image(2048, 2048, scale=0.26)
    x_list = [ 123.12, 345.98, 567.25, 1094.94, 924.15, 1532.74, 1743.11, 888.39, 1033.29, 1409.31 ]
    y_list = [ 345.43, 567.45, 1094.32, 924.29, 1532.92, 1743.83, 888.83, 1033.19, 1409.20, 123.11 ]
    sigma = 1.3
    g1 = 0.23
    g2 = -0.17
    psf = galsim.Gaussian(sigma=sigma).shear(g1=g1, g2=g2)
    for x,y in zip(x_list, y_list):
        bounds = galsim.BoundsI(int(x-31), int(x+32), int(y-31), int(y+32))
        psf.drawImage(image[bounds], center=galsim.PositionD(x,y), method='no_pixel')
    image.addNoise(galsim.GaussianNoise(rng=galsim.BaseDeviate(1234), sigma=1e-6))
    sky = 10.
    image += sky
    image_file = os.path.join('output','test_load_images_im.fits')
    image.write(image_file)

    dtype = [ ('x','f8'), ('y','f8') ]
    data = np.empty(len(x_list), dtype=dtype)
    data['x'] = x_list
    data['y'] = y_list
    cat_file = os.path.join('output','test_load_images_cat.fits')
    fitsio.write(cat_file, data, clobber=True)

    # Make star data
    config = { 'image_file_name' : image_file,
               'cat_file_name': cat_file,
               'sky': 10
             }
    orig_stars, wcs, pointing = piff.Input.process(config, logger)

    # Fit these with a simple Mean, Gaussian
    model = piff.Gaussian()
    interp = piff.Mean()
    psf = piff.SimplePSF(model, interp)
    psf.fit(orig_stars, wcs, pointing, logger=logger)
    psf_file = os.path.join('output','test_load_images_psf.fits')
    psf.write(psf_file, logger)

    # Read this file back in.  It has the star data, but the images are blank.
    psf2 = piff.read(psf_file, logger)
    assert len(psf2.stars) == 10
    for star in psf2.stars:
        np.testing.assert_array_equal(star.image.array, 0.)

    # First the old API:
    with CaptureLog() as cl:
        loaded_stars = piff.Star.load_images(psf2.stars, image_file, logger=cl.logger)
    assert "WARNING: The Star.load_images function is deprecated." in cl.output
    for star, orig in zip(loaded_stars, psf.stars):
        np.testing.assert_array_equal(star.image.array, orig.image.array)

    # Can optionally supply a different sky to subtract
    # (This is not allowed by the new API, but it probably doesn't make much sense.)
    loaded_stars = piff.Star.load_images(psf2.stars, image_file, sky=2*sky)
    for star, orig in zip(loaded_stars, psf.stars):
        np.testing.assert_array_equal(star.image.array, orig.image.array - sky)

    # Now the new API:
    psf2 = piff.read(psf_file, logger)  # Reset stars to not have images in them.
    loaded_stars = piff.InputFiles(config).load_images(psf2.stars)
    for star, orig in zip(loaded_stars, psf.stars):
        np.testing.assert_array_equal(star.image.array, orig.image.array)

@timer
def test_draw():
    """Test the various options of the PSF.draw command.
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(log_file='output/test_draw.log')

    # Use an existing Piff solution to match as closely as possible how users would actually
    # use this function.
    psf = piff.read('input/test_single_py27.piff', logger=logger)

    # Data that was used to make that file.
    wcs = galsim.TanWCS(
            galsim.AffineTransform(0.26, 0.05, -0.08, -0.24, galsim.PositionD(1024,1024)),
            galsim.CelestialCoord(-5 * galsim.arcmin, -25 * galsim.degrees)
            )
    data = fitsio.read('input/test_single_cat1.fits')
    field_center = galsim.CelestialCoord(0 * galsim.degrees, -25 * galsim.degrees)
    chipnum = 1

    for k in range(len(data)):
        x = data['x'][k]
        y = data['y'][k]
        e1 = data['e1'][k]
        e2 = data['e2'][k]
        s = data['s'][k]
        print('k,x,y = ',k,x,y)
        #print('  true s,e1,e2 = ',s,e1,e2)

        # First, the same test with this file that is in test_wcs.py:test_pickle()
        image_pos = galsim.PositionD(x,y)
        star = piff.Star.makeTarget(x=x, y=y, wcs=wcs, stamp_size=48, pointing=field_center,
                                    chipnum=chipnum)
        star = psf.drawStar(star)
        #print('  fitted s,e1,e2 = ',star.fit.params)
        np.testing.assert_almost_equal(star.fit.params, [s,e1,e2], decimal=6)

        # Now use the regular PSF.draw() command.  This version is equivalent to the above.
        # (It's not equal all the way to machine precision, but pretty close.)
        im1 = psf.draw(x, y, chipnum, stamp_size=48)
        np.testing.assert_allclose(im1.array, star.data.image.array, rtol=1.e-14, atol=1.e-14)

        # The wcs in the image is the wcs of the original image
        assert im1.wcs == psf.wcs[1]

        # The image is 48 x 48
        assert im1.array.shape == (48, 48)

        # The bounds are centered close to x,y.  Within 0.5 pixel.
        np.testing.assert_allclose(im1.bounds.true_center.x, x, atol=0.5)
        np.testing.assert_allclose(im1.bounds.true_center.y, y, atol=0.5)

        # This version draws the star centered at (x,y).  Check the hsm centroid.
        hsm = im1.FindAdaptiveMom()
        #print('hsm = ',hsm)
        np.testing.assert_allclose(hsm.moments_centroid.x, x, atol=0.01)
        np.testing.assert_allclose(hsm.moments_centroid.y, y, atol=0.01)

        # The total flux should be close to 1.
        np.testing.assert_allclose(im1.array.sum(), 1.0, rtol=1.e-3)

        # We can center the star at an arbitrary location on the image.
        # The default is equivalent to center=(x,y).  So check that this is equivalent.
        # Also, 48 is the default stamp size, so that can be omitted here.
        im2 = psf.draw(x, y, chipnum, center=(x,y))
        assert im2.bounds == im1.bounds
        np.testing.assert_allclose(im2.array, im1.array, rtol=1.e-14, atol=1.e-14)

        # Moving by an integer number of pixels should be very close to the same image
        # over a different slice of the array.
        im3 = psf.draw(x, y, chipnum, center=(x+1, y+3))
        assert im3.bounds == im1.bounds
        # (Remember -- numpy indexing is y,x!)
        # Also, the FFTs will be different in detail, so only match to 1.e-6.
        #print('im1 argmax = ',np.unravel_index(np.argmax(im1.array),im1.array.shape))
        #print('im3 argmax = ',np.unravel_index(np.argmax(im3.array),im3.array.shape))
        np.testing.assert_allclose(im3.array[3:,1:], im1.array[:-3,:-1], rtol=1.e-6, atol=1.e-6)
        hsm = im3.FindAdaptiveMom()
        np.testing.assert_allclose(hsm.moments_centroid.x, x+1, atol=0.01)
        np.testing.assert_allclose(hsm.moments_centroid.y, y+3, atol=0.01)

        # Can center at other locations, and the hsm centroids should come out centered pretty
        # close to that location.
        # (Of course the array will be different here, so can't test that.)
        im4 = psf.draw(x, y, chipnum, center=(x+1.3,y-0.8))
        assert im4.bounds == im1.bounds
        hsm = im4.FindAdaptiveMom()
        np.testing.assert_allclose(hsm.moments_centroid.x, x+1.3, atol=0.01)
        np.testing.assert_allclose(hsm.moments_centroid.y, y-0.8, atol=0.01)

        # Also allowed is center=True to place in the center of the image.
        im5 = psf.draw(x, y, chipnum, center=True)
        assert im5.bounds == im1.bounds
        assert im5.array.shape == (48,48)
        np.testing.assert_allclose(im5.bounds.true_center.x, x, atol=0.5)
        np.testing.assert_allclose(im5.bounds.true_center.y, y, atol=0.5)
        np.testing.assert_allclose(im5.array.sum(), 1., rtol=1.e-3)
        hsm = im5.FindAdaptiveMom()
        center = im5.true_center
        np.testing.assert_allclose(hsm.moments_centroid.x, center.x, atol=0.01)
        np.testing.assert_allclose(hsm.moments_centroid.y, center.y, atol=0.01)

        # Some invalid ways to try to do this. (Must be either True or a tuple.)
        np.testing.assert_raises(ValueError, psf.draw, x, y, chipnum, center='image')
        np.testing.assert_raises(ValueError, psf.draw, x, y, chipnum, center=im5.true_center)

        # If providing your own image with bounds far away from the star (say centered at 0),
        # then center=True works fine to draw in the center of that image.
        im6 = im5.copy()
        im6.setCenter(0,0)
        psf.draw(x, y, chipnum, center=True, image=im6)
        assert im6.bounds.center == galsim.PositionI(0,0)
        np.testing.assert_allclose(im6.array.sum(), 1., rtol=1.e-3)
        hsm = im6.FindAdaptiveMom()
        center = im6.true_center
        np.testing.assert_allclose(hsm.moments_centroid.x, center.x, atol=0.01)
        np.testing.assert_allclose(hsm.moments_centroid.y, center.y, atol=0.01)
        np.testing.assert_allclose(im6.array, im5.array, rtol=1.e-14, atol=1.e-14)

        # Check non-even stamp size.  Also, not unit flux while we're at it.
        im7 = psf.draw(x, y, chipnum, center=(x+1.3,y-0.8), stamp_size=43, flux=23.7)
        assert im7.array.shape == (43,43)
        np.testing.assert_allclose(im7.bounds.true_center.x, x, atol=0.5)
        np.testing.assert_allclose(im7.bounds.true_center.y, y, atol=0.5)
        np.testing.assert_allclose(im7.array.sum(), 23.7, rtol=1.e-3)
        hsm = im7.FindAdaptiveMom()
        np.testing.assert_allclose(hsm.moments_centroid.x, x+1.3, atol=0.01)
        np.testing.assert_allclose(hsm.moments_centroid.y, y-0.8, atol=0.01)

        # Can't do mixed even/odd shape with stamp_size, but it will respect a provided image.
        im8 = galsim.Image(43,44)
        im8.setCenter(x,y)  # It will respect the given bounds, so put it near the right place.
        psf.draw(x, y, chipnum, center=(x+1.3,y-0.8), image=im8, flux=23.7)
        assert im8.array.shape == (44,43)
        np.testing.assert_allclose(im8.array.sum(), 23.7, rtol=1.e-3)
        hsm = im8.FindAdaptiveMom()
        np.testing.assert_allclose(hsm.moments_centroid.x, x+1.3, atol=0.01)
        np.testing.assert_allclose(hsm.moments_centroid.y, y-0.8, atol=0.01)

        # The offset parameter can add an additional to whatever center is used.
        # Here center=None, so this is equivalent to im4 above.
        im9 = psf.draw(x, y, chipnum, offset=(1.3,-0.8))
        assert im9.bounds == im1.bounds
        hsm = im9.FindAdaptiveMom()
        np.testing.assert_allclose(im9.array, im4.array, rtol=1.e-14, atol=1.e-14)

        # With both, they are effectively added together.  Not sure if there would be a likely
        # use for this, but it's allowed.  (The above with default center is used in unit
        # tests a number of times, so that version at least is useful if only for us.
        # I'm hard pressed to imaging end users wanting to specify things this way though.)
        im10 = psf.draw(x, y, chipnum, center=(x+0.8, y-0.3), offset=(0.5,-0.5))
        assert im10.bounds == im1.bounds
        np.testing.assert_allclose(im10.array, im4.array, rtol=1.e-14, atol=1.e-14)

def test_depr_select():
    # A bunch of keys used to be allowed in input, but now belong in select.
    # Check that the old way still works, but gives a warning.

    # This is the input from test_stars in test_input.py
    dir = 'input'
    image_file = 'test_input_image_00.fits'
    cat_file = 'test_input_cat_00.fits'
    psf_file = os.path.join('output','test_depr_select.fits')

    config = {
        'input' : {
                'dir' : dir,
                'image_file_name' : image_file,
                'cat_file_name' : cat_file,
                'weight_hdu' : 1,
                'sky_col' : 'sky',
                'gain_col' : 'gain',
                'use_partial' : True,
                'nstars': 15,  # Just to make the test faster
             },
        'select': {
                'max_snr' : 200,
                'min_snr' : 20,
                'hsm_size_reject' : 20,
                'max_edge_frac': 0.25,
                'stamp_center_size': 10,
                'max_mask_pixels' : 20,
                'reserve_frac' : 0.1,
                'seed': 1234,
        },
        'psf': {
            'model' : {'type': 'Gaussian'},
            'interp' : {'type': 'Mean'},
        },
        'output' : { 'file_name' : psf_file },
    }

    logger = piff.config.setup_logger(log_file='output/test_depr_select.log')

    # This is the new API
    print('config = ',config)
    piff.piffify(config, logger)
    psf1 = piff.read(psf_file)
    im1 = psf1.draw(x=23, y=34, stamp_size=16)
    print('len1 = ',len(psf1.stars))

    # This is the old API
    config['input'].update(config['select'])
    del config['select']
    config = galsim.config.CleanConfig(config)
    print('config = ',config)
    with CaptureLog(level=1) as cl:
        piff.piffify(config, cl.logger)
    assert "WARNING: Items [" in cl.output
    assert "] should now be in the 'select' field of the config file." in cl.output
    psf2 = piff.read(psf_file)
    print('len2 = ',len(psf2.stars))
    assert len(psf1.stars) == len(psf2.stars)
    im2 = psf2.draw(x=23, y=34, stamp_size=16)
    np.testing.assert_allclose(im2.array, im1.array)

    # Also ok for some items to be in select, but erroneously put some in input.
    config['input']['min_snr'] = config['select'].pop('min_snr')
    config['input']['max_snr'] = config['select'].pop('max_snr')
    config = galsim.config.CleanConfig(config)
    print('config = ',config)
    with CaptureLog(level=1) as cl:
        piff.piffify(config, cl.logger)
    assert "WARNING: Items ['max_snr', 'min_snr'] should now be in the 'select' field of the config file." in cl.output
    psf3 = piff.read(psf_file)
    print('len3 = ',len(psf3.stars))
    assert len(psf1.stars) == len(psf3.stars)
    im3 = psf3.draw(x=23, y=34, stamp_size=16)
    np.testing.assert_allclose(im3.array, im1.array)


if __name__ == '__main__':
    test_Gaussian()
    test_Mean()
    test_single_image()
    test_invalid_config()
    test_reserve()
    test_model()
    test_interp()
    test_psf()
    test_load_images()
    test_draw()
