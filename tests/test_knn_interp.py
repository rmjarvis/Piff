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
import subprocess
import fitsio

from piff_test_helper import timer

keys = ['focal_x', 'focal_y']
ntarget = 5


def generate_data(n_samples=100):
    # generate as Norm(0, 1) for all parameters
    np_rng = np.random.RandomState(1234)
    X = np_rng.normal(0, 1, size=(n_samples, len(keys)))
    y = np_rng.normal(0, 1, size=(n_samples, ntarget))

    star_list = []
    for Xi, yi in zip(X, y):
        wcs = galsim.JacobianWCS(0.26, 0.05, -0.08, -0.29)
        image = galsim.Image(64,64, wcs=wcs)
        properties = {k:v for k,v in zip(keys, Xi)}
        stardata = piff.StarData(image, image.true_center, properties=properties)

        # params = np.array([yi[ith] for ith in attr_target])
        params = yi
        starfit = piff.StarFit(params)
        star = piff.Star(stardata, starfit)
        star_list.append(star)

    return star_list


@timer
def test_init():
    # make sure we can init the interpolator
    knn = piff.kNNInterp(keys)
    assert knn.property_names == keys

@timer
def test_interp():
    # logger = piff.config.setup_logger(verbose=3, log_file='test_knn_interp.log')
    logger = None
    # make sure we can put in the data
    star_list = generate_data()
    knn = piff.kNNInterp(keys, n_neighbors=1)
    knn.initialize(star_list, logger=logger)
    knn.solve(star_list, logger=logger)

    # make prediction on first 10 items of star_list
    star_list_predict = star_list[:10]
    star_list_predicted = knn.interpolateList(star_list_predict, logger=logger)
    # also on a single star
    star_predict = star_list_predict[0]
    star_predicted = knn.interpolate(star_predict)

    # predicted stars should find their exact partner here, so they have the same data
    np.testing.assert_array_equal(star_predicted.fit.params, star_predict.fit.params)
    for attr in keys:
        np.testing.assert_equal(star_predicted.data[attr], star_predict.data[attr])

    # repeat for a star with its starfit removed
    star_predict = star_list_predict[0]
    star_predict.fit = None
    star_predicted = knn.interpolate(star_predict)

    # predicted stars should find their exact partner here, so they have the same data
    # removed the fit, so don't check that
    # np.testing.assert_array_equal(star_predicted.fit.params, star_predict.fit.params)
    for attr in keys:
        np.testing.assert_equal(star_predicted.data[attr], star_predict.data[attr])


@timer
def test_config():
    # Take DES test image, and test doing a psf run with kNN interpolator
    # Now test running it via the config parser
    psf_file = os.path.join('output','knn_psf.fits')
    config = {
        'input' : {
            'image_file_name' : 'input/DECam_00241238_01.fits.fz',
            'cat_file_name' : 'input/DECam_00241238_01_psfcat_tb_maxmag_17.0_magcut_3.0_findstars.fits',
            # What hdu is everything in?
            'image_hdu': 1,
            'badpix_hdu': 2,
            'weight_hdu': 3,
            'cat_hdu': 2,

            # What columns in the catalog have things we need?
            'x_col': 'XWIN_IMAGE',
            'y_col': 'YWIN_IMAGE',
            'ra': 'TELRA',
            'dec': 'TELDEC',
            'gain': 'GAINA',
            'sky_col': 'BACKGROUND',

            # How large should the postage stamp cutouts of the stars be?
            'stamp_size': 31,
        },
        'psf' : {
            'model' : { 'type': 'GSObjectModel',
                        'fastfit': True,
                        'gsobj': 'galsim.Gaussian(sigma=1.0)' },
            'interp' : { 'type': 'kNNInterp',
                         'keys': ['u', 'v'],
                         'n_neighbors': 115,}
        },
        'output' : { 'file_name' : psf_file },
    }
    if __name__ != '__main__':
        config['verbose'] = 0
        config['input']['nstars'] = 20
        config['psf']['interp']['n_neighbors'] = 19
        test_factor = 0.04
    else:
        test_factor = 0.01

    psf = piff.process(config)

    # by using n_neighbors = 115, when there are only 117 stars in the catalog, we should expect
    # that the standard deviation of the interpolated parameters should be small, since almost the
    # same set of stars are being averaged in every case.
    nstars = len(psf.stars)
    np.testing.assert_array_less(
            np.std([s.fit.params for s in psf.stars], axis=0),
            test_factor*np.mean([s.fit.params for s in psf.stars], axis=0),
            err_msg="Interpolated parameters show too much variation.")

@timer
def test_disk():
    # make sure reading and writing of data works
    star_list = generate_data()
    knn = piff.kNNInterp(keys, n_neighbors=2)
    knn.initialize(star_list)
    knn.solve(star_list)
    knn_file = os.path.join('output','knn_interp.fits')
    with fitsio.FITS(knn_file,'rw',clobber=True) as f:
        knn.write(f, 'knn')
        knn2 = piff.kNNInterp.read(f, 'knn')
    np.testing.assert_array_equal(knn.locations, knn2.locations)
    np.testing.assert_array_equal(knn.targets, knn2.targets)
    np.testing.assert_array_equal(knn.kwargs['keys'], knn2.kwargs['keys'])
    np.testing.assert_equal(knn.knr_kwargs['n_neighbors'], knn2.knr_kwargs['n_neighbors'])
    np.testing.assert_equal(knn.knr_kwargs['algorithm'], knn2.knr_kwargs['algorithm'])


@timer
def test_decam_wavefront():
    file_name = 'input/Science-20121120s1-v20i2.fits'
    extname = 'Science-20121120s1-v20i2'

    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(log_file='output/test_decam.log')
    knn = piff.des.DECamWavefront(file_name, extname, logger=logger)

    n_samples = 2000
    np_rng = np.random.RandomState(1234)
    ccdnums = np_rng.randint(1, 63, n_samples)

    star_list = []
    for ccdnum in ccdnums:
        # make some basic images, pass Xi as properties
        # Draw the PSF onto an image.  Let's go ahead and give it a non-trivial WCS.
        wcs = galsim.JacobianWCS(0.26, 0.05, -0.08, -0.29)
        image = galsim.Image(64,64, wcs=wcs)
        # set icen and jcen
        icen = np_rng.randint(100, 2048)
        jcen = np_rng.randint(100, 4096)
        image.setCenter(icen, jcen)
        image_pos = image.center

        stardata = piff.StarData(image, image_pos, properties={'chipnum': ccdnum})

        star = piff.Star(stardata, None)
        star_list.append(star)

    # get the focal positions
    star_list = piff.des.DECamInfo().pixel_to_focalList(star_list)

    star_list_predicted = knn.interpolateList(star_list)

    # test misalignment
    misalignment = {'z04d': 10, 'z10x': 10, 'z09y': -10}
    knn.misalign_wavefront(misalignment)
    star_list_misaligned = knn.interpolateList(star_list)

    # test the prediction algorithm
    y_predicted = np.array([s.fit.params for s in star_list_predicted])
    y_misaligned = np.array([s.fit.params for s in star_list_misaligned])
    X = np.array([knn.getProperties(s) for s in star_list])

    # check the misalignments work
    np.testing.assert_array_almost_equal(y_predicted[:,0], y_misaligned[:,0] - misalignment['z04d'])
    np.testing.assert_array_almost_equal(y_predicted[:,5], y_misaligned[:,5] - misalignment['z09y'] * X[:,0])
    np.testing.assert_array_almost_equal(y_predicted[:,6], y_misaligned[:,6] - misalignment['z10x'] * X[:,1])

    # Check shape of misalignment if array
    np.testing.assert_raises(ValueError, knn.misalign_wavefront, knn.misalignment[:,:2])
    np.testing.assert_raises(ValueError, knn.misalign_wavefront, knn.misalignment[:-1,:])

    # empty dict is equivalent to no misalignment
    knn.misalign_wavefront({})
    np.testing.assert_equal(knn.misalignment, 0.)


@timer
def test_decam_disk():
    file_name = 'input/Science-20121120s1-v20i2.fits'
    extname = 'Science-20121120s1-v20i2'
    knn = piff.des.DECamWavefront(file_name, extname, n_neighbors=30)

    misalignment = {'z04d': 10, 'z10x': 10, 'z09y': -10}
    knn.misalign_wavefront(misalignment)

    knn_file = os.path.join('output','decam_wavefront.fits')
    with fitsio.FITS(knn_file,'rw',clobber=True) as f:
        knn.write(f, 'decam_wavefront')
        knn2 = piff.des.DECamWavefront.read(f, 'decam_wavefront')
    np.testing.assert_array_equal(knn.locations, knn2.locations)
    np.testing.assert_array_equal(knn.targets, knn2.targets)
    np.testing.assert_array_equal(knn.keys, knn2.keys)
    np.testing.assert_array_equal(knn.misalignment, knn2.misalignment)
    assert knn.knr_kwargs['n_neighbors'] == knn2.knr_kwargs['n_neighbors'], 'n_neighbors not equal'
    assert knn.knr_kwargs['algorithm'] == knn2.knr_kwargs['algorithm'], 'algorithm not equal'


@timer
def test_decaminfo():
    # test switching between focal and pixel coordinates
    n_samples = 500000
    np_rng = np.random.RandomState(1234)
    chipnums = np_rng.randint(1, 63, n_samples)
    icen = np_rng.randint(1, 2048, n_samples)
    jcen = np_rng.randint(1, 4096, n_samples)

    decaminfo = piff.des.DECamInfo()
    xPos, yPos = decaminfo.getPosition(chipnums, icen, jcen)
    chipnums_ret, icen_ret, jcen_ret = decaminfo.getPixel(xPos, yPos)
    xPos_ret, yPos_ret = decaminfo.getPosition(chipnums_ret, icen_ret, jcen_ret)

    np.testing.assert_allclose(chipnums, chipnums_ret)
    np.testing.assert_allclose(xPos, xPos_ret)
    np.testing.assert_allclose(yPos, yPos_ret)
    np.testing.assert_allclose(icen, icen_ret)
    np.testing.assert_allclose(jcen, jcen_ret)

if __name__ == '__main__':
    test_init()
    test_interp()
    test_config()
    test_disk()
    test_decam_wavefront()
    test_decam_disk()
    test_decaminfo()
