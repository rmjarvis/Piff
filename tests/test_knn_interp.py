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
import numpy
import piff
import os
import yaml
import fitsio

attr_interp = ['focal_x', 'focal_y']
attr_target = range(5)

def generate_data(n_samples=100):
    # generate as Norm(0, 1) for all parameters
    X = numpy.random.normal(0, 1, size=(n_samples, len(attr_interp)))
    y = numpy.random.normal(0, 1, size=(n_samples, len(attr_target)))

    star_list = []
    for Xi, yi in zip(X, y):
        # make some basic images, pass Xi as properties
        # Draw the PSF onto an image.  Let's go ahead and give it a non-trivial WCS.
        wcs = galsim.JacobianWCS(0.26, 0.05, -0.08, -0.29)
        image = galsim.Image(64,64, wcs=wcs)
        properties = {attr_interp[ith]: Xi[ith] for ith in xrange(len(attr_interp))}
        stardata = piff.StarData(image, image.trueCenter(), properties=properties)

        params = numpy.array([yi[ith] for ith in attr_target])
        starfit = piff.StarFit(params)
        star = piff.Star(stardata, starfit)
        star_list.append(star)

    return star_list

def test_init():
    # make sure we can init the interpolator
    knn = piff.kNNInterp()
    knn.build(attr_interp, attr_target)

def test_interp():
    # logger = piff.config.setup_logger(verbose=3, log_file='test_knn_interp.log')
    logger = None
    # make sure we can put in the data
    star_list = generate_data()
    knn = piff.kNNInterp(n_neighbors=1)
    knn.build(attr_interp, attr_target)
    knn.initialize(star_list, logger=logger)

    # make prediction on first 10 items of star_list
    star_list_predict = star_list[:10]
    star_list_predicted = knn.interpolateList(star_list_predict, logger=logger)
    # also on a single star
    star_predict = star_list_predict[0]
    star_predicted = knn.interpolate(star_predict)

    # predicted stars should find their exact partner here, so they have the same data
    numpy.testing.assert_array_equal(star_predicted.fit.params, star_predict.fit.params)
    for attr in attr_interp:
        numpy.testing.assert_equal(star_predicted.data[attr], star_predict.data[attr])

    # repeat for a star with its starfit removed
    star_predict = star_list_predict[0]
    star_predict.fit = None
    star_predicted = knn.interpolate(star_predict)

    # predicted stars should find their exact partner here, so they have the same data
    # removed the fit, so don't check that
    # numpy.testing.assert_array_equal(star_predicted.fit.params, star_predict.fit.params)
    for attr in attr_interp:
        numpy.testing.assert_equal(star_predicted.data[attr], star_predict.data[attr])

def test_attr_target():
    # make sure we can do the interpolation only over certain indices in params
    # make sure we can put in the data
    star_list = generate_data()
    attr_target_one = [1]
    attr_interp_one = ['focal_y']
    knn = piff.kNNInterp(n_neighbors=1)
    knn.build(attr_interp_one, attr_target_one)
    knn.initialize(star_list)

    # predict
    star_predict = star_list[0]
    star_predicted = knn.interpolate(star_predict)

    # predicted stars should find their exact partner here, so they have the same data
    # but here the fit params are not the same!!
    numpy.testing.assert_equal(star_predict.fit.params[attr_target_one[0]], star_predicted.fit.params[0])
    # we should still have the other interp parameter, however, so look at both!
    for attr in attr_interp:
        numpy.testing.assert_equal(star_predicted.data[attr], star_predict.data[attr])

    # repeat for a star with its starfit removed
    star_predict = star_list[0]
    star_predict.fit = None
    star_predicted = knn.interpolate(star_predict)

    # predicted stars should find their exact partner here, so they have the same data
    # we should still have the other interp parameter, however, so look at both!
    for attr in attr_interp:
        numpy.testing.assert_equal(star_predicted.data[attr], star_predict.data[attr])

def test_yaml():
    # using piffify executable
    pass

def test_disk():
    # make sure reading and writing of data works
    star_list = generate_data()
    knn = piff.kNNInterp()
    knn.build(attr_interp, attr_target)
    knn.initialize(star_list)
    knn_file = os.path.join('output','knn_interp.fits')
    with fitsio.FITS(knn_file,'rw',clobber=True) as f:
        knn.write(f, 'knn')
        knn2 = piff.kNNInterp.read(f, 'knn')
    numpy.testing.assert_array_equal(knn.locations, knn2.locations)
    numpy.testing.assert_array_equal(knn.targets, knn2.targets)
    numpy.testing.assert_array_equal(knn.attr_target, knn2.attr_target)
    numpy.testing.assert_array_equal(knn.attr_interp, knn2.attr_interp)

def test_decam():
    file_name = 'wavefront_test/Science-20121120s1-v20i2.fits'
    extname = 'Science-20121120s1-v20i2'
    knn = piff.DECamWavefront()
    knn.load_wavefront(file_name, extname)

    n_samples = 2000
    ccdnums = numpy.random.randint(1, 63, n_samples)

    star_list = []
    for ccdnum in ccdnums:
        # make some basic images, pass Xi as properties
        # Draw the PSF onto an image.  Let's go ahead and give it a non-trivial WCS.
        wcs = galsim.JacobianWCS(0.26, 0.05, -0.08, -0.29)
        image = galsim.Image(64,64, wcs=wcs)
        # set icen and jcen
        icen = numpy.random.randint(100, 2048)
        jcen = numpy.random.randint(100, 4096)
        image.setCenter(icen, jcen)
        image_pos = image.center()

        stardata = piff.StarData(image, image_pos, properties={'ccdnum': ccdnum})
        stardata = piff.pixel_to_focal(stardata)

        star = piff.Star(stardata, None)
        star_list.append(star)

    star_list_predicted = knn.interpolateList(star_list)

    # test misalignment
    misalignment = {'z04d': 10, 'z10x': 10, 'z09y': -10}
    knn.misalign_wavefront(misalignment)
    star_list_misaligned = knn.interpolateList(star_list)

    # test the prediction algorithm
    y_predicted = numpy.array([knn.getFitProperties(star) for star in star_list_predicted])
    y_misaligned = numpy.array([knn.getFitProperties(star) for star in star_list_misaligned])
    X = numpy.array([knn.getProperties(star) for star in star_list])

    # check the misalignments work
    numpy.testing.assert_array_almost_equal(y_predicted[:,0], y_misaligned[:,0] - misalignment['z04d'])
    numpy.testing.assert_array_almost_equal(y_predicted[:,5], y_misaligned[:,5] - misalignment['z09y'] * X[:,0])
    numpy.testing.assert_array_almost_equal(y_predicted[:,6], y_misaligned[:,6] - misalignment['z10x'] * X[:,1])


def test_decam_disk():
    file_name = 'wavefront_test/Science-20121120s1-v20i2.fits'
    extname = 'Science-20121120s1-v20i2'
    knn = piff.DECamWavefront()
    knn.load_wavefront(file_name, extname)

    misalignment = {'z04d': 10, 'z10x': 10, 'z09y': -10}
    knn.misalign_wavefront(misalignment)

    knn_file = os.path.join('output','decam_wavefront.fits')
    with fitsio.FITS(knn_file,'rw',clobber=True) as f:
        knn.write(f, 'decam_wavefront')
        knn2 = piff.DECamWavefront.read(f, 'decam_wavefront')
    numpy.testing.assert_array_equal(knn.locations, knn2.locations)
    numpy.testing.assert_array_equal(knn.targets, knn2.targets)
    numpy.testing.assert_array_equal(knn.attr_target, knn2.attr_target)
    numpy.testing.assert_array_equal(knn.attr_interp, knn2.attr_interp)
    numpy.testing.assert_array_equal(knn.misalignment, knn2.misalignment)


if __name__ == '__main__':
    print('test init')
    test_init()
    print('test interp')
    test_interp()
    print('test attr_target')
    test_attr_target()
    print('test disk')
    test_disk()
    print('test decam')
    test_decam()
    print('test decam disk')
    test_decam_disk()
