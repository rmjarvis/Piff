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
import fitsio

from piff_test_helper import timer

@timer
def test_chisq():
    """Test the Chisq outlier class
    """
    if __name__ == '__main__':
        logger = piff.config.setup_logger(verbose=2)
    else:
        logger = piff.config.setup_logger(log_file='output/test_single_image.log')

    # Make the image
    image = galsim.Image(512, 512, scale=0.26)

    nstars = 1000  # enough that there could be some overlaps.  Also, some over the edge.
    rng = np.random.RandomState(1234)
    x = rng.random_sample(nstars) * 512
    y = rng.random_sample(nstars) * 512
    sigma = np.ones_like(x) * 0.4   # Most have this sigma
    g1 = np.ones_like(x) * 0.023    # Most are pretty round.
    g2 = np.ones_like(x) * 0.012
    flux = np.exp(rng.normal(size=nstars))
    print('flux range = ',np.min(flux),np.max(flux),np.median(flux),np.mean(flux))

    # Make a few intentionally wrong.
    g1[35] = 0.29
    g1[188] = -0.15
    sigma[239] = 0.2
    g2[347] = -0.15
    sigma[551] = 1.3
    g2[809] = 0.05
    g1[922] = -0.03

    # Draw a Gaussian PSF at each location on the image.
    for i in range(nstars):
        psf = galsim.Gaussian(sigma=sigma[i]).shear(g1=g1[i], g2=g2[i])
        stamp = psf.drawImage(scale=0.26, center=galsim.PositionD(x[i],y[i]))
        b = stamp.bounds & image.bounds
        image[b] += stamp[b]

    noise = 0.02
    image.addNoise(galsim.GaussianNoise(rng=galsim.BaseDeviate(1234), sigma=noise))

    image_file = os.path.join('output','test_chisq_im.fits')
    image.write(image_file)

    # Write out the catalog to a file
    dtype = [ ('x','f8'), ('y','f8') ]
    data = np.empty(len(x), dtype=dtype)
    data['x'] = x
    data['y'] = y
    cat_file = os.path.join('output','test_chisq_cat.fits')
    fitsio.write(cat_file, data, clobber=True)

    # Read the catalog in as stars.
    config = { 'image_file_name' : image_file,
               'cat_file_name': cat_file,
               'noise': noise**2,  # Variance here is sigma^2
               'stamp_size' : 15,
               'use_partial' : True,
             }
    input = piff.InputFiles(config, logger=logger)
    stars = input.makeStars()

    # Skip the solve step.  Just give it the right answer and see what it finds for outliers
    model = piff.Gaussian()
    interp = piff.Mean()
    interp.mean = np.array([0.4, 0.023, 0.012])
    psf = piff.SimplePSF(model, interp)
    stars = psf.interpolateStarList(stars)
    stars = [ psf.model.reflux(s,logger=logger) for s in stars ]

    outliers1 = piff.ChisqOutliers(nsigma=5)
    stars1, nremoved1 = outliers1.removeOutliers(stars,logger=logger)
    print('nremoved1 = ',nremoved1)
    assert len(stars1) == len(stars) - nremoved1

    # This is what nsigma=5 means in terms of probability
    outliers2 = piff.ChisqOutliers(prob=5.733e-7)
    stars2, nremoved2 = outliers2.removeOutliers(stars,logger=logger)
    print('nremoved2 = ',nremoved2)
    assert len(stars2) == len(stars) - nremoved2
    assert nremoved1 == nremoved2

    # The following is nearly equivalent for this particular data set.
    # For dof=222 (what most of these have, this probability converts to
    # thresh = 455.40143379
    # or ndof = 2.0513578
    # But note that when using the above prop or nsigma, the code uses a tailored threshold
    # different for each star's particular dof, which varies (since some are off the edge).
    outliers3 = piff.ChisqOutliers(thresh=455.401)
    stars3, nremoved3 = outliers3.removeOutliers(stars,logger=logger)
    print('nremoved3 = ',nremoved3)
    assert len(stars3) == len(stars) - nremoved3

    outliers4 = piff.ChisqOutliers(ndof=2.05136)
    stars4, nremoved4 = outliers4.removeOutliers(stars,logger=logger)
    print('nremoved4 = ',nremoved4)
    assert len(stars4) == len(stars) - nremoved4
    assert nremoved3 == nremoved4

    # Regression tests.  If these change, make sure we understand why.
    assert nremoved1 == nremoved2 == 58
    assert nremoved3 == nremoved4 == 16  # Much less, since edge objects aren't being removed
                                         # nearly as often as when they have a custom thresh.

    # Can't provide multiple thresh specifications
    np.testing.assert_raises(TypeError, piff.ChisqOutliers, nsigma=5, prob=1.e-3)
    np.testing.assert_raises(TypeError, piff.ChisqOutliers, nsigma=5, thresh=100)
    np.testing.assert_raises(TypeError, piff.ChisqOutliers, nsigma=5, ndof=3)
    np.testing.assert_raises(TypeError, piff.ChisqOutliers, prob=1.e-3, thresh=100)
    np.testing.assert_raises(TypeError, piff.ChisqOutliers, prob=1.e-3, ndof=3)
    np.testing.assert_raises(TypeError, piff.ChisqOutliers, thresh=100, ndof=3)

    # Need to specifiy it somehow.
    np.testing.assert_raises(TypeError, piff.ChisqOutliers)

@timer
def test_base():
    """Test Outliers base class
    """
    # type is required
    config = { 'nsigma' : 4, }
    with np.testing.assert_raises(ValueError):
        out = piff.Outliers.process(config)

    # Invalid to read a type that isn't a piff.Outliers type.
    # Mock this by pretending that MADOutliers is the only subclass of Outliers.
    if sys.version_info < (3,): return  # mock only available on python 3
    from unittest import mock
    filename = os.path.join('input','D00240560_r_c01_r2362p01_piff.fits')
    with mock.patch('piff.util.get_all_subclasses', return_value=[piff.outliers.MADOutliers]):
        with fitsio.FITS(filename,'r') as f:
            np.testing.assert_raises(ValueError, piff.Outliers.read, f, extname='psf_outliers')


if __name__ == '__main__':
    test_chisq()
    test_base()
