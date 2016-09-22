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

from test_helper import get_script_name

def make_kolmogorov_data(fwhm, g1, g2, u0, v0, flux, noise=0., du=1., fpu=0., fpv=0., nside=32,
                         nom_u0=0., nom_v0=0., rng=None):
    """Make a Star instance filled with a Kolmogorov profile

    :param fwhm:        The fwhm of the Kolmogorov.
    :param g1, g2:      Shear applied to profile.
    :param u0, v0:      The sub-pixel offset to apply.
    :param flux:        The flux of the star
    :param noise:       RMS Gaussian noise to be added to each pixel [default: 0]
    :param du:          pixel size in "wcs" units [default: 1.]
    :param fpu,fpv:     position of this cutout in some larger focal plane [default: 0,0]
    :param nside:       The size of the array [default: 32]
    :param nom_u0, nom_v0:  The nominal u0,v0 in the StarData [default: 0,0]
    :param rng:         If adding noise, the galsim deviate to use for the random numbers
                        [default: None]
    """
    k = galsim.Kolmogorov(fwhm=fwhm, flux=flux).shear(g1=g1, g2=g2).shift(u0,v0)
    if noise == 0.:
        var = 0.1
    else:
        var = noise
    star = piff.Star.makeTarget(x=nside/2+nom_u0/du, y=nside/2+nom_v0/du,
                                u=fpu, v=fpv, scale=du, stamp_size=nside)
    star.image.setOrigin(0,0)
    k.drawImage(star.image, method='no_pixel',
                offset=galsim.PositionD(nom_u0/du,nom_v0/du), use_true_center=False)
    star.data.weight = star.image.copy()
    star.weight.fill(1./var/var)
    if noise != 0:
        gn = galsim.GaussianNoise(sigma=noise, rng=rng)
        star.image.addNoise(gn)
    return star


def test_gp_poly():
    """ Test that Gaussian Process interpolation works reasonably well for densely packed Kolmogorov
    star field with params that vary as polynomials.
    """
    bd = galsim.BaseDeviate(5647382910)
    ud = galsim.UniformDeviate(bd)
    nstars = 50
    upositions = [ud() for i in xrange(nstars)]
    vpositions = [ud() for i in xrange(nstars)]
    flux = 100.0

    g1_fn = lambda u,v: 0.1*u - 0.1*v
    g2_fn = lambda u,v: -0.1*v + 0.1*u*v
    fwhm_fn = lambda u,v: 1.0-0.05*u+0.05*v
    u0_fn = lambda u,v: 0.5*u
    v0_fn = lambda u,v: 0.3*u+0.3*v

    mod = piff.Kolmogorov()  # Center is marginalized, not part of PSF params.

    stars = []
    for u, v in zip(upositions, vpositions):
        s = make_kolmogorov_data(fwhm_fn(u,v), g1_fn(u,v), g2_fn(u,v), u0_fn(u,v), v0_fn(u,v), flux,
                                 noise=0.1, du=0.5, fpu=u, fpv=v, rng=bd)
        s = mod.initialize(s)
        stars.append(s)

    interp = piff.GPInterp(thetaL=1e0, theta0=1e1, thetaU=1e6)
    interp.initialize(stars)

    # Get noiseless copy of the PSF at the center of the FOV
    u,v = 0.5, 0.5
    s0 = make_kolmogorov_data(fwhm_fn(u,v), g1_fn(u,v), g2_fn(u,v), u0_fn(u,v), v0_fn(u,v), flux,
                              du=0.5)
    s0 = mod.initialize(s0)

    oldchisq = 0.
    for iteration in range(10):
        # Refit PSFs star by star:
        for i, s in enumerate(stars):
            stars[i] = mod.fit(s)
        interp.solve(stars)
        chisq = 0.
        dof = 0
        for i, s in enumerate(stars):
            s = interp.interpolate(s)
            s = mod.reflux(s)
            chisq += s.fit.chisq
            dof += s.fit.dof
            stars[i] = s
        print("iteration: {}  chisq: {}  dof: {}".format(iteration, chisq, dof))
        if oldchisq>0 and np.abs(oldchisq-chisq) < dof/10.0:
            break
        else:
            oldchisq = chisq

    s1 = interp.interpolate(s0)
    s1 = mod.reflux(s1)
    print('Flux, ctr, chisq after interpolation: ',s1.fit.flux, s1.fit.center, s1.fit.chisq)
    # Less than 2 dp of accuracy here!
    np.testing.assert_almost_equal(s1.fit.flux/flux, 1.0, decimal=1)

    s1 = mod.draw(s1)
    print('max image abs diff = ',np.max(np.abs(s1.image.array-s0.image.array)))
    print('max image abs value = ',np.max(np.abs(s0.image.array)))
    peak = np.max(np.abs(s0.image.array))
    np.testing.assert_almost_equal(s1.image.array/peak, s0.image.array/peak, decimal=2)


if __name__ == '__main__':
    test_gp_poly()


# def generate_data(n_samples=100):
#     # generate as Norm(0, 1) for all parameters
#     X = np.random.normal(0, 1, size=(n_samples, len(attr_interp)))
#     y = np.random.normal(0, 1, size=(n_samples, len(attr_target)))
#
#     star_list = []
#     for Xi, yi in zip(X, y):
#         # make some basic images, pass Xi as properties
#         # Draw the PSF onto an image.  Let's go ahead and give it a non-trivial WCS.
#         wcs = galsim.JacobianWCS(0.26, 0.05, -0.08, -0.29)
#         image = galsim.Image(64,64, wcs=wcs)
#         properties = {attr_interp[ith]: Xi[ith] for ith in xrange(len(attr_interp))}
#         stardata = piff.StarData(image, image.trueCenter(), properties=properties)
#
#         params = np.array([yi[ith] for ith in attr_target])
#         starfit = piff.StarFit(params)
#         star = piff.Star(stardata, starfit)
#         star_list.append(star)
#
#     return star_list

# def test_init():
#     # make sure we can init the interpolator
#     knn = piff.kNNInterp(attr_interp, attr_target)
#
# def test_interp():
#     # logger = piff.config.setup_logger(verbose=3, log_file='test_knn_interp.log')
#     logger = None
#     # make sure we can put in the data
#     star_list = generate_data()
#     knn = piff.kNNInterp(attr_interp, attr_target, n_neighbors=1)
#     knn.initialize(star_list, logger=logger)
#
#     # make prediction on first 10 items of star_list
#     star_list_predict = star_list[:10]
#     star_list_predicted = knn.interpolateList(star_list_predict, logger=logger)
#     # also on a single star
#     star_predict = star_list_predict[0]
#     star_predicted = knn.interpolate(star_predict)
#
#     # predicted stars should find their exact partner here, so they have the same data
#     np.testing.assert_array_equal(star_predicted.fit.params, star_predict.fit.params)
#     for attr in attr_interp:
#         np.testing.assert_equal(star_predicted.data[attr], star_predict.data[attr])
#
#     # repeat for a star with its starfit removed
#     star_predict = star_list_predict[0]
#     star_predict.fit = None
#     star_predicted = knn.interpolate(star_predict)
#
#     # predicted stars should find their exact partner here, so they have the same data
#     # removed the fit, so don't check that
#     # np.testing.assert_array_equal(star_predicted.fit.params, star_predict.fit.params)
#     for attr in attr_interp:
#         np.testing.assert_equal(star_predicted.data[attr], star_predict.data[attr])
#
# def test_attr_target():
#     # make sure we can do the interpolation only over certain indices in params
#     # make sure we can put in the data
#     star_list = generate_data()
#     attr_target_one = [1]
#     attr_interp_one = ['focal_y']
#     knn = piff.kNNInterp(attr_interp_one, attr_target_one, n_neighbors=1)
#     knn.initialize(star_list)
#
#     # predict
#     star_predict = star_list[0]
#     star_predicted = knn.interpolate(star_predict)
#
#     # predicted stars should find their exact partner here, so they have the same data
#     # but here the fit params are not the same!!
#     np.testing.assert_equal(star_predict.fit.params[attr_target_one[0]], star_predicted.fit.params[0])
#     # we should still have the other interp parameter, however, so look at both!
#     for attr in attr_interp:
#         np.testing.assert_equal(star_predicted.data[attr], star_predict.data[attr])
#
#     # repeat for a star with its starfit removed
#     star_predict = star_list[0]
#     star_predict.fit = None
#     star_predicted = knn.interpolate(star_predict)
#
#     # predicted stars should find their exact partner here, so they have the same data
#     # we should still have the other interp parameter, however, so look at both!
#     for attr in attr_interp:
#         np.testing.assert_equal(star_predicted.data[attr], star_predict.data[attr])
#
# def test_yaml():
#     # Take DES test image, and test doing a psf run with kNN interpolator
#     # Now test running it via the config parser
#     psf_file = os.path.join('output','knn_psf.fits')
#     config = {
#         'input' : {
#             'images' : 'y1_test/DECam_00241238_01.fits.fz',
#             'cats' : 'y1_test/DECam_00241238_01_psfcat_tb_maxmag_17.0_magcut_3.0_findstars.fits',
#             # What hdu is everything in?
#             'image_hdu': 1,
#             'badpix_hdu': 2,
#             'weight_hdu': 3,
#             'cat_hdu': 2,
#
#             # What columns in the catalog have things we need?
#             'x_col': 'XWIN_IMAGE',
#             'y_col': 'YWIN_IMAGE',
#             'ra': 'TELRA',
#             'dec': 'TELDEC',
#             'gain': 'GAINA',
#             'sky_col': 'BACKGROUND',
#
#             # How large should the postage stamp cutouts of the stars be?
#             'stamp_size': 31,
#         },
#         'psf' : {
#             'model' : { 'type': 'Gaussian' },
#             'interp' : { 'type': 'kNNInterp',
#                          'attr_interp': ['u', 'v'],
#                          'attr_target': [0, 1, 2],
#                          'n_neighbors': 117,}
#         },
#         'output' : { 'file_name' : psf_file },
#     }
#
#     # using piffify executable
#     config['verbose'] = 0
#     with open('knn.yaml','w') as f:
#         f.write(yaml.dump(config, default_flow_style=False))
#     piffify_exe = get_script_name('piffify')
#     p = subprocess.Popen( [piffify_exe, 'knn.yaml'] )
#     p.communicate()
#     psf = piff.read(psf_file)
#
#     # by taking every star in ccd as 'nearest' neighbor, we should get same value
#     # for each star's interpolation
#     np.testing.assert_allclose(psf.drawStar(psf.stars[0]).fit.params,
#                                psf.drawStar(psf.stars[-1]).fit.params)
#
# def test_disk():
#     # make sure reading and writing of data works
#     star_list = generate_data()
#     knn = piff.kNNInterp(attr_interp, attr_target, n_neighbors=2)
#     knn.initialize(star_list)
#     knn_file = os.path.join('output','knn_interp.fits')
#     with fitsio.FITS(knn_file,'rw',clobber=True) as f:
#         knn.write(f, 'knn')
#         knn2 = piff.kNNInterp.read(f, 'knn')
#     np.testing.assert_array_equal(knn.locations, knn2.locations)
#     np.testing.assert_array_equal(knn.targets, knn2.targets)
#     np.testing.assert_array_equal(knn.kwargs['attr_target'], knn2.kwargs['attr_target'])
#     np.testing.assert_array_equal(knn.kwargs['attr_interp'], knn2.kwargs['attr_interp'])
#     np.testing.assert_equal(knn.knr_kwargs['n_neighbors'], knn2.knr_kwargs['n_neighbors'])
#     np.testing.assert_equal(knn.knr_kwargs['algorithm'], knn2.knr_kwargs['algorithm'])
#
# def test_decam_wavefront():
#     file_name = 'wavefront_test/Science-20121120s1-v20i2.fits'
#     extname = 'Science-20121120s1-v20i2'
#     knn = piff.des.DECamWavefront(file_name, extname)
#
#     n_samples = 2000
#     ccdnums = np.random.randint(1, 63, n_samples)
#
#     star_list = []
#     for ccdnum in ccdnums:
#         # make some basic images, pass Xi as properties
#         # Draw the PSF onto an image.  Let's go ahead and give it a non-trivial WCS.
#         wcs = galsim.JacobianWCS(0.26, 0.05, -0.08, -0.29)
#         image = galsim.Image(64,64, wcs=wcs)
#         # set icen and jcen
#         icen = np.random.randint(100, 2048)
#         jcen = np.random.randint(100, 4096)
#         image.setCenter(icen, jcen)
#         image_pos = image.center()
#
#         stardata = piff.StarData(image, image_pos, properties={'ccdnum': ccdnum})
#
#         star = piff.Star(stardata, None)
#         star_list.append(star)
#
#     # get the focal positions
#     star_list = piff.des.DECamInfo().pixel_to_focalList(star_list)
#
#     star_list_predicted = knn.interpolateList(star_list)
#
#     # test misalignment
#     misalignment = {'z04d': 10, 'z10x': 10, 'z09y': -10}
#     knn.misalign_wavefront(misalignment)
#     star_list_misaligned = knn.interpolateList(star_list)
#
#     # test the prediction algorithm
#     y_predicted = np.array([knn.getFitProperties(s) for s in star_list_predicted])
#     y_misaligned = np.array([knn.getFitProperties(s) for s in star_list_misaligned])
#     X = np.array([knn.getProperties(s) for s in star_list])
#
#     # check the misalignments work
#     np.testing.assert_array_almost_equal(y_predicted[:,0], y_misaligned[:,0] - misalignment['z04d'])
#     np.testing.assert_array_almost_equal(y_predicted[:,5], y_misaligned[:,5] - misalignment['z09y'] * X[:,0])
#     np.testing.assert_array_almost_equal(y_predicted[:,6], y_misaligned[:,6] - misalignment['z10x'] * X[:,1])
#
#
# def test_decam_disk():
#     file_name = 'wavefront_test/Science-20121120s1-v20i2.fits'
#     extname = 'Science-20121120s1-v20i2'
#     knn = piff.des.DECamWavefront(file_name, extname, n_neighbors=30)
#
#     misalignment = {'z04d': 10, 'z10x': 10, 'z09y': -10}
#     knn.misalign_wavefront(misalignment)
#
#     knn_file = os.path.join('output','decam_wavefront.fits')
#     with fitsio.FITS(knn_file,'rw',clobber=True) as f:
#         knn.write(f, 'decam_wavefront')
#         knn2 = piff.des.DECamWavefront.read(f, 'decam_wavefront')
#     np.testing.assert_array_equal(knn.locations, knn2.locations)
#     np.testing.assert_array_equal(knn.targets, knn2.targets)
#     np.testing.assert_array_equal(knn.attr_target, knn2.attr_target)
#     np.testing.assert_array_equal(knn.attr_interp, knn2.attr_interp)
#     np.testing.assert_array_equal(knn.misalignment, knn2.misalignment)
#     assert knn.knr_kwargs['n_neighbors'] == knn2.knr_kwargs['n_neighbors'], 'n_neighbors not equal'
#     assert knn.knr_kwargs['algorithm'] == knn2.knr_kwargs['algorithm'], 'algorithm not equal'
#
# def test_decaminfo():
#     # test switching between focal and pixel coordinates
#     n_samples = 500000
#     chipnums = np.random.randint(1, 63, n_samples)
#     icen = np.random.randint(1, 2048, n_samples)
#     jcen = np.random.randint(1, 4096, n_samples)
#
#     decaminfo = piff.des.DECamInfo()
#     xPos, yPos = decaminfo.getPosition(chipnums, icen, jcen)
#     chipnums_ret, icen_ret, jcen_ret = decaminfo.getPixel(xPos, yPos)
#     xPos_ret, yPos_ret = decaminfo.getPosition(chipnums_ret, icen_ret, jcen_ret)
#
#     np.testing.assert_allclose(chipnums, chipnums_ret)
#     np.testing.assert_allclose(xPos, xPos_ret)
#     np.testing.assert_allclose(yPos, yPos_ret)
#     np.testing.assert_allclose(icen, icen_ret)
#     np.testing.assert_allclose(jcen, jcen_ret)
#
# if __name__ == '__main__':
#     print('test init')
#     test_init()
#     print('test interp')
#     test_interp()
#     print('test attr_target')
#     test_attr_target()
#     print('test disk')
#     test_disk()
#     print('test decaminfo')
#     test_decaminfo()
#     print('test yaml')
#     test_yaml()
#     print('test decam wavefront')
#     test_decam_wavefront()
#     print('test decam disk')
#     test_decam_disk()
