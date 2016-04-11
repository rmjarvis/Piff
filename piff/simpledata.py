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
.. module:: stardata
"""

from __future__ import print_function
import numpy as np

class SimpleData(object):
    """A very simple, galsim-free implementation of the StarData class, for basic
    tests of the PixelModel class.
    """
    def __init__(self, data, noise, u0, v0, du=1., values_are_sb = False):
        """

        :param data:  2d numpy array holding the pixel data
        :param noise: RMS Gaussian noise to be assumed on each pixel
        :param u0, v0: nominal center of the star relative to lower-left pixel
        :param du:    pixel size in "wcs" units
        :param values_are_sb: True if pixel data give surface brightness, False if they're flux
                          [default: False]


        :returns: None
        """
        self.data = np.copy(data)
        self.noise = noise
        self.weight = np.ones_like(data) * (1./noise/noise)
        self.u0 = u0
        self.v0 = v0
        self.du = du
        self.u = du * np.ones_like(self.data) * np.arange(self.data.shape[1]) - self.u0
        self.v = du * np.ones_like(self.data) * np.arange(self.data.shape[0])[:,np.newaxis] - self.v0
        self.pixel_area = self.du*self.du
        self.values_are_sb = values_are_sb
        
    def addNoise(self):
        """Add noise realization to the image
        """
        self.data += np.random.normal(scale=self.noise, size=self.data.shape)
        return

    def fillFrom(self,function):
        self.data = function(self.u,self.v) * (self.du * self.du)
        return

    # Now implement StarData interface that we care about
    def getDataVector(self):
        """Return as 1d arrays
        """
        mask = self.weight>0.
        return self.data[mask], self.weight[mask], self.u[mask], self.v[mask]

    def setData(self, data):
        """Fill 2d array from 1d array
        """
        mask = self.weight>0.
        newdata = np.zeros_like(self.data)
        print(newdata.shape, mask.shape, self.weight.shape, data.shape)###
        newdata[mask] = data
        return  SimpleData(newdata,
                           noise = self.noise,
                           u0 = self.u0,
                           v0 = self.v0,
                           du = self.du,
                           values_are_sb = self.values_are_sb)

class GaussFunc(object):
    """ Gaussian builder to use in SimpleData.fillFrom()
    """
    def __init__(self, sigma, u0, v0, flux):
        self.sigma = sigma
        self.u0 = u0
        self.v0 = v0
        self.flux = flux
        return
    def __call__(self, u, v):
        out = (u-self.u0)**2 + (v-self.v0)**2
        out = np.exp( out / (-2*self.sigma*self.sigma)) * (self.flux / (2*np.pi*self.sigma*self.sigma))
        return out
    

def test_simplest():
    """Fit a PSF to noiseless Gaussian data at same sampling
    """
    import pixelmodel as pm
    influx = 150.
    du = 0.5
    g = GaussFunc(2.0, 0., 0., influx)
    s = SimpleData(np.zeros((32,32),dtype=float),0.1, 8., 8., du=du)
    s.fillFrom(g)

    # Pixelized model with Lanczos 3 interp
    interp = pm.Lanczos(3)
    mod = pm.PixelModel(du, 32,interp, start_sigma=1.5)
    star = mod.makeStar(s)
    star.fit.flux = np.sum(star.data.data)  # Violating the invariance of Star here!

    star = mod.fit(star)
    print('Flux after fit 1:',star.fit.flux)
    star = mod.reflux(star)
    print('Flux after reflux:',star.fit.flux)
    star = mod.fit(star)
    print('Flux after fit 2:',star.fit.flux)
    star = mod.reflux(star)
    print('Flux after reflux 2:',star.fit.flux)
    star2 = mod.draw(star)
    return star,star2,mod

def test_oversample():
    """Fit to oversampled data, decentered PSF.
    Residual image should be checkeboard from limitations of interpolator.
    """
    
    import pixelmodel as pm
    influx = 150.
    du = 0.25
    nside = 64
    g = GaussFunc(2.0, 0.5, -0.25, influx)
    s = SimpleData(np.zeros((nside,nside),dtype=float),0.1, du*nside/2, du*nside/2, du=du)
    s.fillFrom(g)

    # Pixelized model with Lanczos 3 interp, coarser pix scale
    interp = pm.Lanczos(3)
    mod = pm.PixelModel(2*du, nside/2,interp, start_sigma=1.5)
    star = mod.makeStar(s)
    star.fit.flux = np.sum(star.data.data) # Violating invariance!!

    for i in range(2):
        star = mod.fit(star)
        print('Flux after fit {:d}:'.format(i),star.fit.flux)
        star = mod.reflux(star)
        print('Flux after reflux {:d}:'.format(i),star.fit.flux)
    star2 = mod.draw(star)
    return star,star2,mod

def test_center():
    """Fit with centroid free and PSF center constrained to an initially
    mis-registered PSF.  Residual image when done should be dominated but
    structure off the edge of the fitted region.
    """
    
    import pixelmodel as pm
    influx = 150.
    g = GaussFunc(2.0, 0.6, -0.4, influx)
    s = SimpleData(np.zeros((32,32),dtype=float),0.1, 8., 8., du=0.5)
    s.fillFrom(g)

    # Pixelized model with Lanczos 3 interp, coarser pix scale, smaller
    # than the data
    interp = pm.Lanczos(3)
    # Want an odd-sized model when center=True
    mod = pm.PixelModel(0.5, 29, interp, force_model_center=True, start_sigma=1.5)
    star = mod.makeStar(s)
    star = mod.reflux(star, fit_center=False) # Start with a sensible flux
    star = mod.reflux(star) # and center too
    print('Flux, ctr after reflux:',star.fit.flux,star.fit.center)
    for i in range(3):
        star = mod.fit(star)
        print('Flux, ctr, chisq after fit {:d}:'.format(i),star.fit.flux,star.fit.center, star.fit.chisq)
        star = mod.reflux(star)
        print('Flux, ctr, chisq after reflux {:d}'.format(i),star.fit.flux,star.fit.center, star.fit.chisq)
    star2 = mod.draw(star)
    return star,star2,mod
