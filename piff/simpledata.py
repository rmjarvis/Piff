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
    def __init__(self, data, sigma, u0, v0, du=1.):
        """

        :param data:  2d numpy array holding the pixel data
        :param sigma: noise to be assumed on each pixel
        :param u0, v0: nominal center of the star relative to lower-left pixel
        :param du:    pixel size in "wcs" units

        :returns: None
        """
        self.data = np.copy(data)
        self.sigma = sigma
        self.weight = np.ones_like(data) * (1./sigma/sigma)
        self.u0 = u0
        self.v0 = v0
        self.du = du
        self.u = du * np.ones_like(self.data) * np.arange(self.data.shape[1]) - self.u0
        self.v = du * np.ones_like(self.data) * np.arange(self.data.shape[0])[:,np.newaxis] - self.v0
        self.properties = {'pixel_area':self.du*self.du}
        
    def __getitem__(self,key):
        return self.properties[key]
    
    def addNoise(self):
        """Add noise realization to the image
        """
        self.data += np.random.normal(scale=self.sigma, size=self.data.shape)
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
        self.data[mask] = data
        self.data[np.logical_not(mask)] = 0.
        return

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
    

def test_pix():
    import pixelmodel as pm
    # Test: make a noiseless centered Gaussian, flux = 15
    g = GaussFunc(2.0, 0., 0., 150.)
    s = SimpleData(np.zeros((32,32),dtype=float),0.1, 8., 8., du=0.5)
    s.fillFrom(g)

    # And an identical blank to fill with PSF model
    s2 = SimpleData(np.zeros((32,32),dtype=float),0.1, 8., 8., du=0.5)

    # Pixelized model with Lanczos 3 interp
    interp = pm.Lanczos(3)
    mod = pm.PixelModel(0.5, 32,interp)
    star = mod.makeStar(s)
    star.flux = np.sum(star.data.data)
    star2 = mod.makeStar(s2)
    star2.flux = 1.

    mod.fit(star)
    print('Flux after fit 1:',star.flux)
    mod.reflux(star)
    print('Flux after reflux:',star.flux)
    mod.fit(star)
    print('Flux after fit 2:',star.flux)
    mod.reflux(star)
    print('Flux after reflux 2:',star.flux)
    star2.params = star.params.copy()
    star2.flux = star.flux
    mod.draw(star2)
    return star,star2,mod

def test_pix2():
    # Fit to oversampled data
    import pixelmodel as pm
    influx = 150.
    g = GaussFunc(2.0, 0.5, 0.5, influx)
    s = SimpleData(np.zeros((64,64),dtype=float),0.1, 8., 8., du=0.25)
    s.fillFrom(g)

    # And an identical blank to fill with PSF model
    s2 = SimpleData(np.zeros((64,64),dtype=float),0.1, 8., 8., du=0.25)

    # Pixelized model with Lanczos 3 interp, coarser pix scale
    interp = pm.Lanczos(3)
    mod = pm.PixelModel(0.5, 32,interp)
    star = mod.makeStar(s)
    star.flux = np.sum(star.data.data)
    star2 = mod.makeStar(s2)
    star2.flux = 1.

    mod.fit(star)
    print('Flux after fit 1:',star.flux)
    mod.reflux(star)
    print('Flux after reflux:',star.flux)
    mod.fit(star)
    print('Flux after fit 2:',star.flux)
    mod.reflux(star)
    print('Flux after reflux 2:',star.flux)
    star2.params = star.params.copy()
    star2.flux = star.flux
    mod.draw(star2)
    return star,star2,mod

def test_center():
    # Fit with centroid free and PSF center constrained
    import pixelmodel as pm
    influx = 150.
    g = GaussFunc(2.0, 0.6, 0.6, influx)
    s = SimpleData(np.zeros((32,32),dtype=float),0.1, 8., 8., du=0.5)
    s.fillFrom(g)

    # And an identical blank to fill with PSF model
    s2 = SimpleData(np.zeros((32,32),dtype=float),0.1, 8., 8., du=0.5)

    # Pixelized model with Lanczos 3 interp, coarser pix scale, smaller
    # than the data
    interp = pm.Lanczos(3)
    # Want an odd-sized model when center=True
    mod = pm.PixelModel(0.5, 29, interp, force_model_center=True, start_sigma=1.5)
    star = mod.makeStar(s)
    star.flux = np.sum(star.data.data)
    star2 = mod.makeStar(s2)

    mod.reflux(star, fit_center=False)
    mod.reflux(star)
    print('Flux, ctr after reflux:',star.flux,star.center)
    mod.fit(star)
    print('Flux, ctr after fit 1:',star.flux,star.center)
    mod.reflux(star, fit_center=False)
    mod.reflux(star)
    print('Flux, ctr after reflux 1:',star.flux,star.center)
    mod.fit(star)
    print('Flux, ctr after fit 2:',star.flux,star.center)
    mod.reflux(star, fit_center=False)
    mod.reflux(star)
    print('Flux, ctr after reflux 2:',star.flux,star.center)
    star2.params = star.params.copy()
    star2.flux = star.flux
    mod.draw(star2)
    return star,star2,mod
