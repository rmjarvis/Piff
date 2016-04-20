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
    tests of the PixelModel class.  u is first index of internal 2d array, v is second.
    """
    def __init__(self, data, noise, u0, v0, du=1., values_are_sb = False, fpu=0., fpv=0.):
        """

        :param data:  2d numpy array holding the pixel data
        :param noise: RMS Gaussian noise to be assumed on each pixel
        :param u0, v0: nominal center of the star relative to lower-left pixel
        :param du:    pixel size in "wcs" units
        :param values_are_sb: True if pixel data give surface brightness, False if they're flux
                          [default: False]
        :param fpu,fpv: position of this cutout in some larger focal plane


        :returns: None
        """
        self.data = np.copy(data)
        self.noise = noise
        self.weight = np.ones_like(data) * (1./noise/noise)
        self.u0 = u0
        self.v0 = v0
        self.du = du
        self.v = du * np.ones_like(self.data) * np.arange(self.data.shape[1]) - self.v0
        self.u = du * np.ones_like(self.data) * np.arange(self.data.shape[0])[:,np.newaxis] - self.u0
        self.pixel_area = self.du*self.du
        self.values_are_sb = values_are_sb
        self.properties = {'u':fpu, 'v':fpv}
        
    def __getitem__(self,key):
        return self.properties[key]
    
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
        newdata[mask] = data
        out =   SimpleData(newdata,
                           noise = self.noise,
                           u0 = self.u0,
                           v0 = self.v0,
                           du = self.du,
                           values_are_sb = self.values_are_sb,
                           fpu = self.properties['u'],
                           fpv = self.properties['v'])
        out.weight = self.weight
        return out

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


def test_interp():
    """First test of use with interpolator.  Make a bunch of noisy
    versions of the same PSF, interpolate them with constant interp
    to get an average PSF
    """
    from polynomial_interp import Polynomial
    import pixelmodel as pm
    
    # Pixelized model with Lanczos 3 interpolation, slightly smaller than data
    # than the data
    pixinterp = pm.Lanczos(3)
    mod = pm.PixelModel(0.5, 25, pixinterp, start_sigma=1.5)

    # Interpolator will be simple mean
    interp = Polynomial( np.zeros(mod._nparams, dtype=int) )
    
    # Draw stars on a 2d grid of "focal plane" with 0<=u,v<=1
    positions = np.linspace(0.,1.,10.)
    influx = 150.
    g = GaussFunc(1.0, 0., 0., influx)
    stars = []
    for u in positions:
        for v in positions:
            # Draw stars in focal plane positions around a unit ring
            s = SimpleData(np.zeros((32,32),dtype=float),0.1, 8., 8., du=0.5,
                        fpu = u, fpv=v)
            s.fillFrom(g)
            s.addNoise()
            s = mod.makeStar(s)
            s = mod.reflux(s, fit_center=False) # Start with a sensible flux
            stars.append(s)
    # Also store away a noiseless copy of the PSF, origin of focal plane
    s0 = SimpleData(np.zeros((32,32),dtype=float),0.1, 8., 8., du=0.5)
    s0.fillFrom(g)
    s0 = mod.makeStar(s0)
    
    # Iterate solution using interpolator
    for iteration in range(3):
        # Refit PSFs star by star:
        for i,s in enumerate(stars):
            stars[i] = mod.fit(s)
        # Run the interpolator
        interp.solve(stars)
        # Install interpolator solution into each
        # star, recalculate flux, report chisq
        chisq = 0.
        dof = 0
        for i,s in enumerate(stars):
            s = interp.interpolate(s)
            s = mod.reflux(s)
            chisq += s.fit.chisq
            dof += s.fit.dof
            stars[i] = s
        print('iteration',iteration,'chisq=',chisq, 'dof=',dof)
    # Now use the interpolator to produce a noiseless rendering
    s1 = interp.interpolate(s0)
    s1 = mod.reflux(s1)
    s1 = mod.draw(s1)
    return s0,s1,mod

            
def test_missing():
    """Next: fit mean PSF to multiple images, with missing pixels.
    """
    from polynomial_interp import Polynomial
    import pixelmodel as pm
    
    # Pixelized model with Lanczos 3 interpolation, slightly smaller than data
    # than the data
    pixinterp = pm.Lanczos(3)
    mod = pm.PixelModel(0.5, 25, pixinterp, start_sigma=1.5)

    # Interpolator will be simple mean
    interp = Polynomial( np.zeros(mod._nparams, dtype=int) )
    
    # Draw stars on a 2d grid of "focal plane" with 0<=u,v<=1
    positions = np.linspace(0.,1.,4)
    influx = 150.
    g = GaussFunc(1.0, 0., 0., influx)
    stars = []
    for u in positions:
        for v in positions:
            # Draw stars in focal plane positions around a unit ring
            s = SimpleData(np.zeros((32,32),dtype=float),0.1, 8., 8., du=0.5,
                        fpu = u, fpv=v)
            s.fillFrom(g)
            s.addNoise()
            s = mod.makeStar(s)
            # Kill 10% of each star's pixels
            good = np.random.rand(*s.data.data.shape) > 0.1
            s.data.weight = np.where(good, s.data.weight, 0.)
            s.data.data = np.where(good, s.data.data, -999.)
            s = mod.reflux(s, fit_center=False) # Start with a sensible flux
            stars.append(s)
    # Also store away a noiseless copy of the PSF, origin of focal plane
    s0 = SimpleData(np.zeros((32,32),dtype=float),0.1, 8., 8., du=0.5)
    s0.fillFrom(g)
    s0 = mod.makeStar(s0)
    
    oldchi = 0.
    # Iterate solution using interpolator
    for iteration in range(40):
        # Refit PSFs star by star:
        for i,s in enumerate(stars):
            stars[i] = mod.fit(s)
        # Run the interpolator
        interp.solve(stars)
        # Install interpolator solution into each
        # star, recalculate flux, report chisq
        chisq = 0.
        dof = 0
        for i,s in enumerate(stars):
            s = interp.interpolate(s)
            s = mod.reflux(s)
            chisq += s.fit.chisq
            dof += s.fit.dof
            stars[i] = s
            ###print('   chisq=',s.fit.chisq, 'dof=',s.fit.dof)
        print('iteration',iteration,'chisq=',chisq, 'dof=',dof)
        if oldchi>0 and chisq<oldchi and oldchi-chisq < 1.:
            break
        else:
            oldchi = chisq
    # Now use the interpolator to produce a noiseless rendering
    s1 = interp.interpolate(s0)
    s1 = mod.reflux(s1)
    s1 = mod.draw(s1)
    return s0,s1,mod

def test_gradient():
    """Next: fit spatially-varying PSF to multiple images.
    """
    from polynomial_interp import Polynomial
    import pixelmodel as pm
    
    # Pixelized model with Lanczos 3 interpolation, slightly smaller than data
    # than the data
    pixinterp = pm.Lanczos(3)
    mod = pm.PixelModel(0.5, 25, pixinterp, start_sigma=1.5,degenerate=False)

    # Interpolator will be linear
    interp = Polynomial( np.ones(mod._nparams, dtype=int) )
    
    # Draw stars on a 2d grid of "focal plane" with 0<=u,v<=1
    positions = np.linspace(0.,1.,4)
    influx = 150.
    stars = []
    for u in positions:
        # Put gradient in pixel size
        g = GaussFunc(1.0+u*0.1, 0., 0., influx)
        for v in positions:
            # Draw stars in focal plane positions around a unit ring
            s = SimpleData(np.zeros((32,32),dtype=float),0.1, 8., 8., du=0.5,
                        fpu = u, fpv=v)
            s.fillFrom(g)
            s.addNoise()
            s = mod.makeStar(s)
            s = mod.reflux(s, fit_center=False) # Start with a sensible flux
            stars.append(s)
    # Also store away a noiseless copy of the PSF, origin of focal plane
    g = GaussFunc(1.0, 0., 0., influx)
    s0 = SimpleData(np.zeros((32,32),dtype=float),0.1, 8., 8., du=0.5)
    s0.fillFrom(g)
    s0 = mod.makeStar(s0)
    
    oldchi = 0.
    # Iterate solution using interpolator
    for iteration in range(40):
        # Refit PSFs star by star:
        for i,s in enumerate(stars):
            stars[i] = mod.fit(s)
        # Run the interpolator
        interp.solve(stars)
        # Install interpolator solution into each
        # star, recalculate flux, report chisq
        chisq = 0.
        dof = 0
        for i,s in enumerate(stars):
            s = interp.interpolate(s)
            s = mod.reflux(s)
            chisq += s.fit.chisq
            dof += s.fit.dof
            stars[i] = s
            ###print('   chisq=',s.fit.chisq, 'dof=',s.fit.dof)
        print('iteration',iteration,'chisq=',chisq, 'dof=',dof)
        if oldchi>0 and np.abs(oldchi-chisq) < 1.:
            break
        else:
            oldchi = chisq
    # Now use the interpolator to produce a noiseless rendering
    s1 = interp.interpolate(s0)
    s1 = mod.reflux(s1)
    s1 = mod.draw(s1)
    return s0,s1,mod,interp

def test_undersamp():
    """Next: fit PSF to undersampled, dithered data with fixed centroids
    """
    from polynomial_interp import Polynomial
    import pixelmodel as pm
    
    # Pixelized model with Lanczos 3 interpolation, slightly smaller than data
    # than the data
    pixinterp = pm.Lanczos(3)
    du = 0.5
    mod = pm.PixelModel(0.25, 25, pixinterp, start_sigma=1.01)##
    ##,force_model_center=True)

    # Interpolator will be constant
    interp = Polynomial( np.zeros(mod._nparams, dtype=int) )
    
    # Draw stars on a 2d grid of "focal plane" with 0<=u,v<=1
    positions = np.linspace(0.,1.,4)
    influx = 150.
    stars = []
    g = GaussFunc(1.0, 0., 0., influx)
    for u in positions:
        for v in positions:
            # Dither centers by 1 pixel
            phase = (0.5 - np.random.rand(2))*du
            if u==0. and v==0.:
                phase=(0.,0.)
            s = SimpleData(np.zeros((32,32),dtype=float),0.1,
                           8.+phase[0], 8.+phase[1], du=du,
                           fpu = u, fpv=v)
            s.fillFrom(g)
            s.addNoise()
            s = mod.makeStar(s)
            s = mod.reflux(s, fit_center=False) # Start with a sensible flux
            print("phase:",phase,'flux',s.fit.flux)###
            stars.append(s)
    # Also store away a noiseless copy of the PSF, origin of focal plane
    g = GaussFunc(1.0, 0., 0., influx)
    s0 = SimpleData(np.zeros((32,32),dtype=float),0.1, 8., 8., du=0.5,fpu=0., fpv=0.)
    s0.fillFrom(g)
    s0 = mod.makeStar(s0)
    
    oldchi = 0.
    # Iterate solution using interpolator
    for iteration in range(1): ###
        # Refit PSFs star by star:
        stars = [mod.fit(s) for s in stars]
        # Run the interpolator
        interp.solve(stars)
        # Install interpolator solution into each
        # star, recalculate flux, report chisq
        chisq = 0.
        dof = 0
        for i,s in enumerate(stars):
            s = interp.interpolate(s)
            s = mod.reflux(s)
            chisq += s.fit.chisq
            dof += s.fit.dof
            stars[i] = s
            print('   chisq=',s.fit.chisq, 'dof=',s.fit.dof, 'flux=',s.fit.flux)
        print('iteration',iteration,'chisq=',chisq, 'dof=',dof)
        if oldchi>0 and np.abs(oldchi-chisq) < 1.:
            break
        else:
            oldchi = chisq
    # Now use the interpolator to produce a noiseless rendering
    s1 = interp.interpolate(s0)
    s1 = mod.reflux(s1)
    s1 = mod.draw(s1)
    return s0,s1,mod,interp,stars

def test_undersamp_shift():
    """Next: fit PSF to undersampled, dithered data with variable centroids,
    this time using chisq() and summing alpha,beta instead of fit() per star
    """
    from polynomial_interp import Polynomial
    import pixelmodel as pm
    
    # Pixelized model with Lanczos 3 interpolation, slightly smaller than data
    # than the data
    pixinterp = pm.Lanczos(3)
    du = 0.5
    mod = pm.PixelModel(0.3, 25, pixinterp, start_sigma=1.3,force_model_center=True)

    # Interpolator will be constant
    interp = Polynomial( np.zeros(mod._nparams, dtype=int) )
    
    # Draw stars on a 2d grid of "focal plane" with 0<=u,v<=1
    positions = np.linspace(0.,1.,8)
    influx = 150.
    stars = []
    for u in positions:
        for v in positions:
            # Nominal star centers move by +-1/2 pix, real centers another 1/2 pix
            phase1 = (0.5 - np.random.rand(2))*du
            phase2 = (0.5 - np.random.rand(2))*du
            if u==0. and v==0.:
                phase1 = phase2 =(0.,0.)
            g = GaussFunc(1.0, phase2[0], phase2[1], influx)
            s = SimpleData(np.zeros((32,32),dtype=float),0.1,
                           8.+phase1[0], 8.+phase1[1], du=du,
                           fpu = u, fpv=v)
            s.fillFrom(g)
            s.addNoise()
            s = mod.makeStar(s)
            s = mod.reflux(s, fit_center=False) # Start with a sensible flux
            print("phase:",phase2,'flux',s.fit.flux)###
            stars.append(s)
    # Also store away a noiseless copy of the PSF, origin of focal plane
    g = GaussFunc(1.0, 0., 0., influx)
    s0 = SimpleData(np.zeros((32,32),dtype=float),0.1, 8., 8., du=0.5,fpu=0., fpv=0.)
    s0.fillFrom(g)
    s0 = mod.makeStar(s0)
    params = s0.fit.params.copy()
    
    oldchi = 0.
    # Iterate solution using mean of chisq
    for iteration in range(10):
        # Refit PSFs star by star:
        stars = [mod.chisq(s) for s in stars]
        # take mean alpha, beta
        alpha = np.mean(np.stack([s.fit.alpha for s in stars]),axis=0)
        beta = np.mean(np.stack([s.fit.beta for s in stars]),axis=0)
        dparam = np.linalg.solve(alpha,beta)
        params += dparam
        chisq = 0.
        dof = 0
        for i,s in enumerate(stars):
            s.fit.params = params
            s = mod.reflux(s)
            chisq += s.fit.chisq
            dof += s.fit.dof
            stars[i] = s
            ###print('   chisq=',s.fit.chisq, 'dof=',s.fit.dof, 'flux=',s.fit.flux, 'center=',s.fit.center)
        print('iteration',iteration,'chisq=',chisq, 'dof=',dof)
        if oldchi>0 and np.abs(oldchi-chisq) < 1.:
            break
        else:
            oldchi = chisq
    # Now use the interpolator to produce a noiseless rendering
    s0.fit.params = params.copy()
    s1 = mod.reflux(s0)
    s1 = mod.draw(s1)
    return s0,s1,mod,stars

def test_undersamp2():
    """Next: fit PSF to undersampled, dithered data with fixed centroids,
    this time using chisq() and summing alpha,beta instead of fit() per star
    """
    from polynomial_interp import Polynomial
    import pixelmodel as pm
    
    # Pixelized model with Lanczos 3 interpolation, slightly smaller than data
    # than the data
    pixinterp = pm.Lanczos(3)
    du = 0.5
    mod = pm.PixelModel(0.25, 25, pixinterp, start_sigma=1.3)##
    ##,force_model_center=True)

    # Interpolator will be constant
    interp = Polynomial( np.zeros(mod._nparams, dtype=int) )
    
    # Draw stars on a 2d grid of "focal plane" with 0<=u,v<=1
    positions = np.linspace(0.,1.,8)
    influx = 150.
    stars = []
    g = GaussFunc(1.0, 0., 0., influx)
    for u in positions:
        for v in positions:
            # Dither centers by 1 pixel
            phase = (0.5 - np.random.rand(2))*du
            if u==0. and v==0.:
                phase=(0.,0.)
            s = SimpleData(np.zeros((32,32),dtype=float),0.1,
                           8.+phase[0], 8.+phase[1], du=du,
                           fpu = u, fpv=v)
            s.fillFrom(g)
            s.addNoise()
            s = mod.makeStar(s)
            s = mod.reflux(s, fit_center=False) # Start with a sensible flux
            print("phase:",phase,'flux',s.fit.flux)###
            stars.append(s)
    # Also store away a noiseless copy of the PSF, origin of focal plane
    g = GaussFunc(1.0, 0., 0., influx)
    s0 = SimpleData(np.zeros((32,32),dtype=float),0.1, 8., 8., du=0.5,fpu=0., fpv=0.)
    s0.fillFrom(g)
    s0 = mod.makeStar(s0)
    params = s0.fit.params.copy()
    
    oldchi = 0.
    # Iterate solution using mean of chisq
    for iteration in range(10):
        # Refit PSFs star by star:
        stars = [mod.chisq(s) for s in stars]
        # take mean alpha, beta
        alpha = np.mean(np.stack([s.fit.alpha for s in stars]),axis=0)
        beta = np.mean(np.stack([s.fit.beta for s in stars]),axis=0)
        dparam = np.linalg.solve(alpha,beta)
        params += dparam
        chisq = 0.
        dof = 0
        for i,s in enumerate(stars):
            s.fit.params = params
            s = mod.reflux(s)
            chisq += s.fit.chisq
            dof += s.fit.dof
            stars[i] = s
            print('   chisq=',s.fit.chisq, 'dof=',s.fit.dof, 'flux=',s.fit.flux)
        print('iteration',iteration,'chisq=',chisq, 'dof=',dof)
        if oldchi>0 and np.abs(oldchi-chisq) < 1.:
            break
        else:
            oldchi = chisq
    # Now use the interpolator to produce a noiseless rendering
    s0.fit.params = params
    s1 = mod.reflux(s0)
    s1 = mod.draw(s1)
    return s0,s1,mod,stars

            
