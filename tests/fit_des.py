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

# Test routines for the pixellated Piff PSF model
"""
Program to fit a set of stars in a FITS image using stars listed in a FITS table.
"""

import numpy as np
import galsim
import astropy.io.fits as pyfits
import piff

import logging

def stardata_from_fits(hdu_list, xysky, stamp_radius=25, badmask=0x7FFF,
                       logger=None):
    """Create StarData instances from data in FITS hdu's.

    :param scihdu:   An astropy FITS HDU list holding the science image
                     and weight image extensions.
    :param xysky:    list of (x,y,sky) tuples for stars
    :param stamp_radius: "radius" of postage stamp to use
    :param badmask:  Bits set in mask plane that invalidate a pixel.
                     [Note: don't use bit 15, it is usually messed up
                     due to int/uint confusions in FITS readers.]

    :returns:        StarData instance
    """
    # Get the image data and weight and mask planes
    sci_extn = -1
    wgt_extn = -1
    msk_extn = -1
    for i,extn in enumerate(hdu_list):
        try:
            desext = extn.header['DES_EXT'].strip()
            if desext=='IMAGE':
                sci_extn = i
            elif desext=='WEIGHT':
                wgt_extn = i
            elif desext=='MASK':
                msk_extn = i
        except KeyError:
            pass

    # ??? Errors if extensions are <0
    def fatal(msg,logger):
        if logger:
            logger_error(msg)
        print(msg)
        sys.exit(1)

    if sci_extn<0:
        fatal('Cannot find IMAGE extension of FITS file')
    if msk_extn<0:
        fatal('Cannot find MASK extension of FITS file')
    if wgt_extn<0:
        fatal('Cannot find WEIGHT extension of FITS file')

    sci = galsim.fits.read(hdu_list=hdu_list[sci_extn], compression='rice')
    wgt = galsim.fits.read(hdu_list=hdu_list[wgt_extn], compression='rice')
    msk = galsim.fits.read(hdu_list=hdu_list[msk_extn], compression='rice')
    hdr = hdu_list[sci_extn].header

    # Null weights using mask bits
    good = np.bitwise_and(msk.array, badmask)==0
    wgt *= np.where(good, 1., 0.)

    # Determine gain and telescope pointing
    props = {}
    try:
        props['gain'] = hdr['GAIN']
    except KeyError:
        # Try GAINA if no GAIN ??? pick correct side??
        props['gain'] = hdr['GAINA']

    # Get exposure pointing from header
    ra = galsim.Angle(hdr['CRVAL1'],galsim.degrees)
    dec = galsim.Angle(hdr['CRVAL2'],galsim.degrees)
    ra = galsim.HMS_Angle(hdr['TELRA'])
    dec = galsim.DMS_Angle(hdr['TELDEC'])
    pointing = galsim.CelestialCoord(ra,dec)
    if logger:
        logger.info("pointing = %s hours, %s deg", pointing.ra/galsim.hours,
                    pointing.dec/galsim.degrees)

    # Now iterate through all stars
    stardata = []
    for x,y,sky in xysky:
        x0 = int(np.floor(x+0.5))
        y0 = int(np.floor(y+0.5))
        xmin = max(x0-stamp_radius, sci.bounds.xmin)
        xmax = min(x0+stamp_radius, sci.bounds.xmax)
        ymin = max(y0-stamp_radius, sci.bounds.ymin)
        ymax = min(y0+stamp_radius, sci.bounds.ymax)
        b = galsim.BoundsI(xmin,xmax,ymin,ymax)

        # Subtract sky counts, get data & weight
        stamp = sci[b] - sky
        weight = wgt[b].copy()

        if np.all(weight.array==0.):
            # No good pixels in a star
            if logger:
                logger.info('Discarding star at (%d,%d) with no valid pixels',x0,y0)
            continue
        # Create StarData
        props['sky'] = sky
        stardata.append(piff.StarData(stamp,
                                      image_pos=galsim.PositionD(x,y),
                                      weight=weight,
                                      pointing=pointing,
                                      properties=props.copy()))
    return stardata, sci.wcs


def fit_des(imagefile, catfile, order=2, nstars=None,
            scale=0.15, size=41, start_sigma=0.4,
            logger=None):
    """
    Fit polynomial interpolated pixelized PSF to a DES image,
    using the stars in a catalog.  For a single CCD.

    :param imagefile:   Path to the image file
    :param catfile:     Path to FITS SExtractor catalog holding only stars
    :param order:       What order to use for the PSF interpolation [default: 2]
    :param nstars:      If desired, a number fo stars to select from the full set to use.
                        [default: None, which means use all stars]
    :param scale:       The scale to use for the Pixel model [default: 0.15]
    :param size:        The size to sue for the Pixel grid [default: 41]
    :param start_sigma: The starting sigma value for Pixel mode [default: 0.4]

    :returns: a completed PSF instance.
    """
    # Get the stellar images
    if logger:
        logger.info("Opening FITS images")
    ff = pyfits.open(imagefile)
    cat = pyfits.getdata(catfile,2)  # ??? hard-wired extension right now

    if nstars is not None:
        index = np.random.choice(len(cat), size=nstars, replace=False)
        cat = cat[index]
    # ??? make any other object cuts here!

    xysky = zip(cat['XWIN_IMAGE'],cat['YWIN_IMAGE'],cat['BACKGROUND'])

    if logger:
        logger.info("Creating %d StarDatas",len(xysky))
    original, wcs = stardata_from_fits(ff, xysky, logger=logger)

    if logger:
        logger.info("...Done making StarData")

    # Add shot noise to data
    data = [s.addPoisson() for s in original]

    stars = [ piff.Star(d, None) for d in data ]

    # Make model, force PSF centering
    model = piff.PixelModel(scale=scale, size=size, interp=piff.Lanczos(3),
                            force_model_center=True, start_sigma = start_sigma,
                            logger=logger)
    # Interpolator will be zero-order polynomial.
    # Find u, v ranges
    u = [s['u'] for s in data]
    v = [s['v'] for s in data]
    uvrange = ( (np.min(u),np.max(u)), (np.min(v),np.max(v)) )
    interp = piff.BasisPolynomial(order, ranges=uvrange, logger=logger)


    # Make a psf
    if logger:
        logger.info("Building PSF")
    wcs = {0 : wcs}
    psf = piff.PSF.build(stars, wcs, model, interp, logger=logger)

    # ??? Do a "refinement" run with the model used to generate
    # the Poisson noise instead of the signal.

    return psf

def subtract_stars(img, psf):
    """Subtract modeled stars from the image.

    :param img:    GalSim Image of a CCD
    :param psf:    PSF model that has been fit to stars on this image

    :returns:      Image with the stellar models subtracted
    """
    for s in psf.stars:
        fitted = psf.draw(s.data, s.fit.flux, s.fit.center)
        img[fitted.data.image.bounds] -= fitted.data.image
    return img

def main():

    logger = piff.config.setup_logger(3)

    image_file = 'y1_test/DECam_00241238_01.fits.fz'
    cat_file = 'y1_test/DECam_00241238_01_psfcat_tb_maxmag_17.0_magcut_3.0_findstars.fits'
    out_file = 'output/no_stars.fits.fz'

    psf = fit_des(image_file, cat_file, order=2, nstars=25, scale=0.2, size=21, logger=logger)

    orig_image = galsim.fits.read(image_file)
    no_stars_img = subtract_stars(orig_image, psf)
    no_stars_img.write(out_file)

    cmd = 'ds9 -zscale -zoom 0.5 %s %s -blink interval 1 -blink'%(image_file,out_file)
    logger.warn('To open this in ds9, blinking the stars on and off, execute the command:')
    logger.warn('\n%s\n',cmd)


if __name__ == '__main__':
    main()
