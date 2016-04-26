# Test routines for the pixellated Piff PSF model
"""
Program to fit a set of stars in a FITS image using stars
listed in a FITS table.
"""

import numpy as np
import galsim as gs
import astropy.io.fits as pf
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
    
    sci = gs.fits.read(hdu_list=hdu_list[sci_extn], compression='rice')
    wgt = gs.fits.read(hdu_list=hdu_list[wgt_extn], compression='rice')
    msk = gs.fits.read(hdu_list=hdu_list[msk_extn], compression='rice')
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
    ra = gs.Angle(hdr['CRVAL1'],gs.degrees)
    dec = gs.Angle(hdr['CRVAL2'],gs.degrees)
    pointing = gs.CelestialCoord(ra,dec)

    # Now iterate through all stars
    stardata = []
    for x,y,sky in xysky:
        x0 = int(np.floor(x+0.5))
        y0 = int(np.floor(y+0.5))
        xmin = max(x0-stamp_radius, sci.bounds.xmin)
        xmax = min(x0+stamp_radius, sci.bounds.xmax)
        ymin = max(y0-stamp_radius, sci.bounds.ymin)
        ymax = min(y0+stamp_radius, sci.bounds.ymax)
        b = gs.BoundsI(xmin,xmax,ymin,ymax)

        # Subtract sky counts, get data & weight
        stamp = sci[b] - sky
        weight = wgt[b].copy()

        if np.all(weight.array==0.):
            # No good pixels in a star
            if logger:
                logger.info('Discarding star at (%d,%d) with no valid pixels',x0,y0)
            continue
        # Create StarData
        stardata.append(piff.StarData(stamp,
                                image_pos=gs.PositionD(x,y),
                                weight=weight,
                                pointing=pointing,
                                properties=props.copy()))
    return stardata

    
def fit_des(imagefile, catfile, order=2):
    """
    Fit polynomial interpolated pixelized PSF to a DES image,
    using the stars in a catalog.  For a single CCD.

    :param imagefile:  path to the image file
    :param catfile:    path to FITS SExtractor catalog holding only stars

    :returns:          Completed PSF instance.
    """

    logger = logging.getLogger('PSF')
    logging.basicConfig(level=logging.DEBUG)
    
    model_scale = 0.15 # arcsec per sample in PSF model
    model_rad =   20   # "radius" of PSF model box, in samples
    model_start_sigma = 1.0/2.38  # sigma of Gaussian to use for starting guess


    # Get the stellar images
    logger.info("Opening FITS images")
    ff = pf.open(imagefile)
    cat = pf.getdata(catfile,2)  # ??? hard-wired extension right now
    # ??? make object cuts here!
    xysky = zip(cat['XWIN_IMAGE'],cat['YWIN_IMAGE'],cat['BACKGROUND'])

    logger.info("Creating %d StarDatas",len(xysky))
    original = stardata_from_fits(ff, xysky)

    logger.info("...Done making StarData")

    # Add shot noise to data
    data = [s.addPoisson() for s in original]

    # Make model, force PSF centering
    model = piff.PixelModel(du=model_scale, n_side = 2*model_rad+1, interp=piff.Lanczos(3),
                            force_model_center=True, start_sigma = model_start_sigma,
                            logger=logger)
    # Interpolator will be zero-order polynomial.
    # Find u, v ranges
    u = [s['u'] for s in data]
    v = [s['v'] for s in data]
    uvrange = ( (np.min(u),np.max(u)), (np.min(v),np.max(v)) )
    basis = piff.PolyBasis(order, ranges=uvrange)
    interp = piff.BasisInterpolator(basis=basis, logger=logger)

    
    # Make a psf
    logger.info("Building PSF")
    psf = piff.PSF.build(data, model, interp, logger=logger)

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

