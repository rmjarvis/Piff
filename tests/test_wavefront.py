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

from piff.wavefront import Wavefront
from piff_test_helper import timer

decaminfo = piff.des.DECamInfo()

def make_blank_star(x, y, chipnum, properties={}, stamp_size=19, **kwargs):
    # make a single empty star object at desired x,y,chipnum
    wcs = decaminfo.get_nominal_wcs(chipnum)
    x_fp,y_fp = decaminfo.getPosition([chipnum], [x], [y])
    properties_in = {'chipnum': chipnum, 'x_fp':x_fp[0], 'y_fp':y_fp[0]}
    properties_in.update(properties)
    star = piff.Star.makeTarget(x=x, y=y, wcs=wcs, stamp_size=stamp_size, properties=properties_in, **kwargs)
    return star

def make_blank_stars(nstars=100,npixels=19):
    # make a list of blank stars
    rng = np.random.default_rng(123459)
    chiplist =  [1] + list(range(3,62+1))  # omit chipnum=2
    chipnum = rng.choice(chiplist,nstars)
    pixedge = 20
    icen = rng.uniform(1+pixedge,2048-pixedge,nstars)   # random pixel position inside CCD
    jcen = rng.uniform(1+pixedge,4096-pixedge,nstars)

    # fill the blank stars
    blank_stars = []
    for i in range(nstars):
        # make the shell of a Star object
        blank_stars.append(make_blank_star(icen[i],jcen[i],chipnum[i],stamp_size=npixels))

    return blank_stars


@timer
def test_init():
    # test the Wavefront constructor

    # Configure Wavefront with two sources, with just one Zernike term each
    #   source1 is from DES data, and is divided by chipnum
    #   source2 is from Zemax simulation, and interpolates over the entire focal plane
    config = {'wavefront_kwargs':
                {'survey': 'des',
                 'source1':
                   {'file': 'input/GPInterp-20140212s2-v22i2.npz',
                    'zlist': [4],
                    'keys': {"x_fp":"xfp","y_fp":"yfp"},     # key in Star: key in .npz
                    'chip': "chipnum",
                    'wavelength': 700.0 },
                 'source2':
                   {'file':  'input/decam_2012-iband-700nm.npz',
                    'zlist': [22],
                    'keys': {"x_fp":"xfp","y_fp":"yfp"},
                    'chip': 'None',
                    'wavelength': 700.0 }
                }
              }
    # make sure we can init the wavefront class
    logger = piff.config.setup_logger(verbose=2)
    wfobj = Wavefront(config['wavefront_kwargs'],logger=logger)
    assert wfobj.interp_objects != None

@timer
def test_interp1():

    # Configure Wavefront with two sources, with just one Zernike term each
    #   source1 is from DES data, and is divided by chipnum
    #   source2 is from Zemax simulation, and interpolates over the entire focal plane

    iZ_source1 = 4
    iZ_source2 = 22
    config = {'wavefront_kwargs':
                {'survey': 'des',
                 'source1':
                   {'file': 'input/GPInterp-20140212s2-v22i2.npz',
                    'zlist': [iZ_source1],
                    'keys': {"x_fp":"xfp","y_fp":"yfp"},     # key in Star: key in .npz
                    'chip': "chipnum",
                    'wavelength': 700.0 },
                 'source2':
                   {'file':  'input/decam_2012-iband-700nm.npz',
                    'zlist': [iZ_source2],
                    'keys': {"x_fp":"xfp","y_fp":"yfp"},
                    'chip': 'None',
                    'wavelength': 700.0 }
                }
              }
    # make sure we can init the wavefront class
    logger = piff.config.setup_logger(verbose=2)
    wfobj = Wavefront(config['wavefront_kwargs'],logger=logger)

    # get wavefronts for a set of locations on the focal Plane
    nstars = 100
    stars = make_blank_stars(nstars=nstars)

    # fill Wavefront in star.data.properties
    new_stars = wfobj.fillWavefront(stars,logger=logger)

    # check that wavefront property is filled for desired Zernike terms in both sources
    for star in new_stars:
        assert star.data.properties['wavefront'][iZ_source1] != 0.0
        assert star.data.properties['wavefront'][iZ_source2] != 0.0

@timer
def test_interp2():

    # Configure Wavefront with two sources, with just one Zernike term each
    #   source1 is from DES data, and is divided by chipnum
    #   source2 is from Zemax simulation, and interpolates over the entire focal plane

    iZ_source1 = 4
    iZ_source2 = 22
    config = {'wavefront_kwargs':
                {'survey': 'des',
                 'source1':
                   {'file': 'input/GPInterp-20140212s2-v22i2.npz',
                    'zlist': [iZ_source1],
                    'keys': {"x_fp":"xfp","y_fp":"yfp"},     # key in Star: key in .npz
                    'chip': "chipnum",
                    'wavelength': 700.0 },
                 'source2':
                   {'file':  'input/decam_2012-iband-700nm.npz',
                    'zlist': [iZ_source2],
                    'keys': {"x_fp":"xfp","y_fp":"yfp"},
                    'chip': 'None',
                    'wavelength': 700.0 }
                }
              }
    # make sure we can init the wavefront class
    logger = piff.config.setup_logger(verbose=2)
    wfobj = Wavefront(config['wavefront_kwargs'],logger=logger)

    # get wavefronts for a set of locations on the focal Plane
    nstars = 100
    stars = make_blank_stars(nstars=nstars)

    # fill Wavefront in star.data.properties
    wf_arr = wfobj.fillWavefront(stars,logger=logger,addtostars=False)

    # check that wavefront property is filled for desired Zernike terms in both sources
    for i in range(len(stars)):
        assert wf_arr[i][iZ_source1] != 0.0
        assert wf_arr[i][iZ_source2] != 0.0

@timer
def test_interp_values():

    iZ_source = 6
    # Configure Wavefront with from DES data, and is divided by chipnum
    config_data = {'wavefront_kwargs':
                {'survey': 'des',
                 'source1':
                   {'file': 'input/GPInterp-20140212s2-v22i2.npz',
                    'zlist': [iZ_source],
                    'keys': {"x_fp":"xfp","y_fp":"yfp"},     # key in Star: key in .npz
                    'chip': "chipnum",
                    'wavelength': 700.0 }
                }
                  }

    logger = piff.config.setup_logger(verbose=2)
    wfobj_data = Wavefront(config_data['wavefront_kwargs'],logger=logger)

    # Configure Wavefront with Zemax simulation, and interpolates over the entire focal plane
    config_zemax = {'wavefront_kwargs':
                {'survey': 'des',
                 'source1':
                   {'file':  'input/decam_2012-iband-700nm.npz',
                    'zlist': [iZ_source],
                    'keys': {"x_fp":"xfp","y_fp":"yfp"},
                    'chip': 'None',
                    'wavelength': 700.0 }
                }
              }
    wfobj_zemax = Wavefront(config_zemax['wavefront_kwargs'],logger=logger)

    # get a list of positions & interpolated values from a file
    interp_file = fitsio.FITS('input/wavefront_interp_unittest.fits')

    interp_table = interp_file[1].read()
    chipnums = interp_table['chipnum']
    ix = interp_table['ix']
    iy = interp_table['iy']
    z6_data = interp_table['z6_data']
    z6_zemax = interp_table['z6_zemax']

    # make stars
    test_locations = []
    n = len(chipnums)
    for i in range(n):
        # make the shell of a Star object
        test_locations.append(make_blank_star(ix[i],iy[i],chipnums[i],stamp_size=19))

    # get wavefront at these Locations
    wftest_data = wfobj_data.fillWavefront(test_locations,logger=logger,addtostars=False)
    wftest_zemax = wfobj_zemax.fillWavefront(test_locations,logger=logger,addtostars=False)

    # check that wavefront value is filled for desired Zernike terms in both the data & zemax wavefront cases
    for i in range(n):
        np.testing.assert_almost_equal(z6_data[i],wftest_data[i,6])
        np.testing.assert_almost_equal(z6_zemax[i],wftest_zemax[i,6])



@timer
def test_interp_scalewavelenth():

    # Configure Wavefront with two sources, with just one Zernike term each
    #   source1 is from DES data, and is divided by chipnum
    #   source2 is from Zemax simulation, and interpolates over the entire focal plane

    iZ_source1 = 4
    iZ_source2 = 22
    config = {'wavefront_kwargs':
                {'survey': 'des',
                 'source1':
                   {'file': 'input/GPInterp-20140212s2-v22i2.npz',
                    'zlist': [iZ_source1],
                    'keys': {"x_fp":"xfp","y_fp":"yfp"},     # key in Star: key in .npz
                    'chip': "chipnum",
                    'wavelength': 700.0 },
                 'source2':
                   {'file':  'input/decam_2012-iband-700nm.npz',
                    'zlist': [iZ_source2],
                    'keys': {"x_fp":"xfp","y_fp":"yfp"},
                    'chip': 'None',
                    'wavelength': 700.0 }
                }
              }
    # make sure we can init the wavefront class
    logger = piff.config.setup_logger(verbose=2)
    wfobj = Wavefront(config['wavefront_kwargs'],logger=logger)

    # get wavefronts for a set of locations on the focal Plane
    nstars = 100
    stars = make_blank_stars(nstars=nstars)

    # fill Wavefront in star.data.properties
    wf_arr = wfobj.fillWavefront(stars,logger=logger,addtostars=False)

    # fill Wavefront in star.data.properties
    wf_arr_mod = wfobj.fillWavefront(stars,logger=logger,addtostars=False,wavelength=900.)

    # check that wavefront property is filled for desired Zernike terms in both sources
    for i in range(len(stars)):
        np.testing.assert_almost_equal(wf_arr_mod[i][iZ_source1]/wf_arr[i][iZ_source1],(700./900.),decimal=6)
        np.testing.assert_almost_equal(wf_arr_mod[i][iZ_source2]/wf_arr[i][iZ_source2],(700./900.),decimal=6)

@timer
def test_interp_des():

    # Configure Wavefront with two sources, one from des and another from 'other'

    iZ_source = [5,6,7,8]

    config = {'wavefront_kwargs':
                {'survey': 'other',
                 'source1':
                   {'file': 'input/decam_2012-iband-700nm.npz',
                    'zlist': iZ_source,
                    'keys': {"x_fp":"xfp","y_fp":"yfp"},     # key in Star: key in .npz
                    'chip': 'None',
                    'wavelength': 700.0 }
                }
                  }

    config_des = {'wavefront_kwargs':
                {'survey': 'des',
                 'source1':
                   {'file':  'input/decam_2012-iband-700nm.npz',
                    'zlist': iZ_source,
                    'keys': {"x_fp":"xfp","y_fp":"yfp"},
                    'chip': 'None',
                    'wavelength': 700.0 }
                }
              }

    # make sure we can init the wavefront class
    logger = piff.config.setup_logger(verbose=2)
    wfobj = Wavefront(config['wavefront_kwargs'],logger=logger)
    wfobj_des = Wavefront(config_des['wavefront_kwargs'],logger=logger)

    # get wavefronts for a set of locations on the focal Plane
    nstars = 100
    stars = make_blank_stars(nstars=nstars)

    # fill Wavefront in star.data.properties
    wf_arr = wfobj.fillWavefront(stars,logger=logger,addtostars=False)
    wf_arr_des = wfobj_des.fillWavefront(stars,logger=logger,addtostars=False)

    # check that wavefront terms are shifted from nominal to des coordinates
    for i in range(len(stars)):
        np.testing.assert_almost_equal(wf_arr_des[i][5],wf_arr[i][5],decimal=6)
        np.testing.assert_almost_equal(wf_arr_des[i][6],-wf_arr[i][6],decimal=6)
        np.testing.assert_almost_equal(wf_arr_des[i][7],-wf_arr[i][8],decimal=6)
        np.testing.assert_almost_equal(wf_arr_des[i][8],-wf_arr[i][7],decimal=6)

@timer
def test_interp_des2():

    # Configure Wavefront with two sources, one from des and another unspecified

    iZ_source = [5,6,7,8]

    config = {'wavefront_kwargs':
               {'source1':
                   {'file': 'input/decam_2012-iband-700nm.npz',
                    'zlist': iZ_source,
                    'keys': {"x_fp":"xfp","y_fp":"yfp"},     # key in Star: key in .npz
                    'chip': 'None',
                    'wavelength': 700.0 }
                }
                  }

    config_des = {'wavefront_kwargs':
                {'survey': 'des',
                 'source1':
                   {'file':  'input/decam_2012-iband-700nm.npz',
                    'zlist': iZ_source,
                    'keys': {"x_fp":"xfp","y_fp":"yfp"},
                    'chip': 'None',
                    'wavelength': 700.0 }
                }
              }

    # make sure we can init the wavefront class
    logger = piff.config.setup_logger(verbose=2)
    wfobj = Wavefront(config['wavefront_kwargs'],logger=logger)
    wfobj_des = Wavefront(config_des['wavefront_kwargs'],logger=logger)

    # get wavefronts for a set of locations on the focal Plane
    nstars = 100
    stars = make_blank_stars(nstars=nstars)

    # fill Wavefront in star.data.properties
    wf_arr = wfobj.fillWavefront(stars,logger=logger,addtostars=False)
    wf_arr_des = wfobj_des.fillWavefront(stars,logger=logger,addtostars=False)

    # check that wavefront terms are shifted from nominal to des coordinates
    for i in range(len(stars)):
        np.testing.assert_almost_equal(wf_arr_des[i][5],wf_arr[i][5],decimal=6)
        np.testing.assert_almost_equal(wf_arr_des[i][6],-wf_arr[i][6],decimal=6)
        np.testing.assert_almost_equal(wf_arr_des[i][7],-wf_arr[i][8],decimal=6)
        np.testing.assert_almost_equal(wf_arr_des[i][8],-wf_arr[i][7],decimal=6)



if __name__ == '__main__':
    test_init()
    test_interp1()
    test_interp2()
    test_interp_values()
    test_interp_scalewavelenth()
    test_interp_des()
    test_interp_des2()
