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
import pickle
import piff
import fitsio
import os
import warnings
import coord
import pixmappy
import galsim_extra
import pdb

from piff_test_helper import get_script_name, timer, CaptureLog

@timer
def get_stars(config_file,verbose_level=1):

    config = piff.read_config(config_file)
    logger = piff.setup_logger(verbose=verbose_level)

    stars, wcs, pointing = piff.Input.process(config['input'], logger=logger)
    print("Nstars = ",len(stars))

    good_stars = piff.Select.process(config['select'],stars,logger=logger)
    print("Ngood_stars = ",len(good_stars))

    return good_stars,wcs,pointing

def write_stars(stars,wcs,pointing,filename):

    outdict = {'stars':stars, 'wcs':wcs, 'pointing':pointing}
    pickle.dump(outdict,open(filename,'wb'))

def read_stars(star_file):

    indict = pickle.load(open(star_file,'rb'))
    stars = indict['stars']

    if 'wcs' in indict:
        wcs = indict['wcs']
    else:
        wcs = None
    if 'pointing' in indict:
        pointing = indict['pointing']
    else:
        pointing = None
    return stars,wcs,pointing

@timer
def fit_optatmo(config_file,variables,stars,wcs=None,pointing=None,verbose_level=1):
    """
    This test fits stars with optatmo_psf

    :param config_file:                     Configuration file for optatmo_psf 
    :param variables:                       String with additional configuration variables
    :param stars:                           List of stars to fit
    :param verbose_level:                   Verbose level for logger [default: 1]
    """

    config = piff.read_config(config_file)
    logger = piff.setup_logger(verbose=verbose_level)

    # modify the config from the command line..
    piff.config.parse_variables(config, variables, logger)

    # make a psf object!
    psf = piff.PSF.process(config['psf'],logger=logger)

    fit_results = psf.fit(stars, wcs, pointing, logger=logger)
    fit_results['ofit_chiparam'] = psf.ofit_chiparam

    # dump all output
    fit_results['data_stars'] = psf.stars
    return fit_results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('config_file',type=str,help="Configuration Filename")
    parser.add_argument('variables',type=str,nargs='*',help="add options to configuration",default='')
    parser.add_argument('-i', '--input_file', dest='input_file',type=str,help="Input Filename",default='mkimage.pkl')
    parser.add_argument('-o', '--output_file', dest='output_file',type=str,help="Output Filename",default='optatmo_fit.pkl')
    options = parser.parse_args()
    kwargs = vars(options)

    if options.input_file=="None":
        stars,wcs,pointing = get_stars(options.config_file)
    else:
        stars,wcs,pointing = read_stars(options.input_file)

    outdict = fit_optatmo(options.config_file,options.variables,stars,wcs,pointing)
    pickle.dump(outdict,open(options.output_file,'wb'))
