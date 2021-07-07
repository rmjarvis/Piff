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
def test_fit(config_file='psf_optatmo.yaml',variables='',output_file='temp.pkl',verbose_level=1):
    """This test makes stars
    """
    config = piff.read_config(config_file)
    logger = piff.setup_logger(verbose=verbose_level)

    # modify the config from the command line..
    piff.config.parse_variables(config, variables, logger)

    # load stars
    doinput = False
    if doinput:
        logger = piff.setup_logger(verbose=True)
        stars, wcs, pointing = piff.Input.process(config['input'], logger=logger)
        print(len(stars))

        # save stars in a pickle for fast retreival
        outdict = {'stars':stars, 'wcs':wcs, 'pointing':pointing}
        pickle.dump(outdict,open("stars-228724.pkl",'wb'))

    else:
        newdict = pickle.load(open("stars-228724.pkl",'rb'))
        stars = newdict['stars']
        wcs = newdict['wcs']
        pointing = newdict['pointing']
        print(len(stars))

    # make a psf object!
    psf = piff.PSF.process(config['psf'],logger=logger)

    ofit_results,params,chiparam = psf.fit(stars, wcs, pointing, logger=logger)
    print(ofit_results)
    params.print()

    # dump all output
    resdict = {'results':ofit_results,'params':params,'chiparam':chiparam,'data_stars':psf.stars,'model_stars':psf.ofit_model_stars}
    pickle.dump(resdict,open(output_file,'wb'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('config_file',type=str,help="Configuration Filename")
    parser.add_argument('variables',type=str,nargs='*',help="add options to configuration")
    parser.add_argument('-r', '--output_file', dest='output_file',type=str,help="Output Filename")
    options = parser.parse_args()
    kwargs = vars(options)

    test_fit(**kwargs)
