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
from numpy.random import default_rng
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

from sklearn.gaussian_process.kernels import RBF

from genwavefront import makeStarsGP


@timer
def test_mkimage(config_file='psf_optatmo.yaml',variables='',output_file='temp.pkl',seed=12345,nstars=8000,usekernel=False,verbose_level=1):
    """This test makes an image's worth of stars
    """

    # random number seeds
    nprng = default_rng(seed)
    rng = galsim.BaseDeviate(seed)

    # read the yaml
    config = piff.read_config(config_file)
    logger = piff.setup_logger(verbose=verbose_level)

    # modify the config from the command line..
    piff.config.parse_variables(config, variables, logger)

    # build the PSF
    psf = piff.PSF.process(config['psf'],logger=logger)

    # build the kernel
    if usekernel:
        kernel = 1. * RBF(1.)
    else:
        kernel = None

    # get params object from psf
    init_params = psf._setup_ofit_params(psf.ofit_initvalues,psf.ofit_bounds,psf.ofit_double_zernike_terms,psf.ofit_fix)

    # fill with random values
    init_params.setValue('opt_size',nprng.uniform(0.8,1.2,1)[0])
    init_params.setValue('opt_L0',nprng.uniform(3.,10.,1)[0])
    init_params.setValue('opt_g1',nprng.uniform(-0.05,0.05,1)[0])
    init_params.setValue('opt_g2',nprng.uniform(-0.05,0.05,1)[0])
    for iz in range(4,11+1):
        init_params.setValue('z%df1' % (iz),nprng.uniform(-0.2,0.2,1)[0])
    init_params.setValue('z5f2',nprng.uniform(-0.3,0.3,1)[0])
    init_params.setValue('z5f3',nprng.uniform(-0.3,0.3,1)[0])
    init_params.setValue('z6f2',nprng.uniform(-0.3,0.3,1)[0])
    init_params.setValue('z6f3',nprng.uniform(-0.3,0.3,1)[0])

    # make an image of fake stars
    starsGP = makeStarsGP(nstars,rng,psf,init_params,kernel=kernel,logger=logger)

    # dump out the stars
    outdict = {"init_params":init_params,"stars":starsGP}
    pickle.dump(outdict,open(output_file,'wb'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('config_file',type=str,help="Configuration Filename")
    parser.add_argument('variables',type=str,nargs='*',help="add options to configuration")
    parser.add_argument('-f', '--output_file', dest='output_file',type=str,help="Output Filename",default='temp.pkl')
    parser.add_argument('-s', '--seed', dest='seed',type=int,help="seed",default=12345)
    parser.add_argument('-n', '--nstars', dest='nstars',type=int,help="nstars",default=8000)
    parser.add_argument('-k', '--usekernel', dest='usekernel',type=bool,help="usekernel?",default=False)


    options = parser.parse_args()
    kwargs = vars(options)

    test_mkimage(**kwargs)
