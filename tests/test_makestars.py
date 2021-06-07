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
import pdb

from piff_test_helper import get_script_name, timer, CaptureLog

@timer
def test_makestars():
    """This test makes stars
    """
    config = piff.read_config('psf_optatmo.yaml')

    # load stars
    newdict = pickle.load(open('new_stars-228724.pkl','rb'))
    print(newdict.keys())

    # make a psf object!
    logger = piff.setup_logger(verbose=3)
    psf = piff.PSF.process(config['psf'],logger=logger)

    print(psf.opt_param_names)
    params = np.zeros(18)
    params[psf.opt_param_names['opt_size']] = 1.0
    params[psf.opt_param_names['opt_L0']] = 25.0
    params[psf.opt_param_names['opt_g1']] = 0.05
    params[psf.opt_param_names['opt_g2']] = -0.025
    params[psf.opt_param_names['z4f1']] = 0.1

    model_stars = psf.make_modelstars(params,newdict['stars'][0:1000],psf.model,logger)

if __name__ == '__main__':
    import cProfile, pstats, io
    pr = cProfile.Profile()
    pr.enable()
    test_makestars()
    pr.disable()
    ps = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(10)
    print()
    ps.sort_stats(pstats.SortKey.TIME).print_stats(10)
