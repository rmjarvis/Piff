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


import logging
from piff.config import VERBOSE, PiffLogger

# Some helper functions that mutliple test files might want to use

def make_gaussian_data(sigma, u0, v0, flux, noise=0., du=1., fpu=0., fpv=0., nside=32,
                       nom_u0=0., nom_v0=0., rng=None, wcs=None, g1=0., g2=0.,
                       draw_method='auto'):
    """Make a Star instance filled with a Gaussian profile.

    :param sigma:       The sigma of the Gaussian.
    :param u0, v0:      The world-coordinate offset of the Gaussian from the catalog position.
    :param flux:        The flux of the star.
    :param noise:       RMS Gaussian noise to add to each pixel. [default: 0]
    :param du:          Pixel size, also used to convert nominal offsets to pixels. [default: 1]
    :param fpu, fpv:    Position of this cutout in the larger focal plane. [default: 0, 0]
    :param nside:       The size of the square array. [default: 32]
    :param nom_u0, nom_v0: Nominal offsets from the stamp center, in units of du. [default: 0, 0]
    :param rng:         GalSim random deviate for adding noise. [default: None]
    :param wcs:         Optional WCS replacing PixelScale(du). [default: None]
    :param g1, g2:      Reduced shear of the Gaussian. [default: 0, 0]
    :param draw_method: GalSim drawing method; use 'no_pixel' for sampled intensities.
                        [default: 'auto', which includes pixel integration]
    """
    import galsim
    import piff

    profile = galsim.Gaussian(sigma=sigma, flux=flux).shear(g1=g1, g2=g2).shift(u0, v0)
    if noise == 0.:
        variance = 1.e-6
    else:
        variance = noise**2
    if wcs is None:
        wcs = galsim.PixelScale(du)
    weight = galsim.Image(nside, nside, dtype=float, init_value=1./variance, wcs=wcs)
    star = piff.Star.makeTarget(x=nside/2+nom_u0/du, y=nside/2+nom_v0/du,
                                u=fpu, v=fpv, wcs=wcs, stamp_size=nside, weight=weight)
    star.image.setOrigin(0, 0)
    star.weight.setOrigin(0, 0)
    profile.drawImage(star.image, use_true_center=False, method=draw_method,
                      offset=galsim.PositionD(nom_u0/du, nom_v0/du))
    if noise != 0:
        star.image.addNoise(galsim.GaussianNoise(sigma=noise, rng=rng))
    return star

def which(program):
    """
    Mimic functionality of unix which command
    """
    import sys,os
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    if sys.platform == "win32" and not program.endswith(".exe"):
        program += ".exe"

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None

def get_script_name(file_name):
    """
    Check if the file_name is in the path.  If not, prepend appropriate path to it.
    """
    import os
    if which(file_name) is not None:
        return file_name
    else:
        test_dir = os.path.split(os.path.realpath(__file__))[0]
        root_dir = os.path.split(test_dir)[0]
        script_dir = os.path.join(root_dir, 'scripts')
        exe_file_name = os.path.join(script_dir, file_name)
        print('Warning: The script %s is not in the path.'%file_name)
        print('         Using explcit path for the test:',exe_file_name)
        return exe_file_name

def timer(f):
    import functools

    @functools.wraps(f)
    def f2(*args, **kwargs):
        import time
        import inspect
        t0 = time.time()
        result = f(*args, **kwargs)
        t1 = time.time()
        fname = repr(f).split()[1]
        print('time for %s = %.2f' % (fname, t1-t0))
        return result
    return f2

def timer(f):
    import functools

    @functools.wraps(f)
    def f2(*args, **kwargs):
        import time
        t0 = time.time()
        result = f(*args, **kwargs)
        t1 = time.time()
        fname = repr(f).split()[1]
        print('time for %s = %.2f' % (fname, t1-t0))
        return result
    return f2

class CaptureLog(object):
    """A context manager that saves logging output into a string that is accessible for
    checking in unit tests.

    After exiting the context, the attribute `output` will have the logging output.

    Sample usage:

            >>> with CaptureLog() as cl:
            ...     cl.logger.info('Do some stuff')
            >>> assert cl.output == 'Do some stuff'

    """
    def __init__(self, level=3):
        logging_levels = { 0: logging.ERROR,
                           1: logging.INFO,
                           2: VERBOSE,
                           3: logging.DEBUG }
        self.logger = PiffLogger(logging.getLogger('CaptureLog'))
        self.logger.setLevel(logging_levels[level])
        try:
            from StringIO import StringIO
        except ImportError:
            from io import StringIO
        self.stream = StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.logger.addHandler(self.handler)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.handler.flush()
        self.output = self.stream.getvalue().strip()
        self.handler.close()

# Context to make it easier to profile bits of the code
class profile(object):
    def __init__(self, sortby='tottime', nlines=30, file=None):
        self.sortby = sortby
        self.nlines = nlines
        self.file = file

    def __enter__(self):
        import cProfile, pstats
        self.pr = cProfile.Profile()
        self.pr.enable()
        return self

    def __exit__(self, type, value, traceback):
        import pstats
        self.pr.disable()
        if self.file:
            self.pr.dump_stats(self.file)
        ps = pstats.Stats(self.pr).sort_stats(self.sortby)
        ps.print_stats(self.nlines)
