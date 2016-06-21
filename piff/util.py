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
.. module:: util
"""

from __future__ import print_function
import numpy

# Courtesy of
# http://stackoverflow.com/questions/3862310/how-can-i-find-all-subclasses-of-a-given-class-in-python
def get_all_subclasses(cls):
    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses


def write_kwargs(fits, extname, kwargs):
    """A helper function for writing a single row table into a fits file with the values
    and column names given by a kwargs dict.
    """
    cols = []
    dtypes = []
    for key, value in kwargs.items():
        t = type(value)
        dt = numpy.dtype(t) # just used to categorize the type into int, float, str
        if dt.kind in numpy.typecodes['AllInteger']:
            i = int(value)
            dtypes.append( (key, int) )
            cols.append([i])
        elif dt.kind in numpy.typecodes['AllFloat']:
            f = float(value)
            dtypes.append( (key, float) )
            cols.append([f])
        else:
            s = str(value)
            dtypes.append( (key, str, len(s)) )
            cols.append([s])
    data = numpy.array(zip(*cols), dtype=dtypes)
    fits.write_table(data, extname=extname)


def read_kwargs(fits, extname):
    """A helper function for reading a single row table from a fits file returning the values
    and column names as a kwargs dict.
    """
    cols = fits[extname].get_colnames()
    data = fits[extname].read()
    assert len(data) == 1
    kwargs = dict([ (col, data[col][0]) for col in cols ])
    return kwargs
