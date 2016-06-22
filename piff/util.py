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
import os

# Courtesy of
# http://stackoverflow.com/questions/3862310/how-can-i-find-all-subclasses-of-a-given-class-in-python
def get_all_subclasses(cls):
    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses

def ensure_dir(target):
    d = os.path.dirname(target)
    if not os.path.exists(d):
        os.makedirs(d)


def write_kwargs(fits, extname, kwargs):
    """A helper function for writing a single row table into a fits file with the values
    and column names given by a kwargs dict.
    """
    def make_dt_tuple(key, t, size):
        # If size == 0, then it's not an array, so return a 2 element tuple.
        # Otherwise, the size is the third item in the tuple.
        if size == 0:
            return (key, t)
        else:
            return (key, t, size)

    def adjust_value(value, t, size):
        # Possibly adjust the value to really be an int, float, etc.
        if size == 0 or t == str:
            return t(value)
        else:
            # For numpy arrays, we can use astype instead.
            return numpy.array(value).astype(t)

    cols = []
    dtypes = []
    for key, value in kwargs.items():
        # Don't add values that are None to the table.
        if value is None:
            continue
        try:
            # Note: this works for either arrays or strings
            size = len(value)
            t = type(value[0])
        except:
            size = 0
            t = type(value)
        dt = numpy.dtype(t) # just used to categorize the type into int, float, str
        if dt.kind in numpy.typecodes['AllInteger']:
            t = int
        elif dt.kind in numpy.typecodes['AllFloat']:
            t = float
        else:
            t = str
        value = adjust_value(value, t, size)
        dtypes.append( make_dt_tuple(key, t, size) )
        cols.append([value])
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
