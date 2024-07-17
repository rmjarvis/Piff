# Copyright (c) 2024 by Mike Jarvis and the other collaborators on GitHub at
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
.. module:: writers
"""

from contextlib import contextmanager

import os
import fitsio
import galsim
import numpy as np

from .util import make_dtype, adjust_value


class Writer:
    """An interface for writing that abstracts the serialization format.

    The `open` static method should be used to obtain the right writer for
    a particular file based on its extension, but specific subclasses can be
    constructed directly as well.

    A `Writer` subclass is always paired with a `Reader` subclass, and the
    methods of `Reader` each directly correspond to a method of `Writer`.

    All `Reader` and `Writer` methods take a ``name`` argument that is
    associated with the low-level data structured being saved.  When writing,
    an object can chose not to write a subobject at all with a given name, and
    then check to see if the corresponding `Reader` method returns `None`, but
    a `Reader` cannot be used to see what type of data structure was saved with
    a given name; if this can change, the write implementation must store this
    explicitly and use it when invoking the `Reader`.
    """

    @staticmethod
    @contextmanager
    def open(file_name):
        """Return a context manager that yields a `Writer` appropriate for the
        given filename.

        :param filename:   Name of the file to open (`str`).

        :returns:  A context manager that yields a `Writer`.
        """
        _, ext = os.path.splitext(file_name)
        if ext == ".fits" or ext == ".piff":
            with FitsWriter._open(file_name) as writer:
                yield writer
            return
        else:
            raise NotImplementedError(f"No writer for extension {ext!r}.")

    def write_struct(self, name, struct):
        """Write a simple flat dictionary.

        :param name:        Name used to save this struct.
        :param struct:      A `dict` with `str` keys and `int`, `float`, `str`,
                            `bool`, or `None` values.
        """
        raise NotImplementedError()

    def write_table(self, name, table, metadata=None):
        """Write a table via a numpy array.

        :param name:      Name used to save this table.
        :param metadata:  A `dict` of simple metadata to save with the table.
                          Keys must be `str` (case may not be preserved) and
                          values must be `int`, `float`, `str`, or `bool`.
        """
        raise NotImplementedError()

    def write_array(self, name, array, metadata=None):
        """Write a numpy array that does not have a structured dtype.

        :param name:      Name used to save this array.
        :param metadata:  A `dict` of simple metadata to save with the array.
                          Keys must be `str` (case may not be preserved) and
                          values must be `int`, `float`, `str`, or `bool`.
        """
        raise NotImplementedError()

    def write_wcs_map(self, name, wcs_map, pointing):
        """Write a regular a map of WCS objects and an optoinal pointing coord.

        :param name:      Name used to save this struct in `Writer.write_array`.
        :param wcs_map:   A `dict` mapping `int` chipnum to `galsim.BaseWCS`.
        :param pointing:  A `galsim.CelestialCoord`, or `None`.
        :param logger:    A logger object for logging debug info.
        """
        raise NotImplementedError()

    def nested(self, name):
        """Return a context manager that yields a new `Writer` that nests all
        names it is given within this one.

        :param name:     Base name for all objects written with the returned object.

        :returns:     A context manager that yields a nested writer object.

        It is implementation-defined whether the content written by the nested
        writer is actually written immediately or only when the context manager
        exits.
        """
        raise NotImplementedError()

    def get_full_name(self, name):
        """Return the full name of a data structure saved with the given name,
        combining it with any base names added if this `Writer` was created by
        the `nested` method.
        """
        raise NotImplementedError()


class FitsWriter(Writer):
    """A `Writer` implementation that writes to multiple FITS HDUs.

    This reader is intended to writes files that would be readable Piff before
    the `Reader` and `Writer` abstractions were added.

    `FitsWriter.nested` yields a writer that writes to the same file and
    prepends all names with its base name to form the EXTNAME, concatenating
    them with ``_``.

    :param fits:        An already-open `fitsio.FITS` object.
    :param base_name:   Base name to prepend to all object names, or `None`.
    :param header:      Fields to be added to all FITS extension headers.
    """

    def __init__(self, fits, base_name, header):
        self._fits = fits
        self._base_name = base_name
        self._header = header

    @classmethod
    @contextmanager
    def _open(cls, file_name):
        """Return a context manager that opens the given FITS file and yields
        a `FitsWriter` object.

        :param file_name:   Name of the file to read.

        :returns:  A context manager that yields a `FitsWriter` instance.

        This also adds an empty primary HDU and sets up the returned writer
        to add the Piff version to all HDUs (including the primary).
        """
        from . import __version__ as piff_version

        header = {"piff_version": piff_version}
        with fitsio.FITS(file_name, "rw", clobber=True) as f:
            if len(f) == 1:
                f.write(data=None, header=header)
            yield cls(f, base_name=None, header=header.copy())

    def write_struct(self, name, struct):
        # Docstring inherited.
        cols = []
        dtypes = []
        for key, value in struct.items():
            # Don't add values that are None to the table.
            if value is None:
                continue
            dt = make_dtype(key, value)
            value = adjust_value(value, dt)
            cols.append([value])
            dtypes.append(dt)
        table = np.array(list(zip(*cols)), dtype=dtypes)
        return self.write_table(name, table)

    def write_table(self, name, array, metadata=None):
        # Docstring inherited.
        if metadata:
            header = self._header.copy()
            header.update(metadata)
        else:
            header = self._header
        self._fits.write_table(array, extname=self.get_full_name(name), header=header)

    def write_array(self, name, array, metadata=None):
        # Docstring inherited.
        if metadata:
            header = self._header.copy()
            header.update(metadata)
        else:
            header = self._header
        self._fits.write_image(
            array, extname=self.get_full_name(name), header=self._header
        )

    def write_wcs_map(self, name, wcs_map, pointing):
        # Docstring inherited.
        import base64

        try:
            import cPickle as pickle
        except ImportError:
            import pickle
        # Start with the chipnums
        chipnums = list(wcs_map.keys())
        cols = [chipnums]
        dtypes = [("chipnums", int)]

        # GalSim WCS objects can be serialized via pickle
        wcs_str = [base64.b64encode(pickle.dumps(w)) for w in wcs_map.values()]
        max_len = np.max([len(s) for s in wcs_str])
        # Some GalSim WCS serializations are rather long.  In particular, the Pixmappy one
        # is longer than the maximum length allowed for a column in a fits table (28799).
        # So split it into chunks of size 2**14 (mildly less than this maximum).
        chunk_size = 2**14
        nchunks = max_len // chunk_size + 1
        cols.append([nchunks] * len(chipnums))
        dtypes.append(("nchunks", int))

        # Update to size of chunk we actually need.
        chunk_size = (max_len + nchunks - 1) // nchunks

        chunks = [
            [s[i : i + chunk_size] for i in range(0, max_len, chunk_size)]
            for s in wcs_str
        ]
        cols.extend(zip(*chunks))
        dtypes.extend(("wcs_str_%04d" % i, bytes, chunk_size) for i in range(nchunks))

        if pointing is not None:
            # Currently, there is only one pointing for all the chips, but write it out
            # for each row anyway.
            dtypes.extend((("ra", float), ("dec", float)))
            ra = [pointing.ra / galsim.hours] * len(chipnums)
            dec = [pointing.dec / galsim.degrees] * len(chipnums)
            cols.extend((ra, dec))

        data = np.array(list(zip(*cols)), dtype=dtypes)
        self.write_table(name, data)

    @contextmanager
    def nested(self, name):
        # Docstring inherited.
        yield FitsWriter(
            self._fits, base_name=self.get_full_name(name), header=self._header
        )

    def get_full_name(self, name):
        # Docstring inherited.
        return name if self._base_name is None else f"{self._base_name}_{name}"
