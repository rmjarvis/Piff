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
.. module:: readers
"""

from contextlib import contextmanager

import os
import fitsio
import galsim
import numpy as np


class Reader:
    """An interface for reading that abstracts the serialization format.

    The `open` static method should be used to obtain the right reader for
    a particular file based on its extension, but specific subclasses can be
    constructed directly as well.

    A `Reader` subclass is always paired with a `Writer` subclass, and the
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
        """Return a context manager that yields a `Reader` appropriate for the
        given filename.

        :param filename:   Name of the file to open (`str`).

        :returns:  A context manager that yields a `Reader`.
        """
        _, ext = os.path.splitext(file_name)
        if ext == ".fits" or ext == ".piff":
            with FitsReader._open(file_name) as reader:
                yield reader
            return
        else:
            raise NotImplementedError(f"No reader for extension {ext!r}.")

    def read_struct(self, name):
        """Load a simple flat dictionary.

        :param name:   Name used to save this struct in `Writer.write_struct`.

        :returns:  A `dict` with `str` keys and `int`, `float`, `str`, `bool`,
                   or `None` values.  If nothing was stored with the given
                   name, `None` is returned.
        """
        raise NotImplementedError()

    def read_table(self, name, metadata=None):
        """Load a table as a numpy array.

        :param name:      Name used to save this table in `Writer.write_table`.
        :param metadata:  If not `None`, a `dict` to be filled with any
                          metadata associated with the table on write.  Key
                          case may not be preserved!

        :returns:  A numpy array with a structured dtype, or `None` if no
                   object with this name was saved.
        """
        raise NotImplementedError()

    def read_array(self, name, metadata=None):
        """Load a regular a numpy array that does not have a structured dtype.

        :param name:      Name used to save this array in `Writer.write_array`.
        :param metadata:  If not `None`, a `dict` to be filled with any
                          metadata associated with the array on write.  Key
                          case may not be preserved!

        :returns:  A numpy array, or `None` if no object with this name was saved.
        """
        raise NotImplementedError()

    def read_wcs_map(self, name, logger):
        """Load a map of WCS objects and an optional pointing coord.

        :param name:      Name used to save this map in `Writer.write_array`.
        :param logger:    A logger object for logging debug info.

        :returns:  A 2-element tuple, where the first entry is a `dict` with
                   `int` (chipnum) keys and `galsim.BaseWCS` values, and the
                   second entry is a `galsim.CelestialCoord` object.  Both
                   entries may be `None`, and if the WCS `dict` is `None` the
                   pointing coord is always `None`.
        """
        raise NotImplementedError()

    def nested(self, name):
        """Return a context manager that yields a new `Reader` that reads
        content written by a similarly nested `Writer`.

        :param name:     Base name for all objects read with the returned object.

        :returns:     A context manager that yields a nested reader object.
        """
        raise NotImplementedError()

    def get_full_name(self, name):
        """Return the full name of a data structure saved with the given name,
        combining it with any base names added if this `Reader` was created by
        the `nested` method.
        """
        raise NotImplementedError()


class FitsReader(Reader):
    """A `Reader` implementation that reads from to multiple FITS HDUs.

    This reader is intended to read files written by Piff before the `Reader`
    and `Writer` abstractions were added.

    `FitsReader.nested` yields a reader that reads from the same file and
    prepends all names with its base name to form the EXTNAME, concatenating
    them with ``_``.

    :param fits:        An already-open `fitsio.FITS` object.
    :param base_name:   Base name to prepend to all object names, or `None`.
    """
    def __init__(self, fits, base_name):
        self._fits = fits
        self._base_name = base_name

    @classmethod
    @contextmanager
    def _open(cls, file_name):
        """Return a context manager that opens the given FITS file and yields
        a `FitsReader` object.

        :param file_name:   Name of the file to read.

        :returns:  A context manager that yields a `FitsReader` instance.
        """
        with fitsio.FITS(file_name, "r") as f:
            yield cls(f, base_name=None)

    def read_struct(self, name):
        # Docstring inherited.
        extname = self.get_full_name(name)
        if extname not in self._fits:
            return None
        cols = self._fits[extname].get_colnames()
        data = self._fits[extname].read()
        assert len(data) == 1
        struct = dict([ (col, data[col][0]) for col in cols ])
        for k, v in struct.items():
            # Convert numpy scalar types to native Python types.  We assume all
            # bytes are supposed to be strs in read_struct, but not in
            # read_table.  Conversions from numeric types are important because
            # numpy scalars don't always satisfy the expected isinstance
            # checks.
            if isinstance(v, bytes):
                struct[k] = v.decode()
            elif isinstance(v, np.bool_):
                struct[k] = bool(v)
            elif isinstance(v, np.integer):
                struct[k] = int(v)
            elif isinstance(v, np.floating):
                struct[k] = float(v)
        return struct

    def read_table(self, name, metadata=None):
        # Docstring inherited.
        return self.read_array(name, metadata)

    def read_array(self, name, metadata=None):
        # Docstring inherited.
        extname = self.get_full_name(name)
        if extname not in self._fits:
            return None
        if metadata is not None:
            metadata.update(self._fits[extname].read_header())
        return self._fits[extname].read()

    def read_wcs_map(self, name, logger):
        # Docstring inherited.
        import base64
        try:
            import cPickle as pickle
        except ImportError:
            import pickle

        extname = self.get_full_name(name)
        if extname not in self._fits:
            return None, None

        assert 'chipnums' in self._fits[extname].get_colnames()
        assert 'nchunks' in self._fits[extname].get_colnames()

        data = self._fits[extname].read()

        chipnums = data['chipnums']
        nchunks = data['nchunks']
        nchunks = nchunks[0]  # These are all equal, so just take first one.

        wcs_keys = [ 'wcs_str_%04d'%i for i in range(nchunks) ]
        wcs_str = [ data[key] for key in wcs_keys ] # Get all wcs_str columns
        try:
            wcs_str = [ b''.join(s) for s in zip(*wcs_str) ]  # Rejoint into single string each
        except TypeError:  # pragma: no cover
            # fitsio 1.0 returns strings
            wcs_str = [ ''.join(s) for s in zip(*wcs_str) ]  # Rejoint into single string each

        wcs_str = [ base64.b64decode(s) for s in wcs_str ] # Convert back from b64 encoding
        # Convert back into wcs objects
        try:
            wcs_list = [ pickle.loads(s, encoding='bytes') for s in wcs_str ]
        except Exception:
            # If the file was written by py2, the bytes encoding might raise here,
            # or it might not until we try to use it.
            wcs_list = [ pickle.loads(s, encoding='latin1') for s in wcs_str ]

        wcs = dict(zip(chipnums, wcs_list))

        try:
            # If this doesn't work, then the file was probably written by py2, not py3
            repr(wcs)
        except Exception:
            logger.info('Failed to decode wcs with bytes encoding.')
            logger.info('Retry with encoding="latin1" in case file written with python 2.')
            wcs_list = [ pickle.loads(s, encoding='latin1') for s in wcs_str ]
            wcs = dict(zip(chipnums, wcs_list))
            repr(wcs)

        # Work-around for a change in the GalSim API with 2.0
        # If the piff file was written with pre-2.0 GalSim, this fixes it.
        for key in wcs:
            w = wcs[key]
            if hasattr(w, '_origin') and isinstance(w._origin, galsim._galsim.PositionD):
                w._origin = galsim.PositionD(w._origin)

        if 'ra' in self._fits[extname].get_colnames():
            ra = data['ra']
            dec = data['dec']
            pointing = galsim.CelestialCoord(ra[0] * galsim.hours, dec[0] * galsim.degrees)
        else:
            pointing = None

        return wcs, pointing

    @contextmanager
    def nested(self, name):
        # Docstring inherited.
        yield FitsReader(self._fits, base_name=self.get_full_name(name))

    def get_full_name(self, name: str) -> str:
        # Docstring inherited.
        return name if self._base_name is None else f"{self._base_name}_{name}"
