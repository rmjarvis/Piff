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

import fitsio
import galsim


class FitsReader:
    """A reader object that reads from to multiple FITS HDUs.

    This reader is intended to read files written by `FitsWriter`.

    :param fits:        An already-open `fitsio.FITS` object.
    :param base_name:   Base name to prepend to all object names, or `None`.
    """
    def __init__(self, fits, base_name):
        self._fits = fits
        self._base_name = base_name

    @classmethod
    @contextmanager
    def open(cls, file_name):
        """Return a context manager that opens the given FITS file and yields
        a `FitsReader` object.

        :param file_name:   Name of the file to read.

        :returns:  A context manager that yields a `FitsReader` instance.
        """
        with fitsio.FITS(file_name, "r") as f:
            yield cls(f, base_name=None)

    def read_struct(self, name):
        """Load a simple flat dictionary.

        :param name:   Name used to save this struct in `FitsWriter.write_struct`.

        :returns:  A `dict` with `str` keys and `int`, `float`, `str`, `bool`,
                   or `None` values.  If nothing was stored with the given
                   name, `None` is returned.
        """
        extname = self.get_full_name(name)
        if extname not in self._fits:
            return None
        cols = self._fits[extname].get_colnames()
        data = self._fits[extname].read()
        assert len(data) == 1
        return dict([ (col, data[col][0]) for col in cols ])

    def read_table(self, name, metadata=None):
        """Load a table as a numpy array.

        :param name:      Name used to save this table in `FitsWriter.write_table`.
        :param metadata:  If not `None`, a `dict` to be filled with any
                          metadata associated with the table on write.  Key
                          case may not be preserved!

        :returns:  A numpy array with a structured dtype, or `None` if no
                   object with this name was saved.
        """
        extname = self.get_full_name(name)
        if extname not in self._fits:
            return None
        if metadata is not None:
            metadata.update(self._fits[extname].read_header())
        return self._fits[extname].read()

    def read_wcs_map(self, name, logger):
        """Load a map of WCS objects and an optional pointing coord.

        :param name:      Name used to save this map in `FitsWriter.write_wcs_map`.
        :param logger:    A logger object for logging debug info.

        :returns:  A 2-element tuple, where the first entry is a `dict` with
                   `int` (chipnum) keys and `galsim.BaseWCS` values, and the
                   second entry is a `galsim.CelestialCoord` object.  Both
                   entries may be `None`, and if the WCS `dict` is `None` the
                   pointing coord is always `None`.
        """
        import base64
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
        """Return a context manager that yields a new `FitsReader` that reads
        content written by a similarly nested `FitsWriter`.

        :param name:     Base name for all objects read with the returned object.

        :returns:     A context manager that yields a nested reader object.
        """
        yield FitsReader(self._fits, base_name=self.get_full_name(name))

    def get_full_name(self, name: str) -> str:
        """Return the full name of a data structure saved with the given name,
        combining it with any base names added if this `FitsReader` was created
        by the `nested` method.

        The FITS implementation concatenates with ``_`` as a delimiter, and
        uses the full name as the EXTNAME header value.
        """
        return name if self._base_name is None else f"{self._base_name}_{name}"
