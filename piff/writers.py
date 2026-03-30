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

import fitsio
import galsim
import numpy as np

from .util import make_dtype, adjust_value


def _make_chunked_cols(objs, prefix):
    # Some serialized payloads are too long for a single FITS table column
    # (max width 28799). Split them into 2**14-sized chunks, which is safely
    # below that limit. This is needed for some GalSim WCS objects (notably
    # Pixmappy) and also for serialized Bandpass objects.

    import base64
    import pickle

    serialized_list = [base64.b64encode(pickle.dumps(obj)) for obj in objs]
    nrows = len(serialized_list)
    max_len = max(len(s) for s in serialized_list)
    chunk_size = 2**14
    nchunks = max_len // chunk_size + 1
    cols = [[nchunks] * nrows]
    dtypes = [(f"{prefix}_nchunks", int)]

    chunk_size = (max_len + nchunks - 1) // nchunks
    chunks = [
        [s[i : i + chunk_size] for i in range(0, max_len, chunk_size)]
        for s in serialized_list
    ]
    cols.extend(zip(*chunks))
    dtypes.extend((f"{prefix}_str_{i:04d}", bytes, chunk_size) for i in range(nchunks))
    return cols, dtypes


class FitsWriter:
    """A writer object that writes to multiple FITS HDUs.

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
    def open(cls, file_name):
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
            f.write(data=None, header=header)
            yield cls(f, base_name=None, header=header.copy())

    def write_struct(self, name, struct):
        """Write a simple flat dictionary.

        :param name:        Name used to save this struct.
        :param struct:      A `dict` with `str` keys and `int`, `float`, `str`,
                            `bool`, or `None` values.
        """
        cols = []
        dtypes = []
        for key, value in struct.items():
            # Don't add values that are None to the table.
            if value is None:
                continue
            if isinstance(value, dict):
                value = repr(value)
            dt = make_dtype(key, value)
            value = adjust_value(value, dt)
            cols.append([value])
            dtypes.append(dt)
        table = np.array(list(zip(*cols)), dtype=dtypes)
        return self.write_table(name, table)

    def write_table(self, name, array, metadata=None):
        """Write a table via a numpy array.

        :param name:      Name used to save this table.
        :param metadata:  A `dict` of simple metadata to save with the table.
                          Keys must be `str` (case may not be preserved) and
                          values must be `int`, `float`, `str`, or `bool`.
        """
        if metadata:
            header = self._header.copy()
            header.update(metadata)
        else:
            header = self._header
        self._fits.write_table(array, extname=self.get_full_name(name), header=header)

    def write_wcs_map(self, name, wcs_map, pointing):
        """Write a regular a map of WCS objects and an optional pointing coord.

        :param name:      Name used to save this WCS map.
        :param wcs_map:   A `dict` mapping `int` chipnum to `galsim.BaseWCS`.
        :param pointing:  A `galsim.CelestialCoord`, or `None`.
        """
        # Start with the chipnums
        chipnums = list(wcs_map.keys())
        cols = [chipnums]
        dtypes = [("chipnums", int)]

        wcs_cols, wcs_dtypes = _make_chunked_cols(list(wcs_map.values()), "wcs")
        cols.extend(wcs_cols)
        dtypes.extend(wcs_dtypes)

        if pointing is not None:
            # Currently, there is only one pointing for all the chips, but write it out
            # for each row anyway.
            dtypes.extend((("ra", float), ("dec", float)))
            ra = [pointing.ra / galsim.hours] * len(chipnums)
            dec = [pointing.dec / galsim.degrees] * len(chipnums)
            cols.extend((ra, dec))

        data = np.array(list(zip(*cols)), dtype=dtypes)
        self.write_table(name, data)

    def write_bandpass(self, name, bandpass):
        """Write a serialized bandpass object in its own extension.
        """
        cols, dtypes = _make_chunked_cols([bandpass], 'bandpass')
        data = np.array(list(zip(*cols)), dtype=dtypes)
        self.write_table(name, data)

    @contextmanager
    def nested(self, name):
        """Return a context manager that yields a new `FitsWriter` that nests
        all names it is given within this one.

        :param name:     Base name for all objects written with the returned object.

        :returns:     A context manager that yields a nested writer object.
        """
        yield FitsWriter(
            self._fits, base_name=self.get_full_name(name), header=self._header
        )

    def get_full_name(self, name):
        """Return the full name of a data structure saved with the given name,
        combining it with any base names added if this `FitsWriter` was created
        by the `nested` method.

        The FITS implementation concatenates with ``_`` as a delimiter, and
        uses the full name as the EXTNAME header value.
        """
        return name if self._base_name is None else f"{self._base_name}_{name}"
