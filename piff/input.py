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
.. module:: input
"""

import numpy as np
import scipy
import glob
import os
import galsim
import yaml
import copy
from dataclasses import dataclass, field

from .util import run_multi, read_asdf_image, get_asdf_value
from .star import Star, StarData
from .config import LoggerWrapper

@dataclass
class ImageMetadata:
    """Meta-data from an image on disk.

    It includes the bounds, wcs, and optional header.
    """
    bounds: galsim.BoundsI
    wcs: galsim.BaseWCS
    header: object


@dataclass(kw_only=True)
class RawImageData:
    """Image and catalog data returned by `InputFiles.getRawImageData`.

    Fields can be accessed by name or unpacked in the order listed below.

    :param image:       Science image after image-level sky subtraction.
    :param weight:      Weight image after masking and any source-variance removal.
    :param image_pos:   List of catalog positions in image coordinates.
    :param extra_props: Dict of additional catalog properties, with an array for each property.
    :param gain_image:  Per-pixel gain image, or None when no gain image was supplied.
    """
    # These are the primary attributes, which are used by callers of getRawImageData()
    image: galsim.Image | None = None
    weight: galsim.Image | None = None
    image_pos: list | None = None
    extra_props: dict = field(default_factory=dict)
    gain_image: galsim.Image | None = None

    # The rest of these are intermediate products we use when building the above data.
    image_num: int
    config: dict
    base: dict
    mask: np.ndarray | None = None
    sky: float | str | galsim.Image | None = None

    def __post_init__(self):
        # Copy one object graph so nested config references stay connected inside the job.
        self.config, self.base = copy.deepcopy((self.config, self.base))
        self.base['index_key'] = 'image_num'
        self.base['image_num'] = self.image_num

    def __iter__(self):
        yield self.image
        yield self.weight
        yield self.image_pos
        yield self.extra_props
        yield self.gain_image


class Input(object):
    """The base class for handling inputs for building a Piff model.

    This is essentially an abstract base class intended to define the methods that should be
    implemented by any derived class.
    """
    # This class-level dict will store all the valid input types.
    # Each subclass should set a cls._type_name, which is the name that should
    # appear in a config dict.  These will be the keys of valid_input_types.
    # The values in this dict will be the Input sub-classes.
    valid_input_types = {}

    nproc = 1  # Sub-classes can overwrite this as an instance attribute.

    # These shouldn't ever be used, but give these defaults before they are set correctly.
    wcs_list = []
    pointing = None
    bandpass = None

    @classmethod
    def process(cls, config, base=None, logger=None):
        """Parse the input field of the config dict.

        :param config:      The configuration dict for the input field.
        :param base:        The base configuration dict.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: stars, wcs, pointing, bandpass

        stars is a list of Star instances with the initial data.
        wcs is a dict of WCS solutions indexed by chipnum.
        pointing is either a galsim.CelestialCoord or None.
        bandpass is either a galsim.Bandpass or None.
        """
        # Get the class to use for handling the input data
        # Default type is 'Files'
        input_type = config.get('type','Files')
        base = base if base is not None else {'input' : config}

        if input_type not in Input.valid_input_types:
            raise ValueError("type %s is not a valid model type. "%input_type +
                             "Expecting one of %s"%list(Input.valid_input_types.keys()))

        input_class = Input.valid_input_types[input_type]

        # Build handler object
        input_handler = input_class(config, base, logger)

        # Creat a lit of StarData objects
        stars = input_handler.makeStars(logger)

        if len(stars) == 0:
            raise RuntimeError("No objects read in from input catalog(s).")

        # Get the wcs for all the input chips
        wcs = input_handler.getWCS(logger)

        # Get the pointing (the coordinate center of the field of view)
        pointing = input_handler.getPointing(logger)

        # Get the observing bandpass, if any
        bandpass = input_handler.getBandpass(logger)

        return stars, wcs, pointing, bandpass

    @classmethod
    def __init_subclass__(cls):
        if hasattr(cls, '_type_name') and cls._type_name is not None:
            if cls._type_name in Input.valid_input_types:
                raise ValueError('Input type %s already registered'%cls._type_name +
                                 'Maybe you subclassed and forgot to set _type_name?')
            Input.valid_input_types[cls._type_name] = cls

    def makeStars(self, logger=None):
        """Process the input images and star data, cutting out stamps for each star along with
        other relevant information.

        Read metadata on first use to establish image WCS solutions and the common pointing.
        Per-image settings are parsed in each worker's private configuration context.

        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a list of Star instances
        """
        logger = LoggerWrapper(logger)

        if self.metadata_list is None:
            self.readMetadata(logger)

        if self.nimages == 1:
            logger.debug("Making star list")
        else:
            logger.debug("Making star list from %d catalogs", self.nimages)

        args = [(image_num,) for image_num in range(self.nimages)]

        all_stars = run_multi(self._makeStarsFromImage, self.nproc, raise_except=True,
                              args=args, logger=logger)

        # Concatenate the star lists into a single list
        stars = [s for slist in all_stars if slist is not None for s in slist if slist]

        logger.info("Read a total of %d objects from %d image%s",len(stars),self.nimages,
                    "s" if self.nimages > 1 else "")

        return stars

    def getWCS(self, logger=None):
        """Get the WCS solutions for all the chips in the field of view.

        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a dict of WCS solutions (galsim.BaseWCS instances) indexed by chipnum
        """
        if self.metadata_list is None:
            self.readMetadata(logger)
        return { chipnum : w for w, chipnum in zip(self.wcs_list, self.chipnums) }

    def getPointing(self, logger=None):
        """Get the pointing coordinate of the (noinal) center of the field of view.

        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a galsim.CelestialCoord of the pointing direction.
        """
        if self.metadata_list is None:
            self.readMetadata(logger)
        return self.pointing

    def getBandpass(self, logger=None):
        """Get the observing bandpass for the input images.

        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a galsim.Bandpass or None.
        """
        if self.metadata_list is None:
            self.readMetadata(logger)
        return self.bandpass


class InputFiles(Input):
    """An Input handler that just takes a list of image files and catalog files.

    Parse the input config dict (Normally the 'input' field in the overall configuration dict).

    We currently support FITS and ASDF file formats for the input images, and FITS or Parquet
    files for the catalog data.

    For FITS files, the *_ext parameters refer to the HDU with the given data, which may be
    specified either by number or name.

    For ASDF files, the *_ext parameters refer to the path in the ASDF tree. Use dots to
    indicate heirarchy levels. E.g. roman.data means af['roman']['data'].

    FITS files automatically load the WCS data given in the image header if present.
    ASDF files don't seem to have a standard location, so you must specify the WCS
    explicitly with an input.wcs field.

    The two required fields in the input dict are:

        :image_file_name:   The file name(s) of the input image(s).
        :cat_file_name:     The file name(s) of the input catalog(s).

    There are a number of ways to specify these file names.

    1. A string giving a single file name.  e.g.::

            image_file_name: image.fits
            cat_file_name: input_cat.fits

    2. A list of several file names.  e.g.::

            image_file_name: [image_00.fits, image_01.fits, image_02.fits]
            cat_file_name: [input_cat_00.fits, input_cat_01.fits, input_cat_02.fits]

    3. A string that glob can recognize to list several file names.  e.g.::

            image_file_name: image_*.fits
            cat_file_name: input_cat_*.fits

    4. A dict parseable as a string value according to the GalSim configuration parsing types.
       In this case, you also must specify nimages to say how many file names to generate
       in this way.  e.g.::

            nimages: 20
            image_file_name:
                type: FormattedStr
                format: image_%03d_%02d.fits.fz
                items:
                    - { type : Sequence, first: 0, repeat: 4 }  # Exposure number
                    - { type : Sequence, first: 1, last: 4 }    # Chip number
            cat_file_name:
                type: Eval
                str: "image_file_name.replace('image','input_cat')"
                simage_file_name: '@input.image_file_name'

        See the description of the GalSim config parser for more details about the various
        types that are valid here.

            `https://github.com/GalSim-developers/GalSim/wiki/Config-Values`_

    For chromatic PSF fitting, you must define the SED of each star in one of the
    following ways:

    1. Use a single SED for each every star.

       In this case, the SED should probably be some kind of median SED of the stars
       in your sample.  You won't be capturing the variation in the chromaticity among
       the various stars, but at least you will be using the correct average chromatic
       effect of the stellar population.  For this case, give the SED file name as
       ``sed_file_name``.

       Sample config file excerpt::

            input:
                [...]
                sed_file_name: 'stellar_sed.fits'
                sed_wave_type: 'nm'
                sed_flux_type: 'erg cm**(-2) s**(-1) angstrom**(-1)'

    2. Specify a file name for the SED of each star.

       You may instead want to specify a (potentially) different SED file for each star.
       In this case, you can give that file name in the input catalog as one of the
       columns.  Set ``sed_col`` to the name of the catalog column containing the file names.

       Sample config file excerpt::

            input:
                [...]
                sed_col: 'sed_file'
                sed_wave_type: 'nm'
                sed_flux_type: 'erg cm**(-2) s**(-1) angstrom**(-1)'

    3. Specify the SED of each star by some label.

       Another way to give each star its own SED is to have the values in the input catalog
       be some kind of label indicating which file to use.  The mapping of these labels to
       file names would then be in a YAML file, which should parse as a dict giving
       ``sed_file_name[label]``.  In this case, specify ``sed_col`` to be the input column
       with the labels and ``sed_file_name`` to be the YAML file with the mapping.

       Sample config file excerpt::

            input:
                [...]
                sed_col: 'sed_label'
                sed_file_name: 'sed_mapping.yaml'
                sed_wave_type: 'nm'
                sed_flux_type: 'erg cm**(-2) s**(-1) angstrom**(-1)'

       If you want, you can also put the mapping dict directly in the config file::

            input:
                [...]
                sed_col: 'stellar_type'
                sed_file_name:
                    'F': 'xsl_spectrum_X0529_merged.fits'
                    'G': 'xsl_spectrum_X0203_merged.fits'
                    'K': 'xsl_spectrum_X0066_merged_scl.fits'
                    'M': 'xsl_spectrum_X0527_merged.fits'
                sed_wave_type: 'nm'
                sed_flux_type: 'erg cm**(-2) s**(-1) angstrom**(-1)'

    In each case, ``sed_wave_type`` and ``sed_flux_type`` are required values, specifying
    the units to use for the wavelengths and flux values in the SED file.  For SED files
    that are FITS tables, you may also want to set ``sed_wave_key`` and ``sed_flux_key`` to
    specify which columns correspond to the wave and flux arrays.

    There are many other optional parameters, which help govern how the input files are
    read or interpreted:

        :chipnum:       The id number of this chip used to reference this image [default:
                        image_num]

        :image_ext:     The FITS extension number/name or ASDF path to the image.
                        [default: None, which means use either 0 or 1 according to FITS
                        compression; required for ASDF files]
        :weight_ext:    The FITS extension number/name or ASDF path for weight images.
                        [default: None, which means use the weight file's default extension
                        if supplied, otherwise create a weight image with all 1's]
        :badpix_ext:    The FITS extension number/name or ASDF path for badpix images.
                        Pixels with badpix != 0 will be given weight == 0. [default: None]
        :noise:         Rather than a weight image, provide the noise variance in the image.
                        (Useful for simulations where this is a known value.) [default: None]
        :weight_file_name: If the weight image is in a different file than the main image,
                        this gives the name of the file to use. [default: None]
        :badpix_file_name: If the badpix image is in a different file than the main image,
                        this gives the name of the file to use. [default: None]
        :badpix_zeros:  Whether to treat all zeros in the input image as bad pixels.
                        [default: False]
        :fix_ushort_badpix: Correct a legacy 32768 offset in the bad-pixel mask. Old FITS files
                        sometimes offset their badpix arrays by 32768. Set this to True to
                        undo that. [default: False]
        :sky_file_name: a file containing a sky background image to subtract. [default: none]
        :sky_ext:       the FITS extension number/name or ASDF path in the sky file,
                        or in the image file if sky_file_name is omitted.
                        [default: none, which means use the sky file's default image extension]

        :cat_ext:       the extension number or name in fits catalog files. ignored for parquet
                        catalogs. [default: none, which means use the first table extension]
        :x_col:         the name of the x column in the input catalogs. [default: 'x']
        :y_col:         the name of the y column in the input catalogs. [default: 'y']
        :ra_col:        (alternative to x_col, y_col) the name of a right ascension column in
                        the input catalogs.  will use the wcs to find (x,y) [default: none]
        :dec_col:       (alternative to x_col, y_col) the name of a declination column in
                        the input catalogs.  will use the wcs to find (x,y) [default: none]
        :flag_col:      the name of a flag column in the input catalogs. [default: none]
                        by default, this will skip any objects with flag != 0, but see
                        skip_flag and use_flag for other possible meanings for how the
                        flag column can be used to select stars.
        :skip_flag:     the flag indicating which items to not use. [default: -1]
                        items with flag & skip_flag != 0 will be skipped.
        :use_flag:      the flag indicating which items to use. [default: none]
                        items with flag & use_flag == 0 will be skipped.
        :property_cols: a list of column names of star properties (e.g. star colors).
                        [default: none]
        :properties:    a dict of constant star properties to attach to every star from each
                        input image. the keys are property names. values are parsed using
                        the galsim config parser for each ``image_num``. plain strings that
                        begin with ``$`` or ``@`` are treated as config expressions; other
                        strings are left as literal strings. the explicit ``type``/``value``
                        dict form is also supported. [default: none]

        :sed_col:       the name of a column with per-star sed selectors. if ``sed_file_name``
                        is not provided, selectors are interpreted as sed file names. if
                        ``sed_file_name`` is provided, selectors are treated as keys in the
                        sed mapping. [default: none]
        :sed_file_name: a single sed file name to use for all stars (if ``sed_col`` is
                        not provided), or a dict / yaml file mapping ``sed_col`` selectors to
                        sed file names. [default: none]
        :sed_wave_type: wavelength unit for sed files (e.g. ``nm``, ``angstrom``).
                        [required if sed_col or sed_file_name is not none]
        :sed_flux_type: flux convention for sed files (e.g. ``flambda``, ``fnu``,
                        ``fphotons`` or units parseable by astropy.unit).
                        [required if sed_col or sed_file_name is not none]
        :sed_wave_key:  fits-table column name to use for wavelength when loading
                        seds from fits tables. [default: ``wave``]
        :sed_flux_key:  fits-table column name to use for flux when loading seds
                        from fits tables. [default: ``flux``]
        :sed_tol:       relative tolerance for reducing the number of samples of input seds
                        based on the calculated flux over the given bandpass. the seds are each
                        reduced on first access, and the reduced sed object is then shared
                        by all matching stars. values <= 0 disable this step. [default: 0]
        :sed_max_samples:
                        maximum number of samples to keep in the effective sed. if set, use a
                        greedy knot-selection algorithm to choose up to this many samples, then
                        refit their values for the best piecewise-linear approximation of the
                        original effective sed. values <= 0 disable this step. [default: 0]
        :bandpass:      galsim bandpass config dict describing the observing bandpass for the
                        input image(s). required when using ``sed_col`` or ``sed_file_name``.
                        [default: none]

        :sky_col:       The name of a column with sky values. [default: None]
        :gain_col:      The name of a column with gain values. [default: None]
        :gain_file_name: A file containing a per-pixel effective gain map. [default: None]
        :gain_ext:      The FITS extension number/name or ASDF path in the gain file,
                        or in the image file if gain_file_name is omitted.
                        [default: None, which means use the gain file's default extension]
        :sky:           The sky level to subtract from the image values. [default: None]
                        Specify only one sky source: sky, sky_col, or sky_file_name/sky_ext.
                        If none are provided, the input image will be assumed to have already
                        been sky subtracted.

                        .. note::

                            There are two special values allowed for this parameter:

                            * ``sky: median`` computes the median of the image and uses that
                              as the (constant) sky level.
                            * ``sky: sep`` uses SEP's full spatial background image, with its
                              default background settings. It requires the optional ``sep``
                              package to be installed.

                            Both estimators exclude pixels with zero weight.

        :gain:          The gain to use for adding Poisson noise to the weight map. Gain is defined
                        such that source_variance = source_signal / gain in the image's units.
                        Specify only one gain source: gain, gain_col, or gain_file_name/
                        gain_ext. The gain may also be inferred implicitly from the action of
                        remove_signal_from_weight (see below). If no gain information is provided,
                        then no additional noise will be added to account for the Poisson noise
                        from the stars' flux. [default: None]
        :satur:         The staturation level.  If any pixels for a star exceed this, then
                        the star is skipped. [default: None]
        :trust_pos:     Whether to trust the input position as essentially exact (True), or to
                        let the stars move slightly from their nominal positions (False).
                        [default: False]
        :use_partial:   Whether to use stars whose postage stamps are only partially on the
                        full image.  [default: False]
        :nstars:        Stop reading each input catalog at this many stars. This legacy option
                        is deprecated; prefer ``select.nstars``, which applies globally after the
                        selection cuts and keeps the highest-S/N stars. [default: None]
        :nproc:         How many multiprocessing processes to use for reading in data from
                        multiple files at once. [default: 1]

        :wcs:           Normally, the wcs is automatically read in when reading the image.
                        However, this parameter allows you to optionally provide a different
                        WCS.  It should be defined using the same style as a wcs object
                        in GalSim config files. [defulat: None]

        :invert_weight: Whether the weight maps are actually variance maps, and thus should be
                        inverted to make the usual inverse variance weight. [default: False]
        :remove_signal_from_weight: Whether the weight (either inverted or not) includes the
                        Poisson noise from the sources. If so, set this to true so Piff can
                        remove that noise to recover the noise from non-source contributions
                        (read noise, background, dark current, etc.). Piff will account for the
                        source Poisson noise itself using the current model at every iteration,
                        so leaving observed source Poisson noise in the input weight would be an
                        error. [default: False]

                        .. note::

                            This works better if a gain value is provided. But if one is not
                            known, Piff will try to estimate the (average) gain from the
                            data image and the given weights.

        :stamp_size:    The size of the postage stamps to use for the cutouts.  Note: some
                        stamps may be smaller than this if the star is near a chip boundary.
                        [default: 32]
        :ra, dec:       The RA, Dec of the telescope pointing. [default: None; See
                        `readMetadata` for details about how this can be specified]

    :param config:      The configuration dict used to define the above parameters.
    :param base:        The base configuration dict if needed for value parsing. [default: None]
    :param logger:      A logger object for logging debug info. [default: None]
    """
    _type_name = 'Files'
    _req_params = {
        'image_file_name': str,
        'cat_file_name': str,
    }
    _opt_params = {
        'dir': str,
        'chipnum': int,
        'x_col': str,
        'y_col': str,
        'ra_col': str,
        'dec_col': str,
        'ra_units': str,
        'dec_units': str,
        'property_cols': list,
        'properties': dict,
        'sed_col': str,
        'sed_file_name': (str, dict),
        'sed_wave_type': str,
        'sed_flux_type': str,
        'sed_wave_key': str,
        'sed_flux_key': str,
        'sed_tol': float,
        'sed_max_samples': int,
        'sky_col': str,
        'gain_col': str,
        'gain_file_name': str,
        'gain_ext': (int, str),
        'flag_col': str,
        'skip_flag': int,
        'use_flag': int,
        'image_ext': (int, str),
        'weight_ext': (int, str),
        'badpix_ext': (int, str),
        'weight_file_name': str,
        'badpix_file_name': str,
        'badpix_zeros': bool,
        'fix_ushort_badpix': bool,
        'sky_file_name': str,
        'sky_ext': (int, str),
        'cat_ext': (int, str),
        'invert_weight': bool,
        'remove_signal_from_weight': bool,
        'stamp_size': int,
        'use_partial': bool,
        'nstars': int,
    }

    def __init__(self, config, base=None, logger=None):
        logger = LoggerWrapper(logger)

        req = self._req_params
        opt = self._opt_params
        ignore = [ 'nproc', 'nimages', 'ra', 'dec', 'wcs', 'bandpass',
                   'sky', 'gain', 'satur', 'noise', 'trust_pos' ]

        # Normalize filenames and deprecated keys on copies of the caller's configuration.
        config = copy.deepcopy(config)
        base = copy.deepcopy(base) if base is not None else {'input': config}

        # Copy deprecated extension selectors to their canonical names before validation.
        for prefix in ['image', 'weight', 'badpix', 'sky', 'cat']:
            old = prefix + '_hdu'
            new = prefix + '_ext'
            if old in config:
                if new in config:
                    raise ValueError("Cannot provide both %s and %s." % (old, new))
                logger.error("%s is deprecated as of version 1.8. Use %s instead.", old, new)
                # Keep old keys in config dict in case any Current/Eval types use them.
                config[new] = config[old]
                ignore.append(old)

        # A filename and extension together select one plane; alternative sky or gain
        # sources are mutually exclusive.
        sky_sources = sum(['sky' in config, 'sky_col' in config,
                           'sky_file_name' in config or 'sky_ext' in config])
        if sky_sources > 1:
            raise ValueError("Specify only one sky source: sky, sky_col, or "
                             "sky_file_name/sky_ext.")

        gain_sources = sum(['gain' in config, 'gain_col' in config,
                            'gain_file_name' in config or 'gain_ext' in config])
        if gain_sources > 1:
            raise ValueError("Specify only one gain source: gain, gain_col, or "
                             "gain_file_name/gain_ext.")

        # Index GalSim lists and sequences by input image rather than by star.
        base['index_key'] = 'image_num'

        nimages = None
        image_list = None
        cat_list = None
        dir = None

        if 'nproc' in config:
            self.nproc = galsim.config.ParseValue(config, 'nproc', base, int)[0]

        if 'nstars' in config:
            logger.error("input.nstars is deprecated; use select.nstars for the new global "
                         "highest-S/N behavior.")

        # The number of images may be either explicit (nimages) or implicit in the length of
        # a list, which may include a glob (e.g. *.fits). If there are multiple such lists, then
        # the lengths must match.
        if 'nimages' in config:
            nimages = galsim.config.ParseValue(config, 'nimages', base, int)[0]
            if nimages < 1:
                raise ValueError('input.nimages must be >= 1')

        if 'image_ext' in config and isinstance(config['image_ext'], list):
            if len(config['image_ext']) == 0:
                raise ValueError("image_ext may not be an empty list")
            if nimages is None:
                nimages = len(config['image_ext'])
            elif nimages != len(config['image_ext']):
                raise ValueError("nimages = %s doesn't match length of image_ext list (%d)" % (
                                 nimages, len(config['image_ext'])))

        # Apply dir before expanding literal filename patterns, then remove it so later
        # parsing does not apply it again.
        if 'dir' in config:
            dir = galsim.config.ParseValue(config, 'dir', base, str)[0]
            del config['dir']

        # Preserve @/$ expressions as config dictionaries so they are evaluated per image,
        # rather than treated as literal filename patterns by glob.glob later.
        for key in ['image_file_name', 'cat_file_name', 'sky_file_name',
                    'weight_file_name', 'badpix_file_name', 'gain_file_name']:
            value = config.get(key)
            if isinstance(value, str) and value.startswith('@'):
                config[key] = {'type': 'Current', 'key': value[1:]}
            elif isinstance(value, str) and value.startswith('$'):
                config[key] = {'type': 'Eval', 'str': value[1:]}

        # Turn literal science filenames and glob patterns into ordered lists. Config
        # dictionaries remain unevaluated and require the image count to be known already.
        if 'image_file_name' not in config:
            raise TypeError('Parameter image_file_name is required')
        elif isinstance(config['image_file_name'], list):
            image_list = config['image_file_name']
            if len(image_list) == 0:
                raise ValueError("image_file_name may not be an empty list")
            if dir is not None:
                image_list = [os.path.join(dir, n) for n in image_list]
        elif isinstance(config['image_file_name'], str):
            image_file_name = config['image_file_name']
            if dir is not None:
                image_file_name = os.path.join(dir, image_file_name)
            image_list = sorted(glob.glob(image_file_name))
            if len(image_list) == 0:
                raise ValueError("No files found corresponding to "+config['image_file_name'])
        elif isinstance(config['image_file_name'], dict):
            if nimages is None:
                raise TypeError(
                    'input.nimages is required if not using a list or simple string for ' +
                    'file names')
        else:
            raise ValueError("image_file_name should be either a dict or a string")

        # Reuse a single science file when multiple images/extensions were requested;
        # otherwise its list length must agree with the requested image count.
        if image_list is not None:
            logger.debug('image_list = %s',image_list)
            if len(image_list) == 1 and nimages is not None and nimages > 1:
                logger.verbose("Using the same image file for all images")
                image_list = image_list * nimages
            elif nimages is not None and nimages != len(image_list):
                raise ValueError("nimages = %s doesn't match length of image_file_name list (%d)"%(
                        nimages, len(image_list)))
            nimages = len(image_list)
            logger.debug('nimages = %d',nimages)
            config['image_file_name'] = image_list
        logger.debug('nimages = %d',nimages)
        assert nimages is not None

        # Expand auxiliary file patterns in the same order as the science files, reusing
        # a single match for every image.
        for key in ['sky_file_name', 'weight_file_name', 'badpix_file_name', 'gain_file_name']:
            if key not in config:
                continue
            value = config[key]
            if isinstance(value, str):
                pattern = os.path.join(dir, value) if dir is not None else value
                file_list = sorted(glob.glob(pattern))
                if len(file_list) == 0:
                    raise ValueError("No files found corresponding to " + value)
                if len(file_list) == 1 and nimages > 1:
                    logger.verbose("Using the same %s for all images", key)
                    file_list = file_list * nimages
                elif len(file_list) != nimages:
                    raise ValueError("nimages = %s doesn't match length of %s list (%d)" %
                                     (nimages, key, len(file_list)))
                config[key] = file_list
            elif not isinstance(value, dict):
                raise ValueError("%s should be either a dict or a string" % key)

        # Catalog filenames also accept explicit lists, literal names, glob patterns, or
        # per-image config dictionaries. None skips catalog loading when restoring stars.
        if 'cat_file_name' not in config:
            raise TypeError('Parameter cat_file_name is required')
        elif isinstance(config['cat_file_name'], list):
            cat_list = config['cat_file_name']
            if len(cat_list) == 0:
                raise ValueError("cat_file_name may not be an empty list")
            if dir is not None:
                cat_list = [os.path.join(dir, n) for n in cat_list]
        elif isinstance(config['cat_file_name'], str):
            cat_file_name = config['cat_file_name']
            if dir is not None:
                cat_file_name = os.path.join(dir, cat_file_name)
            cat_list = sorted(glob.glob(cat_file_name))
            if len(cat_list) == 0:
                raise ValueError("No files found corresponding to "+config['cat_file_name'])
        elif not isinstance(config['cat_file_name'], (dict, type(None))):
            raise ValueError("cat_file_name should be either a dict or a string")

        # Pair catalogs with science images, or reuse a single catalog for every image.
        if cat_list is not None:
            logger.debug('cat_list = %s',cat_list)
            if len(cat_list) == 1 and nimages > 1:
                logger.verbose("Using the same catlist for all image")
                cat_list = cat_list * nimages
            elif nimages != len(cat_list):
                raise ValueError("nimages = %s doesn't match length of cat_file_name list (%d)"%(
                                 nimages, len(cat_list)))
            config['cat_file_name'] = cat_list

        self.nimages = nimages

        # Just check that optional parameters are valid without parsing them.
        # Most of the opt params will be parsed in later stages.
        galsim.config.CheckAllParams(config, req=req, opt=opt, ignore=ignore)
        self.config = galsim.config.CleanConfig(config)
        self.base = galsim.config.CleanConfig(base)
        self.metadata_list = None

    #
    # The next set of methods are intended to provide hooks for subclasses to potentially
    # change how aspects of the read are performed. Images may be constructed in memory from
    # other data in the input file, or additional operations might be performed on the data, etc.
    # Data are held in a RawImageData object, called data in all of these functions.
    #

    def load_image(self, data, logger):
        """Load the main image.

        Input data: image_num, config, base.
        This function sets: image

        The image includes a header and wcs as appropriate. It is also stored in
        base['_current_image'] for possible use by later config processing.
        """
        kwargs = self._parse_params(data, {'image_file_name': None, 'image_ext': None})
        logger.info("Reading image file %s", kwargs['image_file_name'])
        data.image = self.read_image(kwargs['image_file_name'], kwargs['image_ext'],
                                     read_header=True)
        if data.image.dtype not in [np.float32, np.float64]:
            header = data.image.header  # This doesn't get copied by view().
            data.image = data.image.view(dtype=np.float32)
            data.image.header = header
        data.base['_current_image'] = data.image

    def load_weight(self, data, logger):
        """Load an optional weight image.

        Input data: image_num, config, base.
        This function sets: weight

        This loads the raw weight image, which may or may not be in the eventual inverse
        variance format. That will be done in finalize_weight, since it may need other
        things to be loaded first. Entries that should eventually be zero weight may be
        set to zero here. The inversion will not change them.
        """
        kwargs = self._parse_params(data, {
            'weight_file_name': None,
            'weight_ext': None,
            'image_file_name': None,
        })
        if kwargs['weight_file_name'] is not None or kwargs['weight_ext'] is not None:
            file_name = kwargs['weight_file_name'] or kwargs['image_file_name']
            logger.verbose("Reading weight image from %s, extension %s.",
                           file_name, kwargs['weight_ext'])
            data.weight = self.read_image(file_name, kwargs['weight_ext'], read_header=False)
            if np.all(data.weight.array == 0):
                logger.error("According to the weight mask in %s, all pixels have zero weight!",
                             kwargs['image_file_name'])

    def load_gain(self, data, logger):
        """Load an optional per-pixel gain map.

        Input data: image_num, config, base.
        This function sets: gain_image
        """
        kwargs = self._parse_params(data, {
            'gain_file_name': None,
            'gain_ext': None,
            'image_file_name': None,
        })
        if kwargs['gain_file_name'] is not None or kwargs['gain_ext'] is not None:
            file_name = kwargs['gain_file_name'] or kwargs['image_file_name']
            logger.verbose("Reading gain image from %s, extension %s.",
                           file_name, kwargs['gain_ext'])
            data.gain_image = self.read_image(file_name, kwargs['gain_ext'],
                                            read_header=False)
            if data.gain_image.bounds != data.image.bounds:
                raise ValueError('Gain image bounds do not match science image bounds')

    def build_mask(self, data, logger):
        """Build a mask of which pixels Piff should not use.

        Input data: image_num, image, weight, gain_image, config, base.
        This function sets: mask, and may update image.

        The mask comes from several sources in the default implementation:
        1. A badpix image if one is specified.
        2. Any weight=0 pixels (regardless of whether this is variance or inverse variance yet).
        3. Non-finite pixels in the image, weight or gain.
        4. Pixels with negative weight or gain.
        5. If badpix_zeros, any data pixels with value 0.

        The final mask is True for any masked pixels.
        """
        data.mask = np.zeros(data.image.array.shape, dtype=bool)
        kwargs = self._parse_params(data, {
            'badpix_file_name': None,
            'badpix_ext': None,
            'badpix_zeros': False,
            'fix_ushort_badpix': False,
            'image_file_name': None,
        })
        if kwargs['badpix_file_name'] is not None or kwargs['badpix_ext'] is not None:
            file_name = kwargs['badpix_file_name'] or kwargs['image_file_name']
            logger.verbose("Reading badpix image from %s, extension %s.",
                           file_name, kwargs['badpix_ext'])
            badpix = self.read_image(file_name, kwargs['badpix_ext'], read_header=False)

            # Correct legacy unsigned-integer offsets if requested.
            if kwargs['fix_ushort_badpix']:
                if np.any(badpix.array > 32767):
                    logger.debug('min(badpix) = %s',np.min(badpix.array))
                    logger.debug('max(badpix) = %s',np.max(badpix.array))
                    logger.debug("subtracting 32768 from all values in badpix image")
                    badpix -= 32768
                if np.any(badpix.array < -32767):
                    logger.debug('min(badpix) = %s',np.min(badpix.array))
                    logger.debug('max(badpix) = %s',np.max(badpix.array))
                    logger.debug("adding 32768 to all values in badpix image")
                    badpix += 32768
            if np.all(badpix.array != 0):
                logger.error("According to the bad pixel array in %s, all pixels are masked!",
                             kwargs['image_file_name'])
            data.mask |= badpix.array != 0

        if data.weight is not None:
            weight = data.weight.array
            if np.any(weight < 0):
                logger.warning("Warning: weight map has invalid negative-valued pixels. "+
                               "Taking them to be 0.0")
            nonfinite = ~np.isfinite(weight)
            if np.any(nonfinite):
                logger.warning("Warning: weight map has non-finite pixels. "
                               "Taking them to be 0.0")
            data.mask |= nonfinite | (weight <= 0)
        if data.gain_image is not None:
            gain = data.gain_image.array
            nonfinite = ~np.isfinite(gain)
            if np.any(nonfinite):
                logger.warning("Warning: gain image has non-finite pixels. Masking them.")
            nonpositive = gain <= 0
            if np.any(nonpositive):
                logger.warning("Warning: gain image has non-positive pixels. Masking them.")
            data.mask |= nonfinite | nonpositive
        if kwargs['badpix_zeros']:
            data.mask |= data.image.array == 0

        # Clear nonfinite science so downstream weight * image cannot produce NaN * 0.
        nonfinite = ~np.isfinite(data.image.array)
        if np.any(nonfinite):
            logger.warning("Warning: image has non-finite pixels. "
                           "Masking and setting them to 0.0")
        data.mask |= nonfinite
        data.image.array[nonfinite] = 0

    def subtract_sky(self, data, logger):
        """Load or estimate the image-level sky and subtract it from the science image.

        Input data: image_num, image, mask, config, base.
        This function sets: sky, and updates: image

        The sky is retained for later calculations, which may use it.

        This doesn't handle any catalog-level sky values. Those are handled by `load_catalog`.
        """
        if 'sky' in data.config:
            data.sky = galsim.config.ParseValue(data.config, 'sky', data.base, (float, str))[0]
        kwargs = self._parse_params(data, {
            'sky_file_name': None,
            'sky_ext': None,
            'image_file_name': None,
        })
        if kwargs['sky_file_name'] is not None or kwargs['sky_ext'] is not None:
            file_name = kwargs['sky_file_name'] or kwargs['image_file_name']
            logger.verbose("Reading sky image from %s, extension %s.",
                           file_name, kwargs['sky_ext'])
            data.sky = self.read_image(file_name, kwargs['sky_ext'],
                                      read_header=False)
        if isinstance(data.sky, str):
            if data.sky == 'median':
                data.sky = np.median(data.image.array[~data.mask])
            elif data.sky == 'sep':
                import sep
                # SEP requires contiguous arrays with native byte order.
                array = np.ascontiguousarray(
                    data.image.array, dtype=data.image.array.dtype.newbyteorder('='))
                background = sep.Background(array, mask=data.mask)
                data.sky = galsim.Image(background.back(), bounds=data.image.bounds,
                                        wcs=data.image.wcs)
            else:
                data.sky = self._handle_depr_image_value(
                    data.sky, data.image.header, 'sky', logger)

        if data.sky is not None:
            data.image -= data.sky

    def finalize_weight(self, data, logger):
        """Finish building the weight image based on all the ancillary data.

        Input data: image_num, image, weight, gain_image, mask, image_pos, extra_props,
                    config, base.
        This function sets or updates: weight, and may update mask, extra_props

        This finishes the calculation of the weight image, converting it to an inverse
        variance weight if necessary and removing any contribution from the signal's
        Poisson variance.
        """
        if data.weight is not None:
            data.weight._piff_default_weight = False
        else:
            noise = None
            if 'noise' in data.config:
                noise = galsim.config.ParseValue(data.config, 'noise', data.base, (float, str))[0]
            if noise is not None:
                noise = self._handle_depr_image_value(noise, data.image.header, 'noise', logger)
                logger.debug("Making uniform weight image based on noise variance = %f", noise)
                data.weight = galsim.ImageF(data.image.bounds, init_value=1./noise)
                data.weight._piff_default_weight = False
            else:
                logger.debug("Making trivial (wt==1) weight image")
                data.weight = galsim.ImageF(data.image.bounds, init_value=1)
                data.weight._piff_default_weight = True
        data.weight.array[data.mask] = 0
        params = self._parse_params(data, {
            'invert_weight': False,
            'remove_signal_from_weight': False,
        })
        if params['invert_weight']:
            data.weight._invertSelf()
        if params['remove_signal_from_weight']:
            nmasked = np.count_nonzero(data.weight.array == 0)
            # Image-level backgrounds have already been subtracted. Catalog backgrounds
            # currently contribute their mean to the whole-image variance calculation.
            if 'sky' in data.extra_props:
                signal = data.image - np.mean(data.extra_props['sky'])
            else:
                signal = data.image
            if data.gain_image is not None:
                data.weight, _ = self._removeSignalFromWeight(signal, data.weight,
                                                            gain=data.gain_image)
            elif 'gain' in data.extra_props:
                gain = np.mean(data.extra_props['gain'])
                data.weight, _ = self._removeSignalFromWeight(signal, data.weight, gain=gain)
            else:
                data.weight, gain = self._removeSignalFromWeight(signal, data.weight)
                data.extra_props['gain'] = np.array([gain] * len(data.image_pos), dtype=float)
                logger.info("Empirically determined gain = %f", gain)
            logger.verbose("Removed signal from weight image.")
            data.mask |= data.weight.array == 0
            added_masks = np.count_nonzero(data.weight.array == 0) - nmasked
            if added_masks:
                logger.warning("Masked %d pixels with invalid variance after removing source noise",
                               added_masks)

    def update_wcs(self, data, logger):
        """Update the image's wcs if necessary.

        Input data: image
        This function updates: image.wcs

        The image's wcs at this point is the WCS from the input file. If there is a different
        WCS specified by the config, it has already been processed and stored in
        self.wcs_list. This step updates the image to have the correct WCS attached.
        """
        data.image.wcs = self.wcs_list[data.image_num]

    def load_catalog(self, data, logger):
        """Load star positions and properties.

        Input data: image_num, image, config, base.
        This function sets: image_pos, extra_props
        """
        params = self._parse_params(data, {
            'cat_file_name': None,
            'cat_ext': None,
            'x_col': 'x',
            'y_col': 'y',
            'ra_col': None,
            'dec_col': None,
            'ra_units': 'deg',
            'dec_units': 'deg',
            'flag_col': None,
            'skip_flag': -1,
            'use_flag': None,
            'property_cols': None,
            'properties': None,
            'sed_col': None,
            'sed_file_name': None,
            'sed_wave_type': None,
            'sed_flux_type': None,
            'sed_wave_key': 'WAVE',
            'sed_flux_key': 'FLUX',
            'sed_tol': 0.0,
            'sed_max_samples': 0,
            'sky_col': None,
            'gain_col': None,
            'nstars': None,
            'stamp_size': 32,
        })
        if self.bandpass is None and (params['sed_col'] is not None or
                                      params['sed_file_name'] is not None):
            raise ValueError("bandpass is required when using sed_col or sed_file_name")
        image = data.image
        config = data.config
        base = data.base

        if params['cat_file_name'] is None:
            # This is possible e.g. when loading images into an existing list of star instances.
            logger.info("Not reading input catalog.")
            data.image_pos = data.extra_props = None
            return

        # Read in the star catalog
        logger.info("Reading input catalog %s.",params['cat_file_name'])
        cat = self.read_catalog(params['cat_file_name'], params['cat_ext'])

        gain = None
        if 'gain' in config:
            gain = galsim.config.ParseValue(config, 'gain', base, (float, str))[0]

        satur = None
        if 'satur' in config:
            satur = galsim.config.ParseValue(config, 'satur', base, (float, str))[0]

        trust_pos = None
        if 'trust_pos' in config:
            trust_pos = galsim.config.ParseValue(config, 'trust_pos', base, bool)[0]

        if params['flag_col'] is not None:
            if params['flag_col'] not in cat.dtype.names:
                raise ValueError("flag_col = %s is not a column in %s" %
                                 (params['flag_col'], params['cat_file_name']))
            col = cat[params['flag_col']]
            if len(col.shape) == 2:
                logger.info("Flag col (%s) is multidimensional.  Treating as an array of bool",
                            params['flag_col'])
            if params['use_flag'] is not None:
                # Remove any objects with flag & use_flag == 0
                mask = self._flag_select(col, params['use_flag']) == 0
                logger.verbose("Removing objects with flag (col %s) & %d == 0",
                               params['flag_col'], params['use_flag'])
                if params['skip_flag'] != -1:
                    mask |= self._flag_select(col, params['skip_flag']) != 0
                    logger.verbose("Removing objects with flag (col %s) & %d != 0",
                                   params['flag_col'], params['skip_flag'])
            else:
                # Remove any objects with flag & skip_flag != 0
                mask = self._flag_select(col, params['skip_flag']) != 0
                if params['skip_flag'] == -1:
                    logger.verbose("Removing objects with flag (col %s) != 0",params['flag_col'])
                else:
                    logger.verbose("Removing objects with flag (col %s) & %d != 0",
                                   params['flag_col'], params['skip_flag'])
            cat = cat[mask == 0]

        # Limit to nstars objects
        if params['nstars'] is not None and params['nstars'] < len(cat):
            logger.verbose("Limiting to %d objects for %s",
                           params['nstars'], params['cat_file_name'])
            cat = cat[:params['nstars']]

        # Make the list of positions:
        if params['ra_col'] is not None or params['dec_col'] is not None:
            if params['ra_col'] is None or params['dec_col'] is None:
                raise ValueError("ra_col and dec_col are both required if one is provided.")
            if params['ra_col'] not in cat.dtype.names:
                raise ValueError("ra_col = %s is not a column in %s" %
                                 (params['ra_col'], params['cat_file_name']))
            if params['dec_col'] not in cat.dtype.names:
                raise ValueError("dec_col = %s is not a column in %s" %
                                 (params['dec_col'], params['cat_file_name']))
            logger.debug("Starting to make a list of positions from ra, dec")
            ra_values = cat[params['ra_col']]
            dec_values = cat[params['dec_col']]
            params['ra_units'] = galsim.AngleUnit.from_name(params['ra_units'])
            params['dec_units'] = galsim.AngleUnit.from_name(params['dec_units'])
            ra = ra_values * params['ra_units']
            dec = dec_values * params['dec_units']
            logger.debug("Initially %d positions",len(ra))

            # First limit to only those that could possibly be on the image by checking the
            # min/max ra and dec from the image corners.
            cen = image.wcs.toWorld(image.center)
            logger.debug("Center at %s",cen)
            x_corners = [image.xmin, image.xmin, image.xmax, image.xmax]
            y_corners = [image.ymin, image.ymax, image.ymax, image.ymin]
            corners = [image.wcs.toWorld(galsim.PositionD(x,y))
                       for (x,y) in zip(x_corners, y_corners)]
            logger.debug("Corners at %s",corners)
            min_ra = np.min([c.ra.wrap(cen.ra) for c in corners])
            max_ra = np.max([c.ra.wrap(cen.ra) for c in corners])
            min_dec = np.min([c.dec.wrap(cen.dec) for c in corners])
            max_dec = np.max([c.dec.wrap(cen.dec) for c in corners])
            logger.debug("RA range = %s .. %s",min_ra,max_ra)
            logger.debug("Dec range = %s .. %s",min_dec,max_dec)
            use = [(r.wrap(cen.ra) > min_ra) & (r.wrap(cen.ra) < max_ra) &
                   (d.wrap(cen.dec) > min_dec) & (d.wrap(cen.dec) < max_dec)
                   for r,d in zip(ra,dec)]
            ra = ra[use]
            dec = dec[use]
            logger.debug("After limiting to image ra,dec range, len = %s",len(ra))

            # Now convert to x,y
            def safe_to_image(wcs, ra, dec):
                try:
                    return wcs.toImage(galsim.CelestialCoord(ra, dec))
                except galsim.GalSimError:  # pragma: no cover
                    # If the ra,dec is way off the image, this might fail to converge.
                    # In this case return None, which we can get rid of simply.
                    return None
            image_pos = [ safe_to_image(image.wcs,r,d) for r,d in zip(ra, dec) ]
            image_pos = [ pos for pos in image_pos if pos is not None ]
            logger.debug("Resulting image_pos list has %s positions",len(image_pos))
        else:
            if params['x_col'] not in cat.dtype.names:
                raise ValueError("x_col = %s is not a column in %s" %
                                 (params['x_col'], params['cat_file_name']))
            if params['y_col'] not in cat.dtype.names:
                raise ValueError("y_col = %s is not a column in %s" %
                                 (params['y_col'], params['cat_file_name']))
            x_values = cat[params['x_col']]
            y_values = cat[params['y_col']]
            logger.debug("Initially %d positions",len(x_values))
            image_pos = [ galsim.PositionD(x,y) for x,y in zip(x_values, y_values) ]

        # Check for objects well off the edge.  We won't use them.
        big_bounds = image.bounds.expand(params['stamp_size'])
        image_pos = [ pos for pos in image_pos if big_bounds.includes(pos) ]
        logger.debug("After remove those that are off the image, len = %s",len(image_pos))

        # Make a dictionary of the star properties:
        extra_props = {}
        if params['property_cols'] is not None:
            logger.debug('property_cols = %s'%params['property_cols'])
            for col_name in params['property_cols']:
                if col_name not in cat.dtype.names:
                    raise ValueError("Entry in property_cols = " +
                                     "%s is not a column in %s"%(col_name,params['cat_file_name']))
                extra_props[col_name] = cat[col_name]

        if params['sed_col'] is not None:
            if params['sed_col'] not in cat.dtype.names:
                raise ValueError("sed_col = %s is not a column in %s" %
                                 (params['sed_col'], params['cat_file_name']))
            sed_map = None
            if params['sed_file_name'] is not None:
                sed_map = self._read_sed_mapping(params['sed_file_name'], params['cat_file_name'])
            sed_file_name_values = []
            for sed_label in cat[params['sed_col']]:
                if sed_map is not None:
                    sed_label = os.fsdecode(sed_label).strip()
                    if sed_label not in sed_map:
                        raise ValueError(
                            "sed_col value %r is not in sed_file_name mapping" % sed_label
                        )
                    params['sed_file_name'] = sed_map[sed_label]
                else:
                    params['sed_file_name'] = sed_label
                params['sed_file_name'] = self._resolve_sed_file_name(
                    params['sed_file_name'], params['cat_file_name'])
                sed_file_name_values.append(params['sed_file_name'])
            extra_props['sed_file_name'] = np.array(sed_file_name_values, dtype=object)
            extra_props['sed_wave_type'] = np.array(
                [params['sed_wave_type']] * len(cat), dtype=object)
            extra_props['sed_flux_type'] = np.array(
                [params['sed_flux_type']] * len(cat), dtype=object)
            extra_props['sed_wave_key'] = np.array(
                [params['sed_wave_key']] * len(cat), dtype=object)
            extra_props['sed_flux_key'] = np.array(
                [params['sed_flux_key']] * len(cat), dtype=object)
            extra_props['sed_tol'] = np.full(len(cat), params['sed_tol'], dtype=float)
            extra_props['sed_max_samples'] = np.full(len(cat), params['sed_max_samples'], dtype=int)
        elif params['sed_file_name'] is not None:
            params['sed_file_name'] = self._resolve_sed_file_name(
                params['sed_file_name'], params['cat_file_name'])
            extra_props['sed_file_name'] = np.array(
                [params['sed_file_name']] * len(cat), dtype=object)
            extra_props['sed_wave_type'] = np.array(
                [params['sed_wave_type']] * len(cat), dtype=object)
            extra_props['sed_flux_type'] = np.array(
                [params['sed_flux_type']] * len(cat), dtype=object)
            extra_props['sed_wave_key'] = np.array(
                [params['sed_wave_key']] * len(cat), dtype=object)
            extra_props['sed_flux_key'] = np.array(
                [params['sed_flux_key']] * len(cat), dtype=object)
            extra_props['sed_tol'] = np.full(len(cat), params['sed_tol'], dtype=float)
            extra_props['sed_max_samples'] = np.full(len(cat), params['sed_max_samples'], dtype=int)

        # If we used a flag column, keep it as a property.
        if params['flag_col'] is not None:
            extra_props[params['flag_col']] = cat[params['flag_col']]

        # Make the list of sky values:
        if params['sky_col'] is not None:
            if params['sky_col'] not in cat.dtype.names:
                raise ValueError("sky_col = %s is not a column in %s" %
                                 (params['sky_col'], params['cat_file_name']))
            extra_props['sky'] = np.float32(cat[params['sky_col']])

        # Make the list of gain values:
        if params['gain_col'] is not None:
            if params['gain_col'] not in cat.dtype.names:
                raise ValueError("gain_col = %s is not a column in %s" %
                                 (params['gain_col'], params['cat_file_name']))
            extra_props['gain'] = cat[params['gain_col']]
        elif gain is not None:
            gain = self._handle_depr_image_value(gain, image.header, 'gain', logger)
            extra_props['gain'] = np.array([gain]*len(cat), dtype=float)

        # Get the saturation level
        if satur is not None:
            satur = self._handle_depr_image_value(satur, image.header, 'satur', logger)
            logger.debug("Using saturation value: %s",satur)
            extra_props['satur'] = np.array([satur]*len(cat), dtype=float)

        # Check whether we should trust the positions
        if trust_pos is not None:
            extra_props['trust_pos'] = np.array([trust_pos]*len(cat), dtype=bool)

        if params['properties'] is not None:
            logger.debug('properties = %s', params['properties'])
            extra_props.update(self._evaluate_properties(
                params['properties'], extra_props, len(cat), base, logger))

        if 'sky' in extra_props and data.sky is not None:
            raise ValueError("A sky property cannot coexist with an image-level sky source.")

        data.image_pos = image_pos
        data.extra_props = extra_props

    def finish_star(self, data, logger, star):
        """Finish one star cutout using per-star properties and whole-image calibration.

        For this function, data is provided just for reference in case subclasses want
        information stored there. The base implementation doesn't use it.

        This takes in a star instance and performs any adjustments required. It returns the
        updated star.

        The base class implementation just subtracts a catalog-level sky from the image,
        since that version of providing the sky information couldn't be handled in
        subtract_sky.

        :param data:        A RawImageData object with the current data.
        :param logger:      Logger for input diagnostics.
        :param star:        Star instance to finish processing.

        :returns:           The updated star.
        """
        if 'sky' in star.data.properties:
            sky = star.data.properties['sky']
            logger.debug("Subtracting off sky = %f", sky)
            logger.debug("Median pixel value = %f", np.median(star.image.array))
            image = star.image - sky
            return Star(star.data.withNew(image=image), star.fit)
        else:
            return star

    def read_metadata(self, file_name, ext):
        """Read bounds, wcs, and header associated with an image on disk without loading the
        pixel array.

        :param file_name:   Science image filename.
        :param ext:         FITS extension or dot-separated ASDF path.

        :returns:           An `ImageMetadata` instance with the meta-data.
        """
        if file_name.endswith('.asdf'):
            import asdf
            if not isinstance(ext, str):
                raise ValueError('ext must be a dot-separated path when used with ASDF files')
            with asdf.open(file_name) as af:
                ny, nx = get_asdf_value(af.tree, ext).shape
            return ImageMetadata(galsim.BoundsI(1, nx, 1, ny), galsim.PixelScale(1), None)
        else:
            if isinstance(ext, str):
                from astropy.io import fits
                with fits.open(file_name) as hdus:
                    header = galsim.FitsHeader(header=hdus[ext].header.copy())
            else:
                header = galsim.FitsHeader(file_name=file_name, hdu=ext)
            wcs, origin = galsim.wcs.readFromFitsHeader(header)
            bounds = galsim.BoundsI(origin.x, origin.x + header['NAXIS1'] - 1,
                                    origin.y, origin.y + header['NAXIS2'] - 1)
            return ImageMetadata(bounds, wcs, header)

    def read_image(self, file_name, ext, *, read_header):
        """Read one stored image plane for science or an auxiliary calibration.

        Reads from either a FITS hdu or an asdf file, for which ext mean the path to the
        data array. For FITS files, it attaches the header if read_header=True. The
        wcs is automatically set for FITS files, but not for ASDF files for which there is
        no standard location to grab the wcs, so that must be specified separately.

        :param file_name:   Filename of the image plane.
        :param ext:         FITS extension number/name, None for the FITS default, or a
                            dot-separated ASDF array path.
        :param read_header: Whether to include the header attribute on the returned image.

        :returns:           The loaded GalSim image.
        """
        if file_name.endswith('.asdf'):
            return read_asdf_image(file_name, ext, read_header=read_header)
        elif isinstance(ext, str):
            # Apparently we never enabled named hdu access in GalSim, so as of GalSim 2.8,
            # it needs a number. So if we get a string, use astropy.io.fits directly to
            # get the correct hdu.
            from astropy.io import fits
            with fits.open(file_name) as hdus:
                return galsim.fits.read(hdu_list=hdus[ext], read_header=read_header)
        else:
            return galsim.fits.read(file_name, hdu=ext, read_header=read_header)

    def read_catalog(self, cat_file_name, cat_ext):
        """Read an unprocessed catalog table for the standard catalog-processing stage.

        This can read either a FITS binary table or a parquet file. It returns a
        numpy structured array with the loaded data.

        :param cat_file_name:   Filename of the catalog.
        :param cat_ext:         FITS table extension selector, or None for the default.
                                Ignored for Parquet files.

        :returns:               Structured array of catalog rows.
        """
        if cat_file_name.endswith('.fits'):
            return self._read_fits_cat(cat_file_name, cat_ext)
        elif cat_file_name.endswith('.parquet'):
            return self._read_parquet_cat(cat_file_name)
        else:
            # For other endings, try parquet then fits so error message is for fits.
            try:
                return self._read_parquet_cat(cat_file_name)
            except Exception:
                return self._read_fits_cat(cat_file_name, cat_ext)

    # User-facing methods.

    def readMetadata(self, logger=None):
        """Read image metadata and set the common pointing before processing stars.

        Reading images/stars or querying metadata calls this automatically on first use.
        It may also be called explicitly to read or refresh metadata beforehand.

        It only reads meta-data such as the image bounds, WCS, and headers.
        No pixel data is loaded here. An explicit input.wcs overrides the stored WCS.

        :param logger:  Optional logger for input diagnostics. [default: None]
        """
        logger = LoggerWrapper(logger)
        self.metadata_list = []
        self.wcs_list = []
        self.center_list = []
        self.chipnums = []
        self.image_file_name = []
        self.cat_file_name = []
        for image_num in range(self.nimages):
            data = RawImageData(image_num=image_num, config=self.config, base=self.base)
            if image_num == 0:
                first_data = data
            params = self._parse_params(data, {'image_file_name': None, 'image_ext': None})
            file_name = params['image_file_name']
            logger.info("Reading metadata from image file %s", file_name)
            metadata = self.read_metadata(file_name, params['image_ext'])
            self.metadata_list.append(metadata)
            self.center_list.append(metadata.bounds.true_center)
            self.image_file_name.append(file_name)
            data.base['_current_image'] = metadata
            params = self._parse_params(data, {'chipnum': image_num, 'cat_file_name': None})
            self.chipnums.append(params['chipnum'])
            self.cat_file_name.append(params['cat_file_name'])
            if 'wcs' in data.config:
                logger.info("Using custom wcs from config for %s", file_name)
                wcs = galsim.config.BuildWCS(data.config, 'wcs', data.base, logger)
            else:
                wcs = metadata.wcs
            self.wcs_list.append(wcs)
            if 'bandpass' in data.config:
                self.bandpass = galsim.config.BuildBandpass(
                    data.config, 'bandpass', data.base, logger)[0]
            else:
                self.bandpass = None

        # Resolve configured coordinates once using the first image's metadata.
        config, base = first_data.config, first_data.base
        ra = None
        if config.get('ra') is not None:
            ra = galsim.config.ParseValue(config, 'ra', base, None)[0]
        dec = None
        if config.get('dec') is not None:
            dec = galsim.config.ParseValue(config, 'dec', base, None)[0]
        if (ra is None) != (dec is None):
            raise ValueError("Only one of ra, dec was specified")

        if ra is None:
            if self.wcs_list[0].isCelestial():
                if self.nimages == 1:
                    # Here we can just use the image center.
                    wcs = self.wcs_list[0]
                    center = self.center_list[0]
                    self.pointing = wcs.toWorld(center)
                    logger.verbose("Setting pointing to image center: %.3f h, %.3f d",
                                   self.pointing.ra / galsim.hours,
                                   self.pointing.dec / galsim.degrees)
                else:
                    # Use the mean of all the image centers
                    plist = [wcs.toWorld(center)
                             for wcs,center in zip(self.wcs_list,self.center_list)]
                    # Do this in x,y,z coords, not ra, dec so we don't mess up near ra=0.
                    xlist, ylist, zlist = zip(*[p.get_xyz() for p in plist])
                    x = np.mean(xlist)
                    y = np.mean(ylist)
                    z = np.mean(zlist)
                    self.pointing = galsim.CelestialCoord.from_xyz(x,y,z)
                    logger.verbose("Setting pointing to mean of image centers: %.3f h, %.3f d",
                                   self.pointing.ra / galsim.hours,
                                   self.pointing.dec / galsim.degrees)
            else:
                self.pointing = None
        else:
            # Deprecated bare keys refer to the primary FITS header; resolve them before
            # interpreting the resulting numerical or sexagesimal coordinates.
            if isinstance(ra, str) and isinstance(dec, str) and not (':' in ra and ':' in dec):
                file_name = self.image_file_name[0]
                if self.nimages == 1:
                    logger.verbose("Setting pointing from keywords %s, %s", ra, dec)
                else:
                    logger.verbose("Setting pointing from keywords %s, %s in %s",
                                   ra, dec, file_name)
                hdu = 1 if file_name.endswith('.fz') else 0
                import fitsio
                header = fitsio.read_header(file_name, ext=hdu)
                ra = self._handle_depr_image_value(ra, header, 'ra', logger)
                dec = self._handle_depr_image_value(dec, header, 'dec', logger)

            if type(ra) in [float, int]:
                ra = float(ra) * galsim.hours
                dec = float(dec) * galsim.degrees
            elif str(ra) != ra:
                raise ValueError("Unable to parse input ra: %s"%ra)
            elif str(dec) != dec:
                raise ValueError("Unable to parse input dec: %s"%dec)
            elif ':' in ra and ':' in dec:
                ra = galsim.Angle.from_hms(ra)
                dec = galsim.Angle.from_dms(dec)
            else:
                raise ValueError("Unable to parse input ra, dec: %s, %s" % (ra, dec))
            self.pointing = galsim.CelestialCoord(ra, dec)
            logger.verbose("Setting pointing to: %.3f h, %.3f d",
                           self.pointing.ra / galsim.hours,
                           self.pointing.dec / galsim.degrees)

    def getRawImageData(self, image_num, logger=None):
        """Read an image and catalog.

        :param image_num:   The index of the image to read.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: A `RawImageData` dataclass with fields image, weight, image_pos,
                  extra_props, and gain_image. Fields can be accessed by name or unpacked
                  in that order.
        """
        if self.metadata_list is None:
            self.readMetadata(logger)
        data = RawImageData(image_num=image_num, config=self.config, base=self.base)
        logger = LoggerWrapper(logger)
        self.load_image(data, logger)
        self.load_weight(data, logger)
        self.load_gain(data, logger)
        self.build_mask(data, logger)
        self.subtract_sky(data, logger)
        self.update_wcs(data, logger)
        self.load_catalog(data, logger)
        self.finalize_weight(data, logger)
        return data

    def load_images(self, stars, logger=None):
        """Load the image data into a list of Stars.

        We don't store the image data for Stars when we write them to a file, since that
        would take up a lot of space and is usually not desired.  However, we do store the
        bounds in the original image where the star was cutout, so if you want to load back in
        the original data from the image file(s), you can do so with this function.

        :param stars:           A list of Star instances.
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: a new list of Stars with the images information loaded.
        """
        logger = LoggerWrapper(logger)
        if self.metadata_list is None:
            self.readMetadata(logger)

        image_data = {}
        for image_num in range(self.nimages):
            image_data[self.chipnums[image_num]] = self.getRawImageData(image_num, logger)

        loaded_stars = []
        for star in stars:
            chipnum = star['chipnum']
            full_data = image_data[chipnum]
            image = full_data.image[star.data.image.bounds].copy()
            weight = full_data.weight[star.data.weight.bounds].copy()
            gain_image = full_data.gain_image
            if gain_image is not None:
                gain_image = gain_image[image.bounds].copy()
            data = star.data.withNew(image=image, weight=weight, orig_weight=weight,
                                     gain_image=gain_image)
            loaded_star = Star(data=data, fit=star.fit)
            loaded_stars.append(self.finish_star(full_data, logger, loaded_star))

        return loaded_stars

    # Implementation details.

    def _parse_params(self, data, params):
        """Parse the given params, taking the default value if not in data.config.

        The input params should be a dict with key: default_value for each parameter
        to be parsed. The return will be the same structure, but with the parsed value
        replacing the default if the paramter is given in the config dict.
        """
        types = self._req_params | self._opt_params
        return {
            key: galsim.config.ParseValue(data.config, key, data.base, types[key])[0]
            if key in data.config else default
            for key, default in params.items()
        }

    def _makeStarsFromImage(self, image_num, logger):
        """Make "stars" from a single input image
        """
        full_data = self.getRawImageData(image_num, logger)
        chipnum = self.chipnums[image_num]
        params = self._parse_params(full_data, {'stamp_size': 32, 'use_partial': False})
        stamp_size = params['stamp_size']
        image = full_data.image
        wt = full_data.weight
        image_pos = full_data.image_pos
        extra_props = full_data.extra_props
        gain_image = full_data.gain_image
        logger.verbose("Processing catalog %s with %d objects",chipnum,len(image_pos))

        objects = []

        # Make a dict of the types to use for the extra_props, so we can preserve the type
        # if we need to output them (e.g. in ShapeCatalog).
        prop_types = {key: extra_props[key].dtype.type for key in extra_props}

        for k in range(len(image_pos)):
            x = image_pos[k].x
            y = image_pos[k].y
            icen = int(x+0.5)
            jcen = int(y+0.5)
            half_size = stamp_size // 2
            bounds = galsim.BoundsI(icen+half_size-stamp_size+1, icen+half_size,
                                    jcen+half_size-stamp_size+1, jcen+half_size)
            if not image.bounds.includes(bounds):
                bounds = bounds & image.bounds
                if not bounds.isDefined():
                    logger.warning("Object at position %f,%f is off the edge of the image.", x, y)
                    logger.warning("Skipping this object.")
                    continue
                if params['use_partial']:
                    logger.verbose("Object at position %f,%f overlaps the edge of the image.  "
                                "Using smaller than the full stamp size: %s", x, y, bounds)
                else:
                    logger.info("Object at position %f,%f overlaps the edge of the image.", x, y)
                    logger.info("Skipping this object.")
                    continue
            stamp = image[bounds].copy()
            wt_stamp = wt[bounds].copy()
            props = { 'chipnum' : chipnum }

            # if an object is totally masked, then don't add it!
            if np.all(wt_stamp.array == 0):
                logger.warning("Object at position %f,%f is completely masked.", x, y)
                logger.warning("Skipping this object.")
                continue

            # Add this object's entry in each list of the extra_props dictionary
            # to the StarData properties dictionary
            for key in extra_props:
                logger.debug("Assigning {} value = {}".format(key, extra_props[key][k]))
                props[key] = extra_props[key][k]

            # If any pixels are saturated, skip it.
            if 'satur' in props:
                max_val = np.max(stamp.array)
                if max_val > props['satur']:
                    logger.info("Object at position %f,%f has saturated pixels.", x, y)
                    logger.info("Maximum value is %f.", max_val)
                    logger.info("Skipping this object.")
                    continue

            pos = galsim.PositionD(x,y)
            gain_stamp = gain_image[bounds].copy() if gain_image is not None else None
            data = StarData(stamp, pos, weight=wt_stamp, pointing=self.pointing,
                            bandpass=self.bandpass,
                            properties=props, property_types=prop_types,
                            gain_image=gain_stamp)
            # Attach an attribute so we can tell later whether the weight is real or
            # if it's just the default w=1.0.
            data._piff_default_weight = getattr(wt, '_piff_default_weight', False)
            star = Star(data, None)
            star = self.finish_star(full_data, logger, star)

            objects.append(star)

        return objects

    @staticmethod
    def _handle_depr_image_value(value, header, name, logger):
        # gain, satur, noise, sky, ra, and dec used to allow bare FITS header keys.
        # Now this functionality should use the ImageHeaderValue type to be more explicit.
        # The old behavior still works, but is deprecated.
        if isinstance(value, str) and value[0] not in '@$':
            logger.error("WARNING: Using a bare FITS header key for %s is deprecated. "
                         "Use {type: ImageHeaderValue, key: %s} instead.", name, value)
            value = header[value]
            # Pointing values may be sexagesimal strings, parsed by readMetadata.
            if name not in ('ra', 'dec'):
                value = float(value)
        return value

    @staticmethod
    def _removeSignalFromWeight(image, weight, gain=None):
        """Remove the image signal from the weight map.

        :param image:   The image to use as the signal
        :param weight:  The weight image.
        :param gain:    Optionally, the gain to use as the proportionality relation. May be
                        either a scalar or an image. If gain is None, then it will be estimated
                        automatically and returned. [default: None]

        :returns: newweight, gain
        """
        use = (weight.array != 0.) & np.isfinite(image.array)
        signal = image.array[use]
        variance = 1./weight.array[use]

        if gain is None:
            fit = np.polyfit(signal, variance, deg=1)
            gain = 1./fit[0]  # fit is [ 1/gain, sky_var ]

        gain_values = gain.array[use] if isinstance(gain, galsim.Image) else gain
        variance -= signal / gain_values
        variance[~np.isfinite(variance) | (variance <= 0)] = 0

        newweight = weight.copy()
        newweight.array[use] = variance
        newweight._invertSelf()
        return newweight, gain

    @staticmethod
    def _flag_select(col, flag):
        if len(col.shape) == 1:
            # Then just treat this as a straightforward bitmask.
            return col & flag
        else:
            # Then treat this as an array of bools rather than a bitmask
            mask = np.zeros(col.shape[0], dtype=bool)
            for bit in range(col.shape[1]):  # pragma: no branch
                if flag % 2 == 1:
                    mask |= col[:,bit]
                flag = flag // 2
                if flag == 0: break
            return mask

    def _read_fits_cat(self, cat_file_name, cat_ext):
        import fitsio
        return fitsio.read(cat_file_name, cat_ext)

    def _read_parquet_cat(self, cat_file_name):
        import pyarrow.parquet as pq
        tab = pq.read_table(cat_file_name)
        # This works, but requires pandas
        #return tab.to_pandas().to_records(index=False)
        # This doesn't required pandas
        return np.rec.fromarrays(tab.columns, names=tab.schema.names)

    @staticmethod
    def _resolve_sed_file_name(value, cat_file_name):
        sed_file_name = os.fsdecode(value).strip()
        if sed_file_name == '':
            raise ValueError("Encountered blank SED file name")
        if os.path.isabs(sed_file_name):
            return sed_file_name
        cat_dir = os.path.dirname(os.path.abspath(cat_file_name))
        cat_relative = os.path.join(cat_dir, sed_file_name)
        if os.path.isfile(cat_relative):
            return cat_relative
        else:
            return sed_file_name

    def _read_sed_mapping(self, sed_file_name, cat_file_name):
        if isinstance(sed_file_name, dict):
            return sed_file_name

        sed_map_file = self._resolve_sed_file_name(sed_file_name, cat_file_name)
        with open(sed_map_file) as f:
            sed_map = yaml.safe_load(f)
        if not isinstance(sed_map, dict):
            raise ValueError("SED mapping file must parse to a dict: %r" % sed_map_file)
        return sed_map

    @staticmethod
    def _evaluate_properties(properties, extra_props, nstars, base, logger):
        if not isinstance(properties, dict):
            raise ValueError("properties should be a dict")

        base['eval_variables'] = {}
        # Load the existing extra properties into eval_variables so they are available
        # to be used in eval strings.
        for prop_name, value in extra_props.items():
            base['eval_variables']['x' + prop_name] = value

        parsed = {}
        for key, value in properties.items():
            try:
                result = galsim.config.ParseValue({'x': value}, 'x', base, None)[0]
                if np.ndim(result) == 0:
                    result = np.array([result] * nstars)
                else:
                    result = np.asarray(result)
            except Exception as e:
                logger.debug("Caught exception trying vector eval for property %r: %r", key, e)
                result = []
                row_base = base.copy()
                row_base['eval_variables'] = {}
                for i in range(nstars):
                    for k, v in base['eval_variables'].items():
                        row_base['eval_variables'][k] = v[i]
                    result.append(galsim.config.ParseValue({'x': value}, 'x', row_base, None)[0])
                result = np.array(result)
            else:
                if len(result) != nstars:
                    raise ValueError("properties[%r] did not evaluate to %d values" % (key, nstars))

            base['eval_variables']['x' + key] = result
            parsed[key] = result

        return parsed
