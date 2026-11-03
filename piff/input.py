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
from typing import NamedTuple

from .util import run_multi, read_asdf_image
from .star import Star, StarData
from .config import LoggerWrapper

class RawImageData(NamedTuple):
    """Image and catalog data returned by `InputFiles.getRawImageData`.

    Fields can be accessed by name or unpacked in the order listed below.

    :param image:       Science image after image-level sky subtraction.
    :param weight:      Weight image after masking and any source-variance removal.
    :param image_pos:   List of catalog positions in image coordinates.
    :param extra_props: Dict of additional catalog properties, with an array for each property.
    :param gain_image:  Per-pixel gain image, or None when no gain image was supplied.
    """
    image: galsim.Image
    weight: galsim.Image
    image_pos: list
    extra_props: dict
    gain_image: galsim.Image | None

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

        The base class implementation expects the derived class to have appropriately set the
        following attributes:

            :stamp_size:    The size of the postage stamp to use for the cutouts
            :x_col:         The name of the column in the catalogs to use for the x position.
            :y_col:         The name of the column in the catalogs to use for the y position.

        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a list of Star instances
        """
        logger = LoggerWrapper(logger)

        if self.nimages == 1:
            logger.debug("Making star list")
        else:
            logger.debug("Making star list from %d catalogs", self.nimages)

        args = [(self.image_kwargs[k], self.cat_kwargs[k], self.wcs_list[k], self.chipnums[k])
                for k in range(self.nimages)]
        kwargs = dict(stamp_size=self.stamp_size,
                      pointing=self.pointing, bandpass=self.bandpass, use_partial=self.use_partial,
                      invert_weight=self.invert_weight,
                      remove_signal_from_weight=self.remove_signal_from_weight,
                      config=self.config, base=self.base)

        all_stars = run_multi(self._makeStarsFromImage, self.nproc, raise_except=True,
                              args=args, logger=logger, kwargs=kwargs)

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
        return { chipnum : w for w, chipnum in zip(self.wcs_list, self.chipnums) }

    def getPointing(self, logger=None):
        """Get the pointing coordinate of the (noinal) center of the field of view.

        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a galsim.CelestialCoord of the pointing direction.
        """
        return self.pointing

    def getBandpass(self, logger=None):
        """Get the observing bandpass for the input images.

        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a galsim.Bandpass or None.
        """
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

    The above values are parsed separately for each input image/catalog.  In addition, there
    are a couple other parameters that are just parsed once:

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
                        `setPointing` for details about how this can be specified]

    :param config:      The configuration dict used to define the above parameters.
    :param base:        The base configuration dict if needed for value parsing. [default: None]
    :param logger:      A logger object for logging debug info. [default: None]
    """
    _type_name = 'Files'

    def __init__(self, config, base=None, logger=None):
        import copy
        logger = LoggerWrapper(logger)

        req = { 'image_file_name': str,
                'cat_file_name': str,
              }
        opt = {
                'dir' : str,
                'chipnum' : int,
                'x_col' : str,
                'y_col' : str,
                'ra_col' : str,
                'dec_col' : str,
                'ra_units' : str,
                'dec_units' : str,
                'property_cols' : list,
                'properties' : dict,
                'sed_col' : str,
                'sed_file_name' : (str, dict),
                'sed_wave_type' : str,
                'sed_flux_type' : str,
                'sed_wave_key' : str,
                'sed_flux_key' : str,
                'sed_tol' : float,
                'sed_max_samples' : int,
                'sky_col' : str,
                'gain_col' : str,
                'gain_file_name': str,
                'gain_ext': (int, str),
                'flag_col' : str,
                'skip_flag' : int,
                'use_flag' : int,
                'image_ext' : (int, str),
                'weight_ext' : (int, str),
                'badpix_ext' : (int, str),
                'weight_file_name' : str,
                'badpix_file_name' : str,
                'badpix_zeros' : bool,
                'fix_ushort_badpix' : bool,
                'sky_file_name' : str,
                'sky_ext': (int, str),
                'cat_ext' : (int, str),
                'invert_weight' : bool,
                'remove_signal_from_weight' : bool,
                'stamp_size' : int,
                'use_partial' : bool,
                'nstars' : int,
              }
        ignore = [ 'nproc', 'nimages', 'ra', 'dec', 'wcs', 'bandpass',
                   'sky', 'gain', 'satur', 'noise', 'trust_pos' ]

        # We're going to change the config dict a bit. Make a copy so we don't mess up the
        # user's original dict (in case they care).
        config = copy.deepcopy(config)
        base = copy.deepcopy(base) if base is not None else {'input': config}

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

        # In GalSim, the base dict holds additional parameters that may be of use.
        # For now, add index_kex, as this will be the only thing we index anything on.
        base['index_key'] = 'image_num'

        # Convert options 2 and 3 above into option 4.  (1 is also parseable by GalSim's config.)
        nimages = None
        image_list = None
        cat_list = None
        dir = None

        if 'nproc' in config:
            self.nproc = galsim.config.ParseValue(config, 'nproc', base, int)[0]

        top_level_opt = {
                'stamp_size' : int,
                'invert_weight' : bool,
                'remove_signal_from_weight' : bool,
                'use_partial' : bool,
              }
        top_level_params = galsim.config.GetAllParams(
                config, base, opt=top_level_opt, ignore=list(req) + list(opt) + ignore)[0]
        config.pop('_get')
        self.stamp_size = top_level_params.get('stamp_size', 32)
        self.invert_weight = top_level_params.get('invert_weight', False)
        self.remove_signal_from_weight = top_level_params.get('remove_signal_from_weight', False)
        self.use_partial = top_level_params.get('use_partial', False)

        if 'nstars' in config:
            logger.error("input.nstars is deprecated; use select.nstars for the new global "
                         "highest-S/N behavior.")

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

        # Deal with dir here, since sometimes we need to have it already attached for glob
        # to work.
        if 'dir' in config:
            dir = galsim.config.ParseValue(config, 'dir', base, str)[0]
            del config['dir']

        # Convert Current/Eval shorthand before literal filenames are expanded as glob patterns.
        for key in ['image_file_name', 'cat_file_name', 'sky_file_name',
                    'weight_file_name', 'badpix_file_name', 'gain_file_name']:
            value = config.get(key)
            if isinstance(value, str) and value.startswith('@'):
                config[key] = {'type': 'Current', 'key': value[1:]}
            elif isinstance(value, str) and value.startswith('$'):
                config[key] = {'type': 'Eval', 'str': value[1:]}

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

        # Set up sky_file_name if appropriate
        if 'sky_file_name' not in config:
            pass
        elif isinstance(config['sky_file_name'], str):
            sky_file_name = config['sky_file_name']
            if dir is not None:
                sky_file_name = os.path.join(dir, sky_file_name)
            sky_list = sorted(glob.glob(sky_file_name))
            if len(sky_list) == 0:
                raise ValueError("No files found corresponding to "+config['sky_file_name'])
            if len(sky_list) == 1 and nimages is not None and nimages > 1:
                logger.verbose("Using the same sky file for all images")
                sky_list = sky_list * nimages
            elif nimages is not None and len(sky_list) != nimages:
                raise ValueError("nimages = %s doesn't match length of sky_file_name list (%d)"%(
                                 nimages, len(sky_list)))
            config['sky_file_name'] = sky_list
        elif not isinstance(config['sky_file_name'], dict):
            raise ValueError("sky_file_name should be either a dict or a string")

        if 'weight_file_name' not in config:
            pass
        elif isinstance(config['weight_file_name'], str):
            weight_file_name = config['weight_file_name']
            if dir is not None:
                weight_file_name = os.path.join(dir, weight_file_name)
            weight_list = sorted(glob.glob(weight_file_name))
            if len(weight_list) == 0:
                raise ValueError("No files found corresponding to "+config['weight_file_name'])
            if len(weight_list) == 1 and nimages is not None and nimages > 1:
                logger.verbose("Using the same weight file for all images")
                weight_list = weight_list * nimages
            elif nimages is not None and len(weight_list) != nimages:
                raise ValueError("nimages = %s doesn't match length of weight_file_name list (%d)"%(
                                 nimages, len(weight_list)))
            config['weight_file_name'] = weight_list
        elif not isinstance(config['weight_file_name'], dict):
            raise ValueError("weight_file_name should be either a dict or a string")

        if 'badpix_file_name' not in config:
            pass
        elif isinstance(config['badpix_file_name'], str):
            badpix_file_name = config['badpix_file_name']
            if dir is not None:
                badpix_file_name = os.path.join(dir, badpix_file_name)
            badpix_list = sorted(glob.glob(badpix_file_name))
            if len(badpix_list) == 0:
                raise ValueError("No files found corresponding to "+config['badpix_file_name'])
            if len(badpix_list) == 1 and nimages is not None and nimages > 1:
                logger.verbose("Using the same badpix file for all images")
                badpix_list = badpix_list * nimages
            elif nimages is not None and len(badpix_list) != nimages:
                raise ValueError("nimages = %s doesn't match length of badpix_file_name list (%d)"%(
                                 nimages, len(badpix_list)))
            config['badpix_file_name'] = badpix_list
        elif not isinstance(config['badpix_file_name'], dict):
            raise ValueError("badpix_file_name should be either a dict or a string")

        if 'gain_file_name' not in config:
            pass
        elif isinstance(config['gain_file_name'], str):
            gain_file_name = config['gain_file_name']
            if dir is not None:
                gain_file_name = os.path.join(dir, gain_file_name)
            gain_list = sorted(glob.glob(gain_file_name))
            if len(gain_list) == 0:
                raise ValueError("No files found corresponding to " + config['gain_file_name'])
            if len(gain_list) == 1 and nimages > 1:
                logger.verbose("Using the same gain file for all images")
                gain_list = gain_list * nimages
            elif len(gain_list) != nimages:
                raise ValueError("nimages = %s doesn't match length of gain_file_name list (%d)" % (
                                 nimages, len(gain_list)))
            config['gain_file_name'] = gain_list
        elif not isinstance(config['gain_file_name'], dict):
            raise ValueError("gain_file_name should be either a dict or a string")

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
        self.chipnums = list(range(nimages))
        self.image_file_name = []
        self.cat_file_name = []
        self.image_kwargs = []
        self.cat_kwargs = []

        self.config = galsim.config.CleanConfig(config)
        self.base = galsim.config.CleanConfig(base)

        logger.verbose("Reading in %d images",nimages)
        for image_num in range(nimages):

            # This changes for each input image.
            galsim.config.RemoveCurrent(base)
            base['index_key'] = 'image_num'
            base['image_num'] = image_num

            logger.debug("config = %s", config)
            params = galsim.config.GetAllParams(config, base, req=req, opt=opt, ignore=ignore)[0]
            logger.debug("image_num = %d: params = %s", image_num, params)

            # Update the chipnum if not just using image_num
            if 'chipnum' in params:
                self.chipnums[image_num] = params['chipnum']

            # Read the image
            image_file_name = params['image_file_name']
            image_ext = params.get('image_ext', None)
            weight_file_name = params.get('weight_file_name', None)
            weight_ext = params.get('weight_ext', None)
            badpix_file_name = params.get('badpix_file_name', None)
            badpix_ext = params.get('badpix_ext', None)
            badpix_zeros = params.get('badpix_zeros', False)
            fix_ushort_badpix = params.get('fix_ushort_badpix', False)
            sky_file_name = params.get('sky_file_name', None)
            sky_ext = params.get('sky_ext', None)
            if sky_file_name is None and sky_ext is not None:
                sky_file_name = image_file_name
            gain_file_name = params.get('gain_file_name', None)
            gain_ext = params.get('gain_ext', None)
            if gain_file_name is None and gain_ext is not None:
                gain_file_name = image_file_name
            self.image_file_name.append(image_file_name)
            self.image_kwargs.append({
                    'image_file_name' : image_file_name,
                    'image_ext' : image_ext,
                    'weight_file_name' : weight_file_name,
                    'weight_ext' : weight_ext,
                    'badpix_file_name' : badpix_file_name,
                    'badpix_ext' : badpix_ext,
                    'badpix_zeros' : badpix_zeros,
                    'fix_ushort_badpix' : fix_ushort_badpix,
                    'sky_file_name' : sky_file_name,
                    'sky_ext' : sky_ext,
                    'gain_file_name': gain_file_name,
                    'gain_ext': gain_ext,
                    'image_num' : image_num,
                    'config' : self.config,
                    'base' : self.base,
                })

            # Read the catalog
            cat_file_name = params['cat_file_name']
            cat_ext = params.get('cat_ext', None)
            x_col = params.get('x_col', 'x')
            y_col = params.get('y_col', 'y')
            ra_col = params.get('ra_col', None)
            dec_col = params.get('dec_col', None)
            ra_units = params.get('ra_units', 'deg')
            dec_units = params.get('dec_units', 'deg')
            flag_col = params.get('flag_col', None)
            skip_flag = params.get('skip_flag', -1)
            use_flag = params.get('use_flag', None)
            property_cols = params.get('property_cols', None)
            properties = params.get('properties', None)
            sed_col = params.get('sed_col', None)
            sed_file_name = params.get('sed_file_name', None)
            sed_wave_type = params.get('sed_wave_type', None)
            sed_flux_type = params.get('sed_flux_type', None)
            sed_wave_key = params.get('sed_wave_key', 'WAVE')
            sed_flux_key = params.get('sed_flux_key', 'FLUX')
            sed_tol = params.get('sed_tol', 0.0)
            sed_max_samples = params.get('sed_max_samples', 0)
            if 'bandpass' in config:
                bandpass = galsim.config.BuildBandpass(config, 'bandpass', base, logger)[0]
            else:
                if sed_col is not None or sed_file_name is not None:
                    raise ValueError("bandpass is required when using sed_col or sed_file_name")
                bandpass = None
            sky_col = params.get('sky_col', None)
            gain_col = params.get('gain_col', None)
            nstars = params.get('nstars', None)

            self.cat_file_name.append(cat_file_name)
            self.cat_kwargs.append({
                    'cat_file_name' : cat_file_name,
                    'cat_ext' : cat_ext,
                    'x_col' : x_col,
                    'y_col' : y_col,
                    'ra_col' : ra_col,
                    'dec_col' : dec_col,
                    'ra_units' : ra_units,
                    'dec_units' : dec_units,
                    'flag_col' : flag_col,
                    'skip_flag' : skip_flag,
                    'use_flag' : use_flag,
                    'property_cols': property_cols,
                    'properties': properties,
                    'sed_col': sed_col,
                    'sed_file_name': sed_file_name,
                    'sed_wave_type': sed_wave_type,
                    'sed_flux_type': sed_flux_type,
                    'sed_wave_key': sed_wave_key,
                    'sed_flux_key': sed_flux_key,
                    'sed_tol': sed_tol,
                    'sed_max_samples': sed_max_samples,
                    'sky_col' : sky_col,
                    'gain_col' : gain_col,
                    'nstars' : nstars,
                    'stamp_size' : self.stamp_size,
                    'config' : self.config,
                    'base' : self.base,
                })

        self.bandpass = bandpass

        # Read all the wcs's, since we'll need this for the pointing, which in turn we'll
        # need for when we make the stars.
        self.setWCS(config, base, logger)

        # Finally, set the pointing coordinate.
        # Note: setPointing will do the config parsing of these if appropriate.
        ra = config.get('ra', None)
        dec = config.get('dec', None)
        self.setPointing(ra, dec, logger)

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

        images = {}
        weight_images = {}
        gain_images = {}
        for image_num in range(self.nimages):
            data = self.getRawImageData(image_num, logger)
            images[self.chipnums[image_num]] = data.image
            weight_images[self.chipnums[image_num]] = data.weight
            gain_images[self.chipnums[image_num]] = data.gain_image

        loaded_stars = []
        for star in stars:
            chipnum = star['chipnum']
            image = images[chipnum][star.data.image.bounds].copy()
            if 'sky' in star.data.properties:
                image -= star['sky']
            weight = weight_images[chipnum][star.data.weight.bounds].copy()
            gain_image = gain_images[chipnum]
            if gain_image is not None:
                gain_image = gain_image[image.bounds].copy()
            data = star.data.withNew(image=image, weight=weight, orig_weight=weight,
                                     gain_image=gain_image)
            loaded_stars.append(Star(data=data, fit=star.fit))

        return loaded_stars

    def getRawImageData(self, image_num, logger=None):
        """Read an image and catalog.

        :param image_num:   The index of the image to read.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: A `RawImageData` named tuple with fields image, weight, image_pos,
                  extra_props, and gain_image. Fields can be accessed by name or unpacked
                  in that order.
        """
        self.base['index_key'] = 'image_num'
        self.base['image_num'] = image_num
        return self._getRawImageData(self.image_kwargs[image_num], self.cat_kwargs[image_num],
                                     self.wcs_list[image_num], self.invert_weight,
                                     self.remove_signal_from_weight,
                                     config=self.config, base=self.base, logger=logger)

    @staticmethod
    def _handle_depr_image_value(value, header, name, logger):
        # gain, satur, noise, sky, ra, and dec used to allow bare FITS header keys.
        # Now this functionality should use the ImageHeaderValue type to be more explicit.
        # The old behavior still works, but is deprecated.
        if isinstance(value, str) and value[0] not in '@$':
            logger.error("WARNING: Using a bare FITS header key for %s is deprecated. "
                         "Use {type: ImageHeaderValue, key: %s} instead.", name, value)
            value = header[value]
            # Pointing values may be sexagesimal strings, parsed by setPointing.
            if name not in ('ra', 'dec'):
                value = float(value)
        return value

    @classmethod
    def _getRawImageData(cls, image_kwargs, cat_kwargs, wcs,
                         invert_weight, remove_signal_from_weight,
                         config, base, logger=None):
        from .config import LoggerWrapper
        logger = LoggerWrapper(logger)
        image, weight, gain_image = cls.readImage(logger=logger, **image_kwargs)

        if invert_weight:
            weight.invertSelf()

        # Update the wcs
        image.wcs = wcs

        image_pos, extra_props = cls.readStarCatalog(
                logger=logger, image=image, **cat_kwargs)

        if remove_signal_from_weight:
            nmasked = np.count_nonzero(weight.array == 0)
            # Subtract the mean catalog sky from the signal used for variance removal.
            # Image-level backgrounds have already been subtracted by readImage.
            if 'sky' in extra_props:
                signal = image - np.mean(extra_props['sky'])
            else:
                signal = image
            if gain_image is not None:
                weight, _ = cls._removeSignalFromWeight(signal, weight, gain=gain_image)
            elif 'gain' in extra_props:
                # If given, use the mean gain when removing the signal.
                # This isn't quite right, but hopefully the gain won't vary too much for
                # different objects, so it should be close.
                gain = np.mean(extra_props['gain'])
                weight, _ = cls._removeSignalFromWeight(signal, weight, gain=gain)
            else:
                # If no gain provided, then we want to estimate the gain from the weight image.
                weight, g = cls._removeSignalFromWeight(signal, weight)
                extra_props['gain'] = np.array([g] * len(image_pos), dtype=float)
                logger.info("Empirically determined gain = %f",g)
            logger.verbose("Removed signal from weight image.")
            added_masks = np.count_nonzero(weight.array == 0) - nmasked
            if added_masks:
                logger.warning("Masked %d pixels with invalid variance after removing source noise",
                               added_masks)

        return RawImageData(image=image, weight=weight, image_pos=image_pos,
                            extra_props=extra_props, gain_image=gain_image)

    @classmethod
    def _makeStarsFromImage(cls, image_kwargs, cat_kwargs, wcs, chipnum,
                            stamp_size, pointing, bandpass, use_partial,
                            invert_weight, remove_signal_from_weight,
                            config, base, logger):
        """Make "stars" from a single input image
        """
        image, wt, image_pos, extra_props, gain_image = cls._getRawImageData(
                image_kwargs, cat_kwargs, wcs, invert_weight, remove_signal_from_weight,
                config, base, logger)
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
                if use_partial:
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

            # Subtract the sky
            if 'sky' in props:
                sky = props['sky']
                logger.debug("Subtracting off sky = %f", sky)
                logger.debug("Median pixel value = %f", np.median(stamp.array))
                stamp -= sky

            pos = galsim.PositionD(x,y)
            gain_stamp = gain_image[bounds].copy() if gain_image is not None else None
            data = StarData(stamp, pos, weight=wt_stamp, pointing=pointing,
                            bandpass=bandpass,
                            properties=props, property_types=prop_types,
                            gain_image=gain_stamp)
            # Attach an attribute so we can tell later whether the weight is real or
            # if it's just the default w=1.0.
            data._piff_default_weight = getattr(wt, '_piff_default_weight', False)
            star = Star(data, None)

            objects.append(star)

        return objects

    def setWCS(self, config, base, logger):
        self.wcs_list = []
        self.center_list = []
        for image_num, kwargs in enumerate(self.image_kwargs):
            galsim.config.RemoveCurrent(base) # Makes any @ items work correctly
            base['image_num'] = image_num
            image_file_name = kwargs['image_file_name']
            image_ext = kwargs['image_ext']
            image = self._read_image(image_file_name, image_ext, read_header=True)
            if 'wcs' in config:
                logger.info("Using custom wcs from config for %s",image_file_name)
                base['index_key'] = 'image_num'
                base['image_num'] = image_num
                base['_current_image'] = image
                wcs = galsim.config.BuildWCS(config, 'wcs', base, logger)
            else:
                logger.info("Getting wcs from image file %s",image_file_name)
                wcs = image.wcs
            self.wcs_list.append(wcs)
            self.center_list.append(image.true_center)

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

        newweight = weight.copy()
        valid = np.isfinite(variance) & (variance > 0)
        new_values = np.zeros_like(variance)
        np.divide(1.0, variance, out=new_values, where=valid)
        newweight.array[use] = new_values
        return newweight, gain

    @staticmethod
    def _read_image(file_name, ext, *, read_header):
        """Read one image, optionally including its metadata.

        Subclasses may override this method to support other file formats. The returned
        GalSim image must include its WCS and, when requested, a ``header`` attribute.
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

    @classmethod
    def readImage(cls, image_file_name, image_ext, weight_file_name, weight_ext,
                  badpix_file_name, badpix_ext, badpix_zeros, fix_ushort_badpix,
                  sky_file_name, sky_ext,
                  gain_file_name, gain_ext, image_num, config, base, logger):
        """Read the science image, weight map, and optional gain image.

        Make a weight map if no weight information is given.

        :param image_file_name: The name of the file to read.
        :param image_ext:       The FITS extension number/name or ASDF path of the image.
        :param weight_file_name: The name if the file with the weight image (if any).
        :param weight_ext:      The FITS extension number/name or ASDF path of the weight image.
        :param badpix_file_name: The name of the file with the bad pixel mask (if any).
        :param badpix_ext:      The FITS extension number/name or ASDF path of the bad pixel mask.
        :param badpix_zeros:    Whether to treat all zeros in the input image as bad pixels.
                                [default: False]
        :param fix_ushort_badpix: Whether to correct badpix arrays with a 32768 offset value.
                                [default: False]
        :param sky_file_name:   A file to use for a sky background to subtract from the image
                                (if any).
        :param sky_ext:         The FITS extension number/name or ASDF path in sky_file_name.
        :param gain_file_name:  The name of the file with the gain image (if any).
        :param gain_ext:        The FITS extension number/name or ASDF path of the gain image.
        :param config:          The input section of the config dict.
        :param base:            The base config dict.
        :param logger:          A logger object for logging debug info.

        :returns: image, weight, gain_image (None when no gain image is supplied)
        """
        # Read in the image
        logger.info("Reading image file %s",image_file_name)
        galsim.config.RemoveCurrent(base)
        base['image_num'] = image_num
        image = cls._read_image(image_file_name, image_ext, read_header=True)
        base['_current_image'] = image

        noise = None
        if 'noise' in config:
            noise = galsim.config.ParseValue(config, 'noise', base, (float, str))[0]

        sky = None
        if 'sky' in config:
            sky = galsim.config.ParseValue(config, 'sky', base, (float, str))[0]

        # Make sure dtype is at least float32
        if image.dtype not in [np.float32, np.float64]:
            header = image.header  # This doesn't get copied by view()
            image = image.view(dtype=np.float32)
            image.header = header

        config['_current_image'] = image

        if noise is not None:
            noise = galsim.config.ParseValue(config, 'noise', base, None)[0]
            noise = cls._handle_depr_image_value(noise, image.header, 'noise', logger)

        # Either read in the weight image, or build a dummy one
        if weight_file_name is not None or weight_ext is not None:
            weight_file_name = weight_file_name or image_file_name
            logger.verbose("Reading weight image from %s, extension %s.",
                           weight_file_name, weight_ext)
            weight = cls._read_image(weight_file_name, weight_ext, read_header=False)
            if np.all(weight.array == 0):
                logger.error("According to the weight mask in %s, all pixels have zero weight!",
                             image_file_name)
            if np.any(weight.array < 0):
                logger.warning("Warning: weight map has invalid negative-valued pixels. "+
                               "Taking them to be 0.0")
                weight.array[weight.array < 0] = 0.
            weight._piff_default_weight = False
        elif noise is not None:
            logger.debug("Making uniform weight image based on noise variance = %f", noise)
            weight = galsim.ImageF(image.bounds, init_value=1./noise)
            weight._piff_default_weight = False
        else:
            logger.debug("Making trivial (wt==1) weight image")
            weight = galsim.ImageF(image.bounds, init_value=1)
            weight._piff_default_weight = True

        # Make sure to do this before anything that could modify the input image.
        if badpix_zeros:
            weight.array[image.array == 0] = 0

        # If requested, set wt=0 for any bad pixels
        if badpix_file_name is not None or badpix_ext is not None:
            badpix_file_name = badpix_file_name or image_file_name
            logger.verbose("Reading badpix image from %s, extension %s.",
                           badpix_file_name, badpix_ext)
            badpix = cls._read_image(badpix_file_name, badpix_ext, read_header=False)

            # Correct legacy unsigned-integer offsets if requested.
            if fix_ushort_badpix:
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
                             image_file_name)
            weight.array[badpix.array != 0] = 0

        # Any Inf or NaN pixels in the science image should also be given weight=0.
        # Additionally, set the value to 0, so downstream processing that does math like
        # weight * image doesn't get NaNs in the calculation.
        nonfinite = ~np.isfinite(image.array)
        weight.array[nonfinite] = 0
        image.array[nonfinite] = 0

        sky_image = cls._read_sky(image, weight, sky, sky_file_name, sky_ext, logger)
        if sky_image is not None:
            image -= sky_image

        gain_image = None
        if gain_file_name is not None:
            logger.verbose("Reading gain image from %s, extension %s.", gain_file_name, gain_ext)
            gain_image = cls._read_image(gain_file_name, gain_ext, read_header=False)
            if gain_image.bounds != image.bounds:
                raise ValueError('Gain image bounds do not match science image bounds')
            use = weight.array != 0
            if np.any(~np.isfinite(gain_image.array[use]) | (gain_image.array[use] <= 0)):
                raise ValueError('Gain image must be finite and positive at unmasked pixels')

        return image, weight, gain_image

    @classmethod
    def _read_sky(cls, image, weight, sky, sky_file_name, sky_ext, logger):
        """Read or estimate the image-level background after assembling the pixel mask.

        Returns a sky image for a file or SEP estimate, or a scalar for a configured value
        or image median. Returns None when no image-level sky is specified. Catalog sky
        values are applied later to individual stars. Subclasses can reuse the returned
        background when constructing background-dependent weight maps.
        """
        if sky_file_name is not None:
            ext_str = " from extension %s" % sky_ext if sky_ext is not None else ""
            logger.verbose("Reading sky image %s.", sky_file_name + ext_str)
            return cls._read_image(sky_file_name, sky_ext, read_header=False)
        elif sky == 'median':
            return np.median(image.array[weight.array != 0])
        elif sky == 'sep':
            import sep
            # SEP requires contiguous arrays with native byte order; FITS arrays may be big-endian.
            data = np.ascontiguousarray(image.array, dtype=image.array.dtype.newbyteorder('='))
            background = sep.Background(data, mask=weight.array == 0)
            return galsim.Image(background.back(), bounds=image.bounds, wcs=image.wcs)
        elif sky is not None:
            # Float value or a (now deprecated) image header key.
            return cls._handle_depr_image_value(sky, image.header, 'sky', logger)
        else:
            return None

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

    @classmethod
    def _read_cat(cls, cat_file_name, cat_ext):
        if cat_file_name.endswith('.fits'):
            return cls._read_fits_cat(cat_file_name, cat_ext)
        elif cat_file_name.endswith('.parquet'):
            return cls._read_parquet_cat(cat_file_name)
        else:
            # For other endings, try parquet then fits so error message is for fits.
            try:
                return cls._read_parquet_cat(cat_file_name)
            except Exception:
                return cls._read_fits_cat(cat_file_name, cat_ext)

    @classmethod
    def _read_fits_cat(cls, cat_file_name, cat_ext):
        import fitsio
        return fitsio.read(cat_file_name, cat_ext)

    @classmethod
    def _read_parquet_cat(cls, cat_file_name):
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

    @classmethod
    def _read_sed_mapping(cls, sed_file_name, cat_file_name):
        if isinstance(sed_file_name, dict):
            return sed_file_name

        sed_map_file = cls._resolve_sed_file_name(sed_file_name, cat_file_name)
        with open(sed_map_file) as f:
            sed_map = yaml.safe_load(f)
        if not isinstance(sed_map, dict):
            raise ValueError("SED mapping file must parse to a dict: %r" % sed_map_file)
        return sed_map

    @classmethod
    def readStarCatalog(cls, cat_file_name, cat_ext, x_col, y_col,
                        ra_col, dec_col, ra_units, dec_units, image,
                        flag_col, skip_flag, use_flag, property_cols, sed_col, sed_wave_type,
                        sed_file_name, sed_flux_type, sed_wave_key, sed_flux_key,
                        sed_tol, sed_max_samples,
                        properties, sky_col, gain_col, nstars, stamp_size,
                        config, base, logger):
        """Read in the star catalogs and return lists of positions for each star in each image.

        :param cat_file_name:   The name of the catalog file to read in.
        :param cat_ext:         The extension number or name to use for FITS catalogs.
        :param x_col:           The name of the column with x values.
        :param y_col:           The name of the column with y values.
        :param ra_col:          The name of a column with RA values.
        :param dec_col:         The name of a column with Dec values.
        :param ra_units:        The units of the ra column.
        :param dec_units:       The units of the dec column.
        :param image:           The image that was already read in (mostly for the wcs).
        :param flag_col:        The name of a column with flag values.
        :param skip_flag:       The flag indicating which items to not use. [default: -1]
                                Items with flag & skip_flag != 0 will be skipped.
        :param use_flag:        The flag indicating which items to use. [default: None]
                                Items with flag & use_flag == 0 will be skipped.
        :param property_cols:   A list of column names with star properties (e.g. star colors).
        :param properties:      A dict of additional star properties to attach to every star.
        :param sed_col:         A column with per-star SED file names.
        :param sed_file_name:   A single SED file name for all stars (if sed_col is None).
        :param sed_wave_type:   Wave type for SED files.
        :param sed_flux_type:   Flux type for SED files.
        :param sed_wave_key:    FITS-table wavelength column name for SED files.
        :param sed_flux_key:    FITS-table flux column name for SED files.
        :param sed_tol:         Relative tolerance for reducing loaded SEDs.
        :param sed_max_samples: Maximum number of retained SED samples.
        :param sky_col:         A column with sky (background) levels.
        :param gain_col:        A column with gain values.
        :param nstars:          Optionally a maximum number of stars to use from this catalog.
        :param stamp_size:      The stamp size being used for the star stamps.
        :param config:          The input section of the config dict.
        :param base:            The base config dict.
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: lists image_pos, extra_props
        """
        import fitsio

        base['_current_image'] = image

        if cat_file_name is None:
            # This is possible e.g. when loading images into an existing list of star instances.
            logger.info("Not reading input catalog.")
            return None, None

        # Read in the star catalog
        logger.info("Reading input catalog %s.",cat_file_name)
        cat = cls._read_cat(cat_file_name, cat_ext)

        gain = None
        if 'gain' in config:
            gain = galsim.config.ParseValue(config, 'gain', base, (float, str))[0]

        satur = None
        if 'satur' in config:
            satur = galsim.config.ParseValue(config, 'satur', base, (float, str))[0]

        trust_pos = None
        if 'trust_pos' in config:
            trust_pos = galsim.config.ParseValue(config, 'trust_pos', base, bool)[0]

        if flag_col is not None:
            if flag_col not in cat.dtype.names:
                raise ValueError("flag_col = %s is not a column in %s"%(flag_col,cat_file_name))
            col = cat[flag_col]
            if len(col.shape) == 2:
                logger.info("Flag col (%s) is multidimensional.  Treating as an array of bool",
                            flag_col)
            if use_flag is not None:
                # Remove any objects with flag & use_flag == 0
                mask = cls._flag_select(col, use_flag) == 0
                logger.verbose("Removing objects with flag (col %s) & %d == 0",flag_col,use_flag)
                if skip_flag != -1:
                    mask |= cls._flag_select(col, skip_flag) != 0
                    logger.verbose("Removing objects with flag (col %s) & %d != 0",flag_col,skip_flag)
            else:
                # Remove any objects with flag & skip_flag != 0
                mask = cls._flag_select(col, skip_flag) != 0
                if skip_flag == -1:
                    logger.verbose("Removing objects with flag (col %s) != 0",flag_col)
                else:
                    logger.verbose("Removing objects with flag (col %s) & %d != 0",flag_col,skip_flag)
            cat = cat[mask == 0]

        # Limit to nstars objects
        if nstars is not None and nstars < len(cat):
            logger.verbose("Limiting to %d objects for %s", nstars, cat_file_name)
            cat = cat[:nstars]

        # Make the list of positions:
        if ra_col is not None or dec_col is not None:
            if ra_col is None or dec_col is None:
                raise ValueError("ra_col and dec_col are both required if one is provided.")
            if ra_col not in cat.dtype.names:
                raise ValueError("ra_col = %s is not a column in %s"%(ra_col,cat_file_name))
            if dec_col not in cat.dtype.names:
                raise ValueError("dec_col = %s is not a column in %s"%(dec_col,cat_file_name))
            logger.debug("Starting to make a list of positions from ra, dec")
            ra_values = cat[ra_col]
            dec_values = cat[dec_col]
            ra_units = galsim.AngleUnit.from_name(ra_units)
            dec_units = galsim.AngleUnit.from_name(dec_units)
            ra = ra_values * ra_units
            dec = dec_values * dec_units
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
            if x_col not in cat.dtype.names:
                raise ValueError("x_col = %s is not a column in %s"%(x_col,cat_file_name))
            if y_col not in cat.dtype.names:
                raise ValueError("y_col = %s is not a column in %s"%(y_col,cat_file_name))
            x_values = cat[x_col]
            y_values = cat[y_col]
            logger.debug("Initially %d positions",len(x_values))
            image_pos = [ galsim.PositionD(x,y) for x,y in zip(x_values, y_values) ]

        # Check for objects well off the edge.  We won't use them.
        big_bounds = image.bounds.expand(stamp_size)
        image_pos = [ pos for pos in image_pos if big_bounds.includes(pos) ]
        logger.debug("After remove those that are off the image, len = %s",len(image_pos))

        # Make a dictionary of the star properties:
        extra_props = {}
        if property_cols is not None:
            logger.debug('property_cols = %s'%property_cols)
            for col_name in property_cols:
                if col_name not in cat.dtype.names:
                    raise ValueError("Entry in property_cols = " +
                                     "%s is not a column in %s"%(col_name,cat_file_name))
                extra_props[col_name] = cat[col_name]

        if sed_col is not None:
            if sed_col not in cat.dtype.names:
                raise ValueError("sed_col = %s is not a column in %s" % (sed_col, cat_file_name))
            sed_map = None
            if sed_file_name is not None:
                sed_map = cls._read_sed_mapping(sed_file_name, cat_file_name)
            sed_file_name_values = []
            for sed_label in cat[sed_col]:
                if sed_map is not None:
                    sed_label = os.fsdecode(sed_label).strip()
                    if sed_label not in sed_map:
                        raise ValueError(
                            "sed_col value %r is not in sed_file_name mapping" % sed_label
                        )
                    sed_file_name = sed_map[sed_label]
                else:
                    sed_file_name = sed_label
                sed_file_name = cls._resolve_sed_file_name(sed_file_name, cat_file_name)
                sed_file_name_values.append(sed_file_name)
            extra_props['sed_file_name'] = np.array(sed_file_name_values, dtype=object)
            extra_props['sed_wave_type'] = np.array([sed_wave_type] * len(cat), dtype=object)
            extra_props['sed_flux_type'] = np.array([sed_flux_type] * len(cat), dtype=object)
            extra_props['sed_wave_key'] = np.array([sed_wave_key] * len(cat), dtype=object)
            extra_props['sed_flux_key'] = np.array([sed_flux_key] * len(cat), dtype=object)
            extra_props['sed_tol'] = np.full(len(cat), sed_tol, dtype=float)
            extra_props['sed_max_samples'] = np.full(len(cat), sed_max_samples, dtype=int)
        elif sed_file_name is not None:
            sed_file_name = cls._resolve_sed_file_name(sed_file_name, cat_file_name)
            extra_props['sed_file_name'] = np.array([sed_file_name] * len(cat), dtype=object)
            extra_props['sed_wave_type'] = np.array([sed_wave_type] * len(cat), dtype=object)
            extra_props['sed_flux_type'] = np.array([sed_flux_type] * len(cat), dtype=object)
            extra_props['sed_wave_key'] = np.array([sed_wave_key] * len(cat), dtype=object)
            extra_props['sed_flux_key'] = np.array([sed_flux_key] * len(cat), dtype=object)
            extra_props['sed_tol'] = np.full(len(cat), sed_tol, dtype=float)
            extra_props['sed_max_samples'] = np.full(len(cat), sed_max_samples, dtype=int)

        # If we used a flag column, keep it as a property.
        if flag_col is not None:
            extra_props[flag_col] = cat[flag_col]

        # Make the list of sky values:
        if sky_col is not None:
            if sky_col not in cat.dtype.names:
                raise ValueError("sky_col = %s is not a column in %s"%(sky_col,cat_file_name))
            extra_props['sky'] = np.float32(cat[sky_col])

        # Make the list of gain values:
        # TODO: SV and Y1 DES images have two gain values, GAINA, GAINB.  It would be nice if we
        #       could get the right one properly.  OTOH, Y3+ will be in electrons, so gain=1 will
        #       the right value for all images.  So maybe not worth worrying about.
        if gain_col is not None:
            if gain_col not in cat.dtype.names:
                raise ValueError("gain_col = %s is not a column in %s"%(gain_col,cat_file_name))
            extra_props['gain'] = cat[gain_col]
        elif gain is not None:
            gain = galsim.config.ParseValue(config, 'gain', base, None)[0]
            gain = cls._handle_depr_image_value(gain, image.header, 'gain', logger)
            extra_props['gain'] = np.array([gain]*len(cat), dtype=float)

        # Get the saturation level
        if satur is not None:
            satur = galsim.config.ParseValue(config, 'satur', base, None)[0]
            satur = cls._handle_depr_image_value(satur, image.header, 'satur', logger)
            logger.debug("Using saturation value: %s",satur)
            extra_props['satur'] = np.array([satur]*len(cat), dtype=float)

        # Check whether we should trust the positions
        if trust_pos is not None:
            extra_props['trust_pos'] = np.array([trust_pos]*len(cat), dtype=bool)

        if properties is not None:
            logger.debug('properties = %s', properties)
            extra_props.update(cls._evaluate_properties(
                properties, extra_props, len(cat), base, logger))

        return image_pos, extra_props

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

    def setPointing(self, ra, dec, logger=None):
        """Set the pointing attribute based on the input ra, dec (given in the initializer)

        There are a number of ways the pointing can be specified.
        Even this is probably not sufficiently generic for all applications, but it's a start.

        1. numerical values (in Hours, Degrees respective) for ra, dec
        2. hh:mm:ss.ssss, dd:mm:ss.ssss strings giving hours/degrees, minutes, seconds for each
        3. FITS header key words to read to get the ra, dec values [deprecated;
           use ImageHeaderValue config types instead]
        4. None, which will attempt to find the spatial center of all the input images using the
           midpoint of the min/max ra and dec values of the image corners according to their
           individual WCS functions. [Not implemented currently.]
        5. dict with GalSim-style config types to be parsed.
        """
        logger = LoggerWrapper(logger)

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
        elif (isinstance(ra, dict) or isinstance(dec, dict) or
              (isinstance(ra, str) and (ra[0] in ['$','@'])) or
              (isinstance(dec, str) and (dec[0] in ['$','@']))):
            image_file_name = self.image_kwargs[0]['image_file_name']
            image_ext = self.image_kwargs[0]['image_ext']
            image = self._read_image(image_file_name, image_ext, read_header=True)
            base = galsim.config.CopyConfig(self.base)
            base['index_key'] = 'image_num'
            base['image_num'] = 0
            base['_current_image'] = image
            ra = galsim.config.ParseValue({'ra': ra}, 'ra', base, None)[0]
            dec = galsim.config.ParseValue({'dec': dec}, 'dec', base, None)[0]
            self.setPointing(ra, dec, logger)
        elif type(ra) in [float, int]:
            ra = float(ra) * galsim.hours
            dec = float(dec) * galsim.degrees
            self.pointing = galsim.CelestialCoord(ra,dec)
            logger.verbose("Setting pointing to: %.3f h, %.3f d",
                           self.pointing.ra / galsim.hours,
                           self.pointing.dec / galsim.degrees)
        elif str(ra) != ra:
            raise ValueError("Unable to parse input ra: %s"%ra)
        elif str(dec) != dec:
            raise ValueError("Unable to parse input dec: %s"%dec)
        elif ':' in ra and ':' in dec:
            ra = galsim.Angle.from_hms(ra)
            dec = galsim.Angle.from_dms(dec)
            self.pointing = galsim.CelestialCoord(ra,dec)
            logger.verbose("Setting pointing to: %.3f h, %.3f d",
                           self.pointing.ra / galsim.hours,
                           self.pointing.dec / galsim.degrees)
        else:
            # This is now deprecated. This uses ra, dec as header keys, but these should now
            # use ImageHeaderValue instead to be more explicit.
            file_name = self.image_file_name[0]
            if len(self.chipnums) == 1:
                logger.verbose("Setting pointing from keywords %s, %s", ra, dec)
            else:
                logger.verbose("Setting pointing from keywords %s, %s in %s", ra, dec, file_name)
            hdu = 1 if file_name.endswith('.fz') else 0
            import fitsio
            header = fitsio.read_header(file_name, ext=hdu)
            ra = self._handle_depr_image_value(ra, header, 'ra', logger)
            dec = self._handle_depr_image_value(dec, header, 'dec', logger)
            # Recurse to do further parsing.
            self.setPointing(ra, dec, logger)
