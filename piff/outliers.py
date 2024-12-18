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
.. module:: outliers
"""

import math
import numpy as np
import math
import galsim
from scipy.stats import chi2


class Outliers(object):
    """The base class for handling outliers.

    This is essentially an abstract base class intended to define the methods that should be
    implemented by any derived class.
    """
    # This class-level dict will store all the valid outlier types.
    # Each subclass should set a cls._type_name, which is the name that should
    # appear in a config dict.  These will be the keys of valid_outliers_types.
    # The values in this dict will be the Outliers sub-classes.
    valid_outliers_types = {}

    @classmethod
    def process(cls, config_outliers, logger=None):
        """Parse the outliers field of the config dict.

        :param config_outliers: The configuration dict for the outliers field.
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: an Outliers instance
        """
        # Get the class to use for the outliers
        if 'type' not in config_outliers:
            raise ValueError("config['outliers'] has no type field")

        outliers_type = config_outliers['type']
        if outliers_type not in Outliers.valid_outliers_types:
            raise ValueError("type %s is not a valid model type. "%outliers_type +
                             "Expecting one of %s"%list(Outliers.valid_outliers_types.keys()))

        outliers_class = Outliers.valid_outliers_types[outliers_type]

        # Read any other kwargs in the outliers field
        kwargs = outliers_class.parseKwargs(config_outliers, logger)

        # Build outliers object
        outliers = outliers_class(**kwargs)

        return outliers

    @classmethod
    def __init_subclass__(cls):
        if hasattr(cls, '_type_name') and cls._type_name is not None:
            if cls._type_name in Outliers.valid_outliers_types:
                raise ValueError('Outliers type %s already registered'%cls._type_name +
                                 'Maybe you subclassed and forgot to set _type_name?')
            Outliers.valid_outliers_types[cls._type_name] = cls

    @classmethod
    def parseKwargs(cls, config_outliers, logger=None):
        """Parse the outliers field of a configuration dict and return the kwargs to use for
        initializing an instance of the class.

        The base class implementation just returns the kwargs as they are, but derived classes
        might want to override this if they need to do something more sophisticated with them.

        :param config_outliers: The outliers field of the configuration dict, config['outliers']
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: a kwargs dict to pass to the initializer
        """
        kwargs = {}
        kwargs.update(config_outliers)
        kwargs.pop('type',None)
        kwargs['logger'] = logger
        return kwargs

    def write(self, writer, name):
        """Write an Outers via a writer object.

        :param writer:      A writer object that encapsulates the serialization format.
        :param name:        A name to associate with the Ootliers in the serialized output.
        """
        # First write the basic kwargs that works for all Outliers classes
        outliers_type = self._type_name
        writer.write_struct(name, dict(self.kwargs, type=outliers_type))

        # Now do any class-specific steps.
        with writer.nested(name) as w:
            self._finish_write(w)

    def _finish_write(self, writer):
        """Finish the writing process with any class-specific steps.

        The base class implementation doesn't do anything, which is often appropriate, but
        this hook exists in case any Outliers classes need to write extra information to the
        fits file.

        :param writer:      A writer object that encapsulates the serialization format.
        :param name:        A name to associate with the outliers in the serialized output.
        """
        pass

    @classmethod
    def read(cls, reader, name):
        """Read a Outliers from a FITS file.

        :param reader:      A reader object that encapsulates the serialization format.
        :param name:        Name associated with the outliers in the serialized output.

        :returns: an Outliers handler, or None if there isn't one.
        """
        kwargs = reader.read_struct(name)
        if kwargs is None:
            return None
        assert 'type' in kwargs
        outliers_type = kwargs.pop('type')

        # Old output files had the full name.  Fix it if necessary.
        if (outliers_type.endswith('Outliers')
                and outliers_type not in Outliers.valid_outliers_types):
            outliers_type = outliers_type[:-len('Outliers')]

        # Check that outliers_type is a valid Outliers type.
        if outliers_type not in Outliers.valid_outliers_types:
            raise ValueError("outliers type %s is not a valid Piff Outliers"%outliers_type)
        outliers_cls = Outliers.valid_outliers_types[outliers_type]

        outliers = outliers_cls(**kwargs)
        with reader.nested(name) as r:
            outliers._finish_read(r)
        return outliers

    def _finish_read(self, reader):
        """Finish the reading process with any class-specific steps.

        The base class implementation doesn't do anything, which is often appropriate, but
        this hook exists in case any Outliers classes need to read extra information from the
        fits file.

        :param reader:      A reader object that encapsulates the serialization format.
        """
        pass

class MADOutliers(Outliers):  # pragma: no cover  (This isn't functional yet.)
    """An Outliers handler using mean absolute deviation (MAD) for defining outliers.

        MAD = < |x - median] >

    The range of values to keep in the sample are

        [median - nmad * MAD, median + nmad * MAD]

    where nmad is a parameter specified by the user.

    The user can specify this parameter in one of two ways.

        1. The user can specify nmad directly.
        2. The user can specify nsigma, in which case nmad = sqrt(pi/2) nsigma, the equivalent
           value for a Gaussian distribution.

    Either nmad or nsigma must be provided.

    :param nmad:        The number of mean absolute deviations from the median at which
                        something is declared an outlier.
    :param nsigma:      The number of sigma equivalent if the underlying distribution is
                        Gaussian.
    :param logger:      A logger object for logging debug info. [default: None]
    """
    _type_name = 'MAD'

    def __init__(self, nmad=None, nsigma=None, logger=None):
        if nmad is None and nsigma is None:
            raise TypeError("Either nmad or nsigma is required")
        if nsigma is not None:
            nmad = math.sqrt(math.pi/2) * nsigma
        self.nmad = nmad

        self.kwargs = {
            'nmad' : nmad,
        }

    def removeOutliers(self, stars):
        """Remove outliers from a list of stars based on their fit parameters.

        :param stars:       A list of Star instances

        :returns:           A new list of stars without outliers
        """
        # I started writing this class, but then realized this isn't what we want to do
        # for the PixelGrid model.  So leave this for now...
        raise NotImplementedError("MAD algorithm not implemented")


class ChisqOutliers(Outliers):
    """An Outliers handler using the chisq of the residual of the interpolated star with the
    original.

    The user can specify the threshold in one of four ways:

        1. The user can specify thresh directly.
        2. The user can specify ndof to give a multiple of the number of degrees of freedom of
           the model, thresh = ndof * dof.
        3. The user can specify prob to reject according to the probability that the chisq
           distribution for the model's number of degrees of freedom would exceed chisq.
        4. The user can specify nsigma, in which case thresh is calculated according to the
           chisq distribution to give the equivalent rejection probability that corresponds
           to that many sigma.

    Exactly one of thresh, ndof, nsigma, prob must be provided.

    .. note::

        Reserve stars do not count toward max_remove when flagging outliers.  Any reserve star
        that is flagged as an outlier still shows up in the output file, but has flag_psf=1.
        You can decide whether or not you want to include it in any diagnostic tests you perform
        using the reserve stars.

    :param thresh:          The threshold in chisq above which an object is declared an outlier.
    :param ndof:            The threshold as a multiple of the model's dof.
    :param prob:            The probability limit that a chisq distribution with the model's dof
                            would exceed the given value.
    :param nsigma:          The number of sigma equivalent for the probability that a chisq
                            distribution would exceed the given value.
    :param max_remove:      The maximum number of outliers to remove on each iteration.  If this
                            is a float < 1.0, then this is interpreted as a maximum fraction of
                            stars to remove.  e.g. 0.01 will remove at most 1% of the stars.
                            [default: None]
    :param logger:          A logger object for logging debug info. [default: None]
    """
    _type_name = 'Chisq'

    def __init__(self, thresh=None, ndof=None, prob=None, nsigma=None, max_remove=None,
                 include_reserve=None, logger=None):
        if all( (thresh is None, ndof is None, prob is None, nsigma is None) ):
            raise TypeError("One of thresh, ndof, prob, or nsigma is required.")
        if thresh is not None and any( (ndof is not None, prob is not None, nsigma is not None) ):
            raise TypeError("Only one of thresh, ndof, prob, or nsigma may be given.")
        if ndof is not None and any( (prob is not None, nsigma is not None) ):
            raise TypeError("Only one of thresh, ndof, prob, or nsigma may be given.")
        if prob is not None and nsigma is not None:
            raise TypeError("Only one of thresh, ndof, prob, or nsigma may be given.")
        if include_reserve is not None:
            logger = galsim.config.LoggerWrapper(logger)
            logger.error("WARNING: include_reserve is no longer used.")

        # The only one of these we can convert now is nsigma, which we can convert into prob.
        # Going from either prob or ndof to thresh requires knowledge of dof.
        if nsigma is not None:
            prob = math.erfc(nsigma / 2**0.5)

        self.thresh = thresh
        self.ndof = ndof
        self.prob = prob
        self.max_remove = max_remove

        self.kwargs = {
            'thresh' : thresh,
            'ndof' : ndof,
            'prob' : prob,
            'max_remove' : max_remove,
        }

    def _get_thresh(self, dof):
        if self.thresh is not None:
            return self.thresh
        elif self.ndof is not None:
            return self.ndof * dof
        else:
            return chi2.isf(self.prob, dof)

    def removeOutliers(self, stars, logger=None):
        """Remove outliers from a list of stars based on their chisq values.

        :param stars:       A list of Star instances
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: stars, nremoved   A new list of stars with outliers flagged, and how many outliers
                                    were flagged.
        """
        logger = galsim.config.LoggerWrapper(logger)

        # First figure out the threshold we actually want to use given max_remove.

        # These are the only stars we want to use to figure out the threshold
        # But note that we apply the threshold to everything eventually.
        use_stars = [star for star in stars if not star.is_flagged and not star.is_reserve]
        nstars = len(use_stars)
        logger.info("Checking %d stars for outliers", nstars)

        chisq = np.array([s.fit.chisq for s in use_stars])
        dof = np.array([s.fit.dof for s in use_stars])

        # Scale up threshold by global chisq/dof.
        factor = np.sum(chisq) / np.sum(dof)
        if factor < 1: factor = 1

        thresh = np.array([self._get_thresh(d) for d in dof]) * factor

        if np.all(dof == dof[0]):
            logger.info("dof = %f, thresh = %f * %f = %f",
                         dof[0], self._get_thresh(dof[0]), factor, thresh[0])
        else:
            min_dof = np.min(dof)
            max_dof = np.max(dof)
            min_thresh = self._get_thresh(min_dof)
            max_thresh = self._get_thresh(max_dof)
            logger.info("Minimum dof = %d with thresh = %f * %f = %f",
                         min_dof, min_thresh, factor, min_thresh*factor)
            logger.info("Maximum dof = %d with thresh = %f * %f = %f",
                         max_dof, max_thresh, factor, max_thresh*factor)

        nremoved = np.sum(~(chisq <= thresh))  # Write it as not chisq <= thresh in case of nans.

        if nremoved == 0:
            # Flag any reserve stars that need it.
            stars = [s.flag_if(s.is_reserve
                               and not(s.fit.chisq <= self._get_thresh(s.fit.dof) * factor)
                              )
                     for s in stars]
            return stars, 0

        logger.info("Found %d stars with chisq > thresh", nremoved)

        # Update max_remove if necessary
        max_remove = self.max_remove
        if max_remove is not None and 0 < max_remove < 1:
            max_remove = int(math.ceil(max_remove * len(use_stars)))

        # Remake the chisq, etc. with all the stars now.
        all_chisq = np.array([s.fit.chisq for s in stars])
        all_dof = np.array([s.fit.dof for s in stars])
        all_thresh = np.array([self._get_thresh(d) for d in all_dof]) * factor
        good = all_chisq <= all_thresh

        if max_remove is None or nremoved <= max_remove:
            stars = [s.flag_if(not g) for s,g in zip(stars, good)]
        else:
            # Since the thresholds are not necessarily all equal, this might be tricky to
            # figure out which ones should be removed.
            # e.g. if max_remove == 1 and we have items with
            #    chisq = 20, thresh = 15
            #    chisq = 40, thresh = 32
            # which one should we remove?
            # The first has larger chisq/thresh, and the second has larger chisq - thresh.
            # I semi-arbitrarily remove based on the difference.
            nremoved  = max_remove
            diff = chisq - thresh
            new_thresh_index = np.argpartition(diff, -nremoved)[-nremoved]
            new_thresh = diff[new_thresh_index]
            all_diff = all_chisq - all_thresh
            good = all_diff < new_thresh
            stars = [s.flag_if(not g) for s,g in zip(stars, good)]

        logger.debug("chisq = %s",all_chisq[~good])
        logger.debug("thresh = %s",all_thresh[~good])
        logger.debug("flux = %s",[s.flux for s,g in zip(stars,good) if not g])

        return stars, nremoved
