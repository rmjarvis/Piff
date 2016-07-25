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

from __future__ import print_function
import math
import numpy as np
from scipy.stats import chi2

from .util import write_kwargs, read_kwargs

class Outliers(object):
    """The base class for handling outliers.

    This is essentially an abstract base class intended to define the methods that should be
    implemented by any derived class.
    """
    @classmethod
    def process(cls, config_outliers, logger=None):
        """Parse the outliers field of the config dict.

        :param config_outliers: The configuration dict for the outliers field.
        :param logger:          A logger object for logging debug info. [default: None]

        :returns: an Outliers instance
        """
        import piff

        if 'type' not in config_outliers:
            raise ValueError("config['outliers'] has no type field")

        # Get the class to use for the outliers
        # Not sure if this is what we'll always want, but it would be simple if we can make it work.
        outliers_class = getattr(piff, config_outliers.pop('type') + 'Outliers')

        # Read any other kwargs in the outliers field
        kwargs = outliers_class.parseKwargs(config_outliers, logger)

        # Build outliers object
        outliers = outliers_class(**kwargs)

        return outliers

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
        return kwargs

    def write(self, fits, extname):
        """Write an Outliers to a FITS file.

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension to write the outliers information.
        """
        # First write the basic kwargs that works for all Outliers classes
        outliers_type = self.__class__.__name__
        write_kwargs(fits, extname, dict(self.kwargs, type=outliers_type))

        # Now do any class-specific steps.
        self._finish_write(fits, extname)

    def _finish_write(self, fits, extname):
        """Finish the writing process with any class-specific steps.

        The base class implementation doesn't do anything, which is often appropriate, but
        this hook exists in case any Outliers classes need to write extra information to the
        fits file.

        :param fits:        An open fitsio.FITS object
        :param extname:     The base name of the extension
        """
        pass

    @classmethod
    def read(cls, fits, extname):
        """Read a Outliers from a FITS file.

        :param fits:        An open fitsio.FITS object
        :param extname:     The name of the extension with the outliers information.

        :returns: an Outliers handler
        """
        import piff

        assert extname in fits
        assert 'type' in fits[extname].get_colnames()
        outliers_type = fits[extname].read()['type']
        assert len(outliers_type) == 1
        outliers_type = outliers_type[0]

        # Check that outliers_type is a valid Outliers type.
        outliers_classes = piff.util.get_all_subclasses(piff.Outliers)
        valid_outliers_types = dict([ (c.__name__, c) for c in outliers_classes ])
        if outliers_type not in valid_outliers_types:
            raise ValueError("outliers type %s is not a valid Piff Outliers"%outliers_type)
        outliers_cls = valid_outliers_types[outliers_type]

        kwargs = read_kwargs(fits, extname)
        kwargs.pop('type')
        outliers = outliers_cls(**kwargs)
        outliers._finish_read(fits, extname)
        return outliers

    def _finish_read(self, fits, extname):
        """Finish the reading process with any class-specific steps.

        The base class implementation doesn't do anything, which is often appropriate, but
        this hook exists in case any Outliers classes need to read extra information from the
        fits file.

        :param fits:        An open fitsio.FITS object.
        :param extname:     The base name of the extension.
        """
        pass

class MADOutliers(Outliers):
    """An Outliers handler using mean absolute deviation (MAD) for defining outliers.

        MAD = < |x - median] >

    The range of values to keep in the sample are

        [median - nmad * MAD, median + nmad * MAD]

    where nmad is a parameter specified by the user.

    The user can specify this parameter in one of two ways.  

        1. The user can specify nmad directly.
        2. The user can specify nsigma, in which case nmad = sqrt(pi/2) nsigma, the equivalent
           value for a Gaussian distribution.
    """
    def __init__(self, nmad=None, nsigma=None):
        """
        Either nmad or nsigma must be provided.

        :param nmad:        The number of mean absolute deviations from the median at which
                            something is declared an outlier.
        :param nsigma:      The number of sigma equivalent if the underlying distribution is
                            Gaussian.
        """
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
        raise NotImplemented("MAD algorithm not implemented")


class MedOutliers(Outliers):
    #Find and remove outliers using Median Absolute Deviation:
        #MedAD = median(abs(datapoint-median(datapoints)))
    def __init__(self, nmad):
        self.nmad = nmad
        self.kwargs = {
            'nmad' : nmad
        }

    def removeOutliers(self, stars, logger=None):
        params = np.array([s.fit.params for s in stars])
        m = np.median(params, axis=0)
        mad = np.median(np.abs(params - m), axis=0)
        # import pdb; pdb.set_trace()
        check = np.all((params >= m - self.nmad * mad) * (params <= m + self.nmad * mad), axis=1)
        allstar = [s[0] for s in zip(stars, check) if s[1]]
        nremoved = len(stars) - len(allstar)
        return allstar,nremoved

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
    """
    def __init__(self, thresh=None, ndof=None, prob=None, nsigma=None, max_remove=None):
        """
        Exactly one of thresh, ndof, nsigma, prop must be provided.

        :param thresh:      The threshold in chisq above which an object is declared an outlier.
        :param ndof:        The threshold as a multiple of the model's dof.
        :param prob:        The probability limit that a chisq distribution with the model's dof
                            would exceed the given galue.
        :param nsigma:      The number of sigma equivalent for the probability that a chisq
                            distribution would exceed the given value.
        :param max_remove:  The maximum number of outliers to remove on each iteration.
                            [default: None]
        """
        if all( (thresh is None, ndof is None, prob is None, nsigma is None) ):
            raise TypeError("One of thresh, ndof, prob, or nsigma is required.")
        if thresh is not None and any( (ndof is not None, prob is not None, nsigma is not None) ):
            raise TypeError("Only one of thresh, ndof, prob, or nsigma may be given.")
        if ndof is not None and any( (prob is not None, nsigma is not None) ):
            raise TypeError("Only one of thresh, ndof, prob, or nsigma may be given.")
        if prob is not None and nsigma is not None:
            raise TypeError("Only one of thresh, ndof, prob, or nsigma may be given.")

        # The only one of these we can convert now is nsigma, which we can convert into prob.
        # Going from either prob or ndof to thresh requires knowledge of dof.
        if nsigma is not None:
            prob = 2. * math.erfc(nsigma)

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

        :returns: stars, nremoved   A new list of stars without outliers, and how many outliers
                                    were removed.
        """
        nstars = len(stars)
        if logger:
            logger.debug("Checking %d stars for outliers", nstars)

        chisq = np.array([ s.fit.chisq for s in stars ])
        dof = np.array([ s.fit.dof for s in stars ])

        thresh = np.array([ self._get_thresh(d) for d in dof ])

        if logger:
            if np.all(dof == dof[0]):
                logger.debug("dof = %f, thresh = %f",dof[0],thresh[0])
            else:
                min_dof = np.min(dof)
                max_dof = np.max(dof)
                logger.debug("Minimum dof = %d with thresh = %f",min_dof,self._get_thresh(min_dof))
                logger.debug("Maximum dof = %d with thresh = %f",max_dof,self._get_thresh(max_dof))

        nremoved = np.sum(chisq > thresh)

        if logger:
            logger.debug("Found %d stars with chisq > thresh", nremoved)
            logger.debug("chisq = %s",chisq[chisq > thresh])
            logger.debug("thresh = %s",thresh[chisq > thresh])

        if nremoved == 0:
            good_stars = stars
        elif self.max_remove is None or nremoved <= self.max_remove:
            good = chisq <= thresh
            good_stars = [ s for g, s in zip(good, stars) if g ]
        else:
            # Since the thresholds are not necessarily all equal, this might be tricky to
            # figure out which ones should be removed.
            # e.g. if max_remove == 1 and we have items with 
            #    chisq = 20, thresh = 15
            #    chisq = 40, thresh = 32
            # which one should we remove?
            # The first has larger chisq/thresh, and the second has larger chisq - thresh.
            # I semi-arbitrarily remove based on the difference.
            nremoved  = self.max_remove
            diff = chisq - thresh
            new_thresh_index = np.argpartition(diff, -nremoved)[-nremoved]
            new_thresh = diff[new_thresh_index]
            good = diff < new_thresh
            good_stars = [ s for g, s in zip(good, stars) if g ]

        assert nremoved == len(stars) - len(good_stars)
        return good_stars, nremoved

