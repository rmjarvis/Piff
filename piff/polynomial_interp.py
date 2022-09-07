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
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

"""
.. module:: interp_mean
"""

import galsim
import numpy as np
import warnings
from numpy.polynomial.polynomial import polyval2d
from numpy.polynomial.chebyshev import chebval2d
from numpy.polynomial.legendre import legval2d
from numpy.polynomial.laguerre import lagval2d
from numpy.polynomial.hermite import hermval2d

from .interp import Interp
from .star import Star, StarFit

polynomial_types = {
    "poly":polyval2d,
    "chebyshev":chebval2d,
    "legendre":legval2d,
    "laguerre":lagval2d,
    "hermite":hermval2d,
}


class Polynomial(Interp):
    """
    An interpolator that uses  scipy curve_fit command to fit a polynomial
    surface to each parameter passed in independently.

    :param order:       The maximum order in the polynomial. i.e. the maximum
                        value of i+j where p(u,v) = sum c_{ij} x^i y^j.
                        [required, unless orders is given]
    :param orders:      Optionally, a list of orders, one for each parameter
                        to be interpolated.  This list should be the same length
                        as the number of parameters that will be given to
                        interpolate.
    :param poly_type:   A string, one of the keys in the polynomial_types
                        dictionary. By default these are "poly" (ordinary
                        polynomials), "chebyshev", "legendre", "laguerre",
                        "hermite". To add more you can add a key to
                        polynomial_types with the value of a function with
                        the signature of numpy.polynomial.polynomial.polyval2d
    """
    def __init__(self, order=None, orders=None, poly_type="poly", logger=None):
        if order is None and orders is None:
            raise TypeError("Either order or orders is required")
        if order is not None and orders is not None:
            raise TypeError("Cannot provide both order and orders")
        self.degenerate_points = False
        self.order = order
        self.orders = orders
        self._set_function(poly_type)
        self.coeffs = None

        self.kwargs = {
            'order' : order,
            'orders' : orders,
            'poly_type' : poly_type
        }

    def _setup_indices(self, nparam):
        """An internal function that sets up the indices, given the number of parameters
        """
        if self.orders is not None:
            if nparam != len(self.orders):
                raise ValueError("The given orders list has the wrong number of values")
            self._orders = self.orders
        else:
            self._orders = [self.order] * nparam
        self.indices = [self._generate_indices(order) for order in self._orders]
        self.nvariables = [len(indices) for indices in self.indices]
        self.nparam=len(self._orders)

    def _set_function(self, poly_type):
        """An internal function that sets the type of the polynomial
        interpolation used. The options are the keys in polynomial_types.

        :param poly_type:   A string value, one of the keys from polynomial_types
        """
        function = polynomial_types.get(poly_type)
        # Raise an error if this is not a valid type, in which case the lookup
        # in the line above will return None.
        if function is None:
            valid_types = ', '.join(polynomial_types.keys())
            raise ValueError(
                "poly_type argument must be one of: {}, not {}".format(
                valid_types, poly_type))
        # If all is valid, set the appropriate values on self.
        self.function=function
        self.poly_type=poly_type

        self.current_parameter=None

    def _generate_indices(self, order):
        """Generate, for internal use, the exponents i,j used in the polynomial model
        p(u,v) = sum c_{ij} u^i v^j

        This needs to be called whenever the order of the polynomial fit is
        changed. At the moment that is just when an object is initialized or
        updated from file.

        :param order:   The maximum order of the polynomial; the max value of
                        i+j where p(x,y) ~ x^i y^j
        """
        indices = []
        for p in range(order+1):
            for i in range(p+1):
                j = p-i
                indices.append((i,j))
        return indices

    def _pack_coefficients(self, parameter_index, C):
        """Pack the 2D matrix of coefficients used as the model fit parameters
        into a vector of coefficients, either so this can be passed as a starting
        point into the curve_fit routine or for serialization to file.

        For subclasses, the 2D matrix format could be whatever you wanted as long
        as _initialGuess, _interpolationModel, and the pack and unpack functions are
        consistent. The intialGuess method can return and the _interpolationModel can
        accept parameters in whatever form you like (e.g. could be a dict if you want)
        as long as _pack_coefficients can convert this into a 1D array and _unpack_coefficients
        convert it the other way.

        :param parameter_index: The integer index of the parameter; this lets us
                                find the order of the parameter from self.
        :param C:               A 2D matrix of polynomial coefficients in the form that
                                the numpy polynomial form is expecting:
                                p(x,y,c) = sum_{i,j} c_{ij} x^i y^j

        :returns coeffs:        A 1D numpy array of coefficients of length self.nvariable
        """
        coeffs = np.zeros(self.nvariables[parameter_index])
        for k,(i,j) in enumerate(self.indices[parameter_index]):
            coeffs[k] = C[i,j]
        return coeffs

    def _unpack_coefficients(self, parameter_index, coeffs):
        """Unpack a sequence of parameters into the 2D matrix for the
        given parameter_index (which determines the order of the matrix)

        This function is the inverse of _pack_coefficients

        :param parameter_index: The integer index of the parameter being used
        :param coeffs:          A 1D numpy array of coefficients  of length self.nvariable

        :returns:               A 2D matrix of polynomial coefficients in the form that
                                the numpy polynomial form is expecting:
                                p(x,y,c) = sum_{i,j} c_{ij} x^i y^j
        """
        k=0
        n=self._orders[parameter_index]+1
        C = np.zeros((n, n))
        for k,(i,j) in enumerate(self.indices[parameter_index]):
            C[i,j] = coeffs[k]
            k+=1
        return C

    def _interpolationModel(self, pos, C):
        """Generate the polynomial variation of some quantity at x and y
        coordinates for a given coefficient matrix.

        TODO: At the moment this function is expecting a numpy array of
        shape (2,nstar) for the positions. We might want to use a galsim
        position object instead since I think some code elsewhere in Piff
        is expecting this.

        This is an internal method used during the fitting.

        :param pos:     A numpy array of the u,v positions at which to build
                        the model
        :param C:       A 2D matrix of polynomial coefficients in the form that
                        the numpy polynomial form is expecting:
                        p(x,y,c) = sum_{i,j} c_{ij} x^i y^j

        :returns:       A numpy array of the calculated p_x(x)*p_y(y) where
                        the p functions are polynomials.
        """
        # Take the u and v components (x and y in the tangent plane)
        # as our interpolants
        u = pos[0]
        v = pos[1]
        # Call the appropriate function to generate the polynomial model.
        # By default this is numpy.polyval. Note that despite appearances
        # this is not a method call - function is a normal python attribute
        # that happens to be a function.
        f = self.function(u, v, C)
        return f

    def _initialGuess(self, positions, parameter, parameter_index):
        """Make an initial guess for a set of parameters
        to use in the fit for your model. This is passed
        to curve_fit as a starting point.

        :param positions:       A list of positions ((u,v) in this case) of stars.
        :param parameter:       A numpy array of the measured values of a parameter
                                for each star
        :param parameter_index: The integer index of the parameter being used

        :returns:          A guess for the parameters. In this case a 2D matrix
                           which is zero everywhere except for (0,0).  This should
                           correspond to a flat function of the parameters with
                           value given by the mean.
        """
        # We need a starting point for the fitter.
        # Use a constant value over the whole field as
        # a reasonable guess.
        n = self._orders[parameter_index]+1
        C = np.zeros((n,n))
        C[0,0] = parameter.mean()
        return C

    def initialize(self, stars, logger=None):
        """Initialization is just solving the interpolator with current stars.
        This then calls interpolateList, which will set the stars to have the
        right type of object in its star.fit.params attribute.

        :param stars:       A list of Star instances to use to initialize.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a new list of Star instances
        """
        parameters = np.array([s.fit.params for s in stars]).T
        positions = np.array([self.getProperties(s) for s in stars]).T
        nparam = len(parameters)
        self._setup_indices(nparam)
        self.coeffs = []
        for i, parameter in enumerate(parameters):
            p0 = self._pack_coefficients(i, self._initialGuess(positions, parameter, i))
            self.coeffs.append(self._unpack_coefficients(i,p0))
        return self.interpolateList(stars)

    def solve(self, stars, logger=None):
        """Solve for the interpolation coefficients given some data,
        using the scipy.optimize.curve_fit routine, which uses Levenberg-Marquardt
        to find the least-squares solution.

        This currently assumes that our positions pos are just u and v.

        :param stars:       A list of Star instances to use for the interpolation.
        :param logger:      A logger object for logging debug info. [default: None]
        """
        import scipy.optimize
        logger = galsim.config.LoggerWrapper(logger)

        # We will want to index things later, so useful
        # to convert these to numpy arrays and transpose
        # them to the order we need.
        parameters = np.array([s.fit.params for s in stars]).T
        positions = np.array([self.getProperties(s) for s in stars]).T

        # We should have the same number of parameters as number of polynomial
        # orders with which we were created here.
        nparam = len(parameters)
        npos = len(positions)
        self._setup_indices(nparam)

        logger.info("Fitting %d parameter vectors using "\
                    "polynomial type %s with %d positions",
                    nparam,self.poly_type,npos)

        coeffs = []

        # This model function adapts our _interpolationModel method
        # into the form that the scipy curve_fit function is expecting.
        # It just needs to unpack a linear exploded list of coefficients
        # into the matrix form that _interpolationModel wants.


        # Loop through the parameters
        for i, parameter in enumerate(parameters):

            def model(uv,*coeffs):
                C = self._unpack_coefficients(i, coeffs)
                return self._interpolationModel(uv,C)

            # Convert the structure the coefficients are held in into
            # a single parameter vector for scipy to fit.
            p0 = self._pack_coefficients(i, self._initialGuess(positions, parameter, i))

            logger.debug("Fitting parameter %d from initial guess %s "
                         "with polynomial order %d", i, p0, self._orders[i])


            # Black box curve fitter from scipy!
            # We may want to look into the tolerance and other parameters
            # of this function.
            # MJ: There are much faster ways to do this, but this is fine for now.
            with warnings.catch_warnings():
                # scipy.optimize has a tendency to emit warnings.  Let's ignore them.
                warnings.simplefilter("ignore", scipy.optimize.OptimizeWarning)
                p,covmat=scipy.optimize.curve_fit(model, positions, parameter, p0)

            # Build up the list of outputs, one for each parameter
            coeffs.append(self._unpack_coefficients(i,p))

        # Each of these is now a list of length nparam, each element
        # of which is a 2D array of coefficients to the corresponding
        # exponents. Where "corresponding" is as-defined in
        # self._unpack_coefficients
        self.coeffs = coeffs

    def _finish_write(self, fits, extname):
        """Write the solution to a FITS binary table.

        :param fits:        An open fitsio.FITS object.
        :param extname:     The base name of the extension
        """
        if self.coeffs is None:
            raise RuntimeError("Coeffs not set yet.  Cannot write this Polynomial.")

        # We will try to be as explicit as possible when saving the
        # coefficients to file - for each coefficient we spell out in
        # full the parameter index and exponent it corresponds to.
        # We don't actually use this information in the _finish_read
        # below, but when we want to generalize or plot things it
        # will be invaluable.
        dtypes = [('PARAM', int), ('U_EXPONENT', int), ('V_EXPONENT', int),
                  ('COEFF', float)]

        # We will build up the data columns parameter by parameter
        # and concatenate the results
        param_col = []
        u_exponent_col = []
        v_exponent_col = []
        coeff_col = []

        for p in range(self.nparam):
            # This is a bit ugly, but we still have to tell self
            # what parameter we are using so the system knows the
            # order of the parameter. Hmm.
            # Now we pack the coeffecients into a 1D vector
            coeffs = self._pack_coefficients(p, self.coeffs[p])
            n = len(coeffs)
            # And build up the columns we will be saving.
            param_col.append(np.repeat(p, n))
            u_exponent_col.append([ind[0] for ind in self.indices[p]])
            v_exponent_col.append([ind[1] for ind in self.indices[p]])
            coeff_col.append(coeffs)

        # This is all the table data we'll actually be saving.
        cols = [np.concatenate(c)
                for c in (param_col, u_exponent_col, v_exponent_col, coeff_col)]

        # nparam isn't one of the construction kwargs, so for convenience on reading,
        # put it in the header of this fits extension.
        header = { 'NPARAM' : self.nparam }

        # Finally, write all of this to a FITS table.
        data = np.array(list(zip(*cols)), dtype=dtypes)
        fits.write_table(data, extname=extname + '_solution', header=header)


    def _finish_read(self, fits, extname):
        """Read the solution from a fits file.

        :param fits:        An open fitsio.FITS object.
        :param extname:     The base name of the extension
        """
        # Read the solution extension.
        data = fits[extname + '_solution'].read()
        header = fits[extname + '_solution'].read_header()

        self.nparam = header['NPARAM']

        # Run setup functions to get these values right.
        self._set_function(self.poly_type)
        self._setup_indices(self.nparam)

        param_indices = data['PARAM']
        coeff_data = data['COEFF']

        self.coeffs = []
        for p in range(self.nparam):
            this_param_range = param_indices==p
            col = coeff_data[this_param_range]
            self.coeffs.append(self._unpack_coefficients(p,col))


    def interpolate(self, star, logger=None):
        """Perform the interpolation to find the interpolated parameter vector at some position.

        :param star:        A Star instance to which one wants to interpolate
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a new Star instance holding the interpolated parameters
        """
        pos = self.getProperties(star)
        p = [self._interpolationModel(pos, coeff) for coeff in self.coeffs]
        fit = star.fit.newParams(p)
        return Star(star.data, fit)
