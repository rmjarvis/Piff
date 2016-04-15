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

from __future__ import print_function

from .interp import Interp
import numpy
import warnings
from numpy.polynomial.polynomial import polyval2d
from numpy.polynomial.chebyshev import chebval2d
from numpy.polynomial.legendre import legval2d
from numpy.polynomial.laguerre import lagval2d
from numpy.polynomial.hermite import hermval2d

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
    """
    def __init__(self, orders, poly_type="poly"):
        """Create a Polynomial interpolator.

        :param orders:  List/array of integers, one for each parameter 
                        to be interpolated. The maximum total order of the 
                        polynomial for that parameter; i.e. the maximum values
                        of i+j where p(x,y) = sum c^{ij} x^i * y^j
        :param poly_type: A string, one of the keys in the polynomial_types
                          dictionary. By default these are "poly" (ordinary 
                          polynomials), "chebyshev", "legendre", "laguerre",
                          "hermite". To add more you can add a key to 
                          polynomial_types with the value of a function with
                          the signature of numpy.polynomial.polynomial.polyval2d

        """
        self._set_orders(orders)
        self._set_function(poly_type)
        self.coeffs = None

    def _set_orders(self, orders):
        """An internal function that sets up the indices and orders of the 

        :param orders:  List/array of integers, one for each parameter 
                        to be interpolated. The maximum total order of the 
                        polynomial for that parameter; i.e. the maximum values
                        of i+j where p(x,y) = sum c^{ij} x^i * y^j

        """
        self.orders=orders
        self.indices = [self._generate_indices(order) for order in self.orders]
        self.nvariables = [len(indices) for indices in self.indices]        
        self.nparam=len(orders)

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
        for p in xrange(order+1):
            for i in xrange(p+1):
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

        :param parameter_index: The integer index of the parameter; the lets us 
                                find the order of the parameter from self.
        :param C:          A 2D matrix of polynomial coefficients in the form that
                           the numpy polynomial form is expecting:
                           p(x,y,c) = sum_{i,j} c_{ij} x^i y^j
        :returns coeffs:    A 1D numpy array of coefficients of length self.nvariable
        """
        coeffs = numpy.zeros(self.nvariables[parameter_index])
        for k,(i,j) in enumerate(self.indices[parameter_index]):
            coeffs[k] = C[i,j]
        return coeffs


    def _unpack_coefficients(self, parameter_index, coeffs):
        """Unpack a sequence of parameters into the 2D matrix for the 
        given parameter_index (which determines the order of the matrix)

        This function is the inverse of _pack_coefficients
                           
        :param parameter_index: The integer index of the parameter being used
        :param coeffs:     A 1D numpy array of coefficients  of length self.nvariable
        :returns:          A 2D matrix of polynomial coefficients in the form that
                           the numpy polynomial form is expecting:
                           p(x,y,c) = sum_{i,j} c_{ij} x^i y^j

        """
        k=0
        n=self.orders[parameter_index]+1
        C = numpy.zeros((n, n))
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
        :param pos:        A numpy array of the u,v positions at which to build 
                           the model
        :param C:          A 2D matrix of polynomial coefficients in the form that
                           the numpy polynomial form is expecting:
                           p(x,y,c) = sum_{i,j} c_{ij} x^i y^j
                           
        :returns:          A numpy array of the calculated p_x(x)*p_y(y) where 
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

        :param positions:  A list of positions ((u,v) in this case) of stars.
        :param parameter:  A numpy array of the measured values of a parameter
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
        n = self.orders[parameter_index]+1
        C = numpy.zeros((n,n))
        C[0,0] = parameter.mean()
        return C



    def solve(self, pos, vectors, logger=None):
        """Solve for the interpolation coefficients given some data,
        using the scipy.optimize.curve_fit routine, which uses Levenberg-Marquardt
        to find the least-squares solution.

        This currently assumes that our positions pos are just u and v.

        :param pos:         A list of positions to use for the interpolation.
        :param vectors:     A list of parameter vectors (numpy arrays) for each star.
        :param logger:      A logger object for logging debug info. [default: None]
        """
        import scipy.optimize

        # We will want to index things later, so useful
        # to convert these to numpy arrays and transpose
        # them to the order we need.
        parameters = numpy.array(vectors).T
        positions = numpy.array(pos).T
       
        # We should have the same number of parameters as number of polynomial 
        # orders with which we were created here.
        nparam = len(parameters)
        if nparam!=self.nparam:
            raise ValueError("Must create Polynomial interpolator with the"
                "same order as the input vectors ({}!={})".format(nparam,
                self.nparam))

        if logger:
            logger.info("Fitting %d parameter vectors using "\
                "polynomial type %s with %d positions",
                nparam,self.poly_type)

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

            if logger:
                logger.debug("Fitting parameter %d from initial guess %s "\
                    "with polynomial order %d", i, p0, self.orders[i])


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

    def writeSolution(self, fits, extname):
        """Read the solution from a FITS binary table.

        We save two columns for the exponents and one column
        of coefficients for each parameter.

        :param fits:        An open fitsio.FITS object.
        :param extname:     The name of the extension with the interp information.
        """

        # We will try to be as explicit as possible when saving the 
        # coefficients to file - for each coefficient we spell out in
        # full the parameter index and exponent it corresponds to.
        # We don't actually use this information in the readSolution
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


        #
        for p in xrange(self.nparam):
            # This is a bit ugly, but we still have to tell self
            # what parameter we are using so the system knows the
            # order of the parameter. Hmm.
            # Now we pack the coeffecients into a 1D vector
            coeffs = self._pack_coefficients(p, self.coeffs[p])
            n = len(coeffs)
            # And build up the columns we will be saving.
            param_col.append(numpy.repeat(p, n))
            u_exponent_col.append([ind[0] for ind in self.indices[p]])
            v_exponent_col.append([ind[1] for ind in self.indices[p]])
            coeff_col.append(coeffs)

        # This is all the table data we'll actually be saving.
        cols = [numpy.concatenate(c) for c in (param_col, u_exponent_col, v_exponent_col, coeff_col)]

        # We will need some more identifying information that this!
        # I would suggest some kind of standard piff collection of header
        # values.
        header={"NPARAM":self.nparam, "POLYTYPE":self.poly_type}
        for i,order in enumerate(self.orders):
            header["ORDER_{}".format(i)] = order

        # Finally, write all of this to a FITS table.
        data = numpy.array(zip(*cols), dtype=dtypes)
        fits.write_table(data, extname=extname, header=header)

    def readSolution(self, fits, extname):
        """Read the solution from a FITS binary table.

        The extension should contain the same values as are saved
        in the writeSolution method.

        :param fits:        An open fitsio.FITS object.
        :param extname:     The name of the extension with the interp information.
        """
        header = fits[extname].read_header()
        data = fits[extname].read()

        # Load the same standard header variables that we saved above.
        # Must keep these in sync
        self.nparam = header['NPARAM']
        poly_type = header['POLYTYPE'].strip()
        orders = [header["ORDER_{}".format(p)] for p in xrange(self.nparam)]

        #Configure self - same methods that are run in __init__
        self._set_function(poly_type)
        self._set_orders(orders)


        # Finally load coefficients from the FITS file.
        # Although we have saved the u and exponents in another
        # column we don't actually use them here - we just use the fact
        # that we know the ordering was made by _pack_coefficients and so
        # we can re-order using _unpack_coefficients
        param_indices = data['PARAM']
        coeff_data = data['COEFF']
        self.coeffs = []
        for p in xrange(self.nparam):
            this_param_range = param_indices==p
            col = coeff_data[this_param_range]
            self.coeffs.append(self._unpack_coefficients(p,col))


    def interpolate(self, pos, logger=None):
        """Perform the interpolation to find the interpolated parameter vector at some position.

        :param pos:         The position to which to interpolate.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: the parameter vector (a numpy array) interpolated to the given position.
        """
        p = [self._interpolationModel(pos, coeff) for coeff in self.coeffs]
        return numpy.array(p)
