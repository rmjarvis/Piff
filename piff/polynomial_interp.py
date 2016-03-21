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
from numpy.polynomial.polynomial import polyval2d
from numpy.polynomial.chebyshev import chebval2d
from numpy.polynomial.legendre import legval2d
from numpy.polynomial.laguerre import lagval2d
from numpy.polynomial.hermite import hermval2d

POLYNOMIAL_TYPES = {
    "POLY":polyval2d,
    "CHEBYSHEV":chebval2d,
    "LEGENDRE":legval2d,
    "LAGUERRE":lagval2d,
    "HERMITE":hermval2d,
}


class Polynomial(Interp):
    """
    An interpolator that uses  scipy curve_fit command to fit a polynomial 
    surface to each parameter passed in independently.

    TODO: This code is written so we can refactor to pull out the general
    bits that allow you to use curve_fit from the specific polynomial model
    used here.
    """
    def __init__(self, order, poly_type="POLY"):
        """Create 
        """
        self.order=order
        #The total number of variables per parameters.
        #We want to go up to a maximum total power of
        #"order", so 

        #Before any fitting is done we just None for the
        #coeffs
        self.coeffs = None
        self.generate_indices()
        self.set_function(poly_type)
        self.nvariable=len(self.indices)

    def set_function(self, poly_type):
        """An internal function that sets the type of the polynomial 
        interpolation used. The options are the keys in POLYNOMIAL_TYPES.

        :param poly_type:   A string value, one of the keys from POLYNOMIAL_TYPES
        """
        function = POLYNOMIAL_TYPES.get(poly_type)
        #Raise an error if this is not a valid type, in which case the lookup
        #in the line above will return None.
        if function is None:
            valid_types = ', '.join(POLYNOMIAL_TYPES.keys())
            raise ValueError(
                "poly_type argument must be one of: {}, not {}".format(
                valid_types, poly_type))
        #If all is valid, set the appropriate values on self.
        self.function=function
        self.poly_type=poly_type

    def generate_indices(self):
        """Generate, for internal use, the exponents i,j used in the polynomial model
        p(u,v) = sum c_{ij} u^i v^j

        This needs to be called whenever the order of the polynomial fit is 
        changed. At the moment that is just when an object is initialized or
        updated from file.
         """
        indices = []
        for p in xrange(self.order+1):
            for i in xrange(p+1):
                j = p-i
                indices.append((i,j))
        self.indices = indices

    def pack_coefficients(self, C):
        """Pack the 2D matrix of coefficients used as the model fit parameters
        into a vector of coefficients, either so this can be passed as a starting
        point into the curve_fit routine or for serialization to file.

        For subclasses, the 2D matrix format could be whatever you wanted as long
        as initialGuess, interpolationModel, and the pack and unpack functions are 
        consistent. The intialGuess method can return and the interpolationModel can
        accept parameters in whatever form you like (e.g. could be a dict if you want)
        as long as pack_coefficients can convert this into a 1D array and unpack_coefficients
        convert it the other way.

        :param C:          A 2D matrix of polynomial coefficients in the form that
                           the numpy polynomial form is expecting:
                           p(x,y,c) = sum_{i,j} c_{ij} x^i y^j
        :returns coeffs:    A 1D numpy array of coefficients of length self.nvariable
        """
        coeffs = numpy.zeros(self.nvariable)
        for k,(i,j) in enumerate(self.indices):
            coeffs[k] = C[i,j]
        return coeffs


    def unpack_coefficients(self, coeffs):
        """Unpack a sequence of parameters into the 2D matrix.

        This function is the inverse of pack_coefficients
                           
        :param coeffs:     A 1D numpy array of coefficients  of length self.nvariable
        :returns:          A 2D matrix of polynomial coefficients in the form that
                           the numpy polynomial form is expecting:
                           p(x,y,c) = sum_{i,j} c_{ij} x^i y^j

        """
        k=0
        C = numpy.zeros((self.order+1, self.order+1))
        for k,(i,j) in enumerate(self.indices):
                C[i,j] = coeffs[k]
                k+=1
        return C

    
    def interpolationModel(self, pos, C):
        """Generate the polynomial variation of some quantity at x and y
        coordinates for a given coefficient matrix.

        TODO: At the moment this function is expecting a numpy array of
        shape (2,nstar) for the positions. We might want to use a galsim
        position object instead since I think some code elsewhere in Piff
        is expecting this.

        This is an internal method used during the fitting.
        :param pos:         A numpy array of the u,v positions at which to build 
                           the model
        :param C:          A 2D matrix of polynomial coefficients in the form that
                           the numpy polynomial form is expecting:
                           p(x,y,c) = sum_{i,j} c_{ij} x^i y^j
                           
        :returns:          A numpy array of the calculated p_x(x)*p_y(y) where 
                           the p functions are polynomials.

        """
        #Take the u and v components (x and y in the tangent plane)
        #as our interpolants
        u = pos[0]
        v = pos[1]
        #Call the appropriate function to generate the polynomial model.
        #By default this is numpy.polyval. Note that despite appearances
        #this is not a method call - function is a normal python attribute
        #that happens to be a function.
        f = self.function(u, v, C)
        return f

    def initialGuess(self, positions, parameter):
        """Make an initial guess for a set of parameters
        to use in the fit for your model. This is passed
        to curve_fit as a starting point.

        :param positions:  A list of positions ((u,v) in this case) of stars.
        :param parameter:  A numpy array of the measured values of a parameter
                           for each star
                           
        :returns:          A guess for the parameters. In this case a 2D matrix
                           which is zero everywhere except for (0,0).  This should
                           correspond to a flat function of the parameters with 
                           value given by the mean.


        """        
        #We need a starting point for the fitter.
        #Use a constant value over the whole field as 
        #a reasonable guess.
        C = numpy.zeros((self.order+1, self.order+1))
        C[0,0] = parameter.mean()
        return C



    def solve(self, pos, vectors, logger=None):
        """Solve for the interpolation coefficients given some data.

        This currently assumes that our positions pos are just u and v.

        :param pos:         A list of positions to use for the interpolation.
        :param vectors:     A list of parameter vectors (numpy arrays) for each star.
        :param logger:      A logger object for logging debug info. [default: None]
        """
        import scipy.optimize

        #We will want to index things later, so useful
        #to convert these to numpy arrays and transpose
        #them to the order we need.
        parameters = numpy.array(vectors).T
        positions = numpy.array(pos).T
       
        #It seems like we can't get the number of parameters (which
        #depends on what model we have fitted) before we get to this
        # point. That's a little awkward.
        self.nparam = len(parameters)

        coeffs = []

        #This model function adapts our interpolationModel method
        #into the form that the scipy curve_fit function is expecting.
        #It just needs to unpack a linear exploded list of coefficients
        #into the matrix form that interpolationModel wants.
        def model(uv,*coeffs):
            C = self.unpack_coefficients(coeffs)
            return self.interpolationModel(uv,C)


        #Loop through the parameters
        for p, parameter in enumerate(parameters):
            #To replace the polynomial function in this code
            #with another model it should only be necessary to 
            #override the methods, initial_guess, fit_function,
            #and perhaps the pack and unpack methods
            p0 = self.pack_coefficients(self.initialGuess(positions, parameter))
            # Black boxes curve fitter from scipy!
            # We may want to look into the tolerance and other parameters
            # of this function.
            p,covmat=scipy.optimize.curve_fit(model, positions, parameter, p0)
            
            #Build up the list of outputs, one for each parameter
            coeffs.append(self.unpack_coefficients(p))

        #Each of these is now a list of length nparam, each element
        #of which is a 2D array of coefficients to the corresponding 
        #exponents. Where "corresponding" is as-defined in 
        #self.unpack_coefficients
        self.coeffs = coeffs

    def writeSolution(self, fits, extname):
        """Read the solution from a FITS binary table.

        We save two columns for the exponents and one column
        of coefficients for each parameter.

        :param fits:        An open fitsio.FITS object.
        :param extname:     The name of the extension with the interp information.
        """


        #First the exponents - we need arrays of the indices,
        #which we get with this piece of zip oddness which unzips
        #a list of pairs into a pair of lists
        cols = zip(*self.indices)
        #We also need to tell fitsio the data types of these.
        dtypes = [ ('u_exponent', int), ('v_exponent', int) ]

        #We will also need one column per parameter that we have
        #fit containing the coefficients for that parameter
        for p in xrange(self.nparam):
            dtypes.append(('coeff_{}'.format(p),float))
            #The pack function converts our format (which in 
            # this case is a 2D matrix) into a linear one.
            col = self.pack_coefficients(self.coeffs[p])
            cols.append(col)

        #We will need some more identifying information that this!
        #I would suggest some kind of standard piff collection of header
        #values.
        header={"NPARAM":self.nparam, "NVAR":self.nvariable, "ORDER":self.order, 
        "POLYTYPE":self.poly_type}
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

        #Load the same standard header variables that we saved above.
        #Must keep these in sync
        self.order = header['ORDER']
        self.nvariable = header['NVAR']
        self.nparam = header['NPARAM']
        poly_type = header['POLYTYPE'].strip()

        #Configure self - same methods that are run in __init__
        self.set_function(poly_type)
        self.generate_indices()
        #Finally load coefficients from the FITS file.
        self.coeffs = []
        for i in xrange(self.nparam):
            col = data["coeff_{}".format(i)]
            self.coeffs.append(self.unpack_coefficients(col))


    def interpolate(self, pos, logger=None):
        """Perform the interpolation to find the interpolated parameter vector at some position.

        :param pos:         The position to which to interpolate.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: the parameter vector (a numpy array) interpolated to the given position.
        """
        p = [self.interpolationModel(pos, self.coeffs[i]) for i in xrange(self.nparam)]
        return numpy.array(p)
