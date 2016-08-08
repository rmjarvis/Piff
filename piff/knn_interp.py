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
.. module:: interp_knn
"""

from .interp import Interp
from .star import Star, StarFit

import numpy as np
import warnings


class KNN(Interp):

	def __init__(self, k=15, algorithm='auto', d=2, w='uniform', logger=None):
		"""Create a Polynomial interpolator.

		:param k:           Number of neighbors to look at.
		:param d:           Distance formula for neighbors. Default 2 uses Euclidean distance, 1 uses
							cityblock distance. Important when weighting based on distance.
		:param algorithm:   Algorithm to be used by the interpolator.
		:param w:           Whether or not to weight neighbors based on distance. Can be take values
							'uniform' and 'distance', where uniform does not weight stars, and
							'distance' weights based on distance from stars..
		"""

		self.k = k
		self.d = d
		self.algorithm = algorithm
		self.w = w
		self.degenerate_points = False

		self.kwargs = {
			"k": k,
			"d": d,
			"algorithm": algorithm,
			"w": w
		}
		
		from sklearn.neighbors import KNeighborsRegressor

		knr = KNeighborsRegressor(n_neighbors=self.k, p=self.d, algorithm=self.algorithm, weights=self.w)
		self.knr = knr

	def initialize(self, stars, logger=None):
		"""Initialization is just solving the interpolator with current stars.
		This then calls interpolateList, which will set the stars to have the
		right type of object in its star.fit.params attribute.

		:param stars:       A list of Star instances to use to initialize.
		:param logger:      A logger object for logging debug info. [default: None]

		:returns: a new list of Star instances
		"""
		self.solve(stars, logger=None)
		return self.interpolateList(stars)

	def solve(self, stars, logger=None):

		parameters = [s.fit.params for s in stars]
		positions = [(s['u'],s['v']) for s in stars]

		self.parameters = parameters
		self.positions = positions

		self.knr.fit(positions,parameters)

	def interpolate(self, star, logger=None):
		"""Perform the interpolation to find the interpolated parameter vector at some position.

		:param star:        A Star instance to which one wants to interpolate
		:param logger:      A logger object for logging debug info. [default: None]

		:returns: a new Star instance holding the interpolated parameters
		"""
		
		return self.interpolateList([star], logger=None)[0]

	def interpolateList(self, stars, logger=None):
		positions = [(s['u'],s['v']) for s in stars]
		parameters = self.knr.predict(positions)

		star_list_fitted = []

		for pars, star in zip(parameters, stars):
			if star.fit == None:
				fit = StarFit(pars)
			else:
				try:
					fit = StarFit(pars)
				except TypeError:
					fit = StarFit(pars)
			star_list_fitted.append(Star(star.data, fit))

		return star_list_fitted

	def _finish_write(self, fits, extname):
		"""Write the solution to a FITS binary table.
		Save the knn params and the positions and parameters arrays
		:param fits:        An open fitsio.FITS object.
		:param extname:     The base name of the extension with the interp information.
		"""

		self.positions = np.array(self.positions)
		self.parameters = np.array(self.parameters)
		dtypes = [('POSITIONS', self.positions.dtype, self.positions.shape),
				  ('PARAMETERS', self.parameters.dtype, self.parameters.shape),
				  ]
		data = np.empty(1, dtype=dtypes)
		# assign
		data['POSITIONS'] = self.positions
		data['PARAMETERS'] = self.parameters

		# write to fits
		fits.write_table(data, extname=extname + '_solution')

	def _finish_read(self, fits, extname):
		"""Read the solution from a FITS binary table.
		:param fits:        An open fitsio.FITS object.
		:param extname:     The base name of the extension with the interp information.
		"""
		data = fits[extname + '_solution'].read()

		# self.positions and self.parameters assigned in _fit
		self._fit(data['POSITIONS'][0], data['PARAMETERS'][0])

