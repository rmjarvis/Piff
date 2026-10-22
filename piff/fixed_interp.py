import numpy as np

from .mean_interp import Mean


class Fixed(Mean):
    """An interpolator that just keeps the initial parameter vector fixed without
    doing any fitting to the data.

    Use ``type: Fixed`` in the interpolator configuration.
    SimplePSF skips model-parameter fitting for this interpolator.
    """
    _type_name = 'Fixed'
    fit_parameters = False

    @property
    def property_names(self):
        return ()

    def initialize(self, stars, logger=None, transform_params=None, reconstruct_params=None):
        """Preserve the initial parameter vector for all stars.

        The initialed model parameters must match for all stars. This is checked.
        A ValueError is raised if any star is found to have different initial params.
        """
        self.mean = stars[0].fit.get_params(self._num)
        if any(not np.array_equal(s.fit.get_params(self._num), self.mean) for s in stars):
            raise ValueError('Fixed requires the same initial parameters for all stars')
        return self.interpolateList(stars)

    def solve(self, stars, logger=None):
        """Keep the initial values rather than estimating any interpolation coefficients."""
        pass
