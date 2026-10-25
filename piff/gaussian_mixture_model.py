import numpy as np
import galsim

from .model import Model
from .star import Star


class GaussianMixture(Model):
    """A unit-flux mixture of shifted elliptical Gaussians.

    The Gaussian mixture is defined by 6 arrays: normalized weights, sigmas, g1, g2, du, dv.
    For sheared components, the sigmas use the usual GalSim definition. I.e. it is the size of
    the round Gaussian profile, which is sheared by (g1,g2) using the reduced shear convention.

    .. warning::

        This class currently does not implement any fitting machinery. It is currently
        only possible to use this with the Fixed interpolant.

    :param weights: Relative nonnegative component fluxes.
    :param sigmas:  Component Gaussian sigmas (arcsec).
    :param g1:      Component reduced-shear g1 values. [default: all zero]
    :param g2:      Component reduced-shear g2 values. [default: all zero]
    :param du:      Component shift in u direction (arcsec). [default: all zero]
    :param dv:      Component shift in v direction (arcsec). [default: all zero]
    :param logger:  Optional logger. [default: None]
    """
    _type_name = 'GaussianMixture'
    _method = 'auto'
    # Component shifts are fixed; stellar offsets are still fitted separately.
    _centered = True
    _model_can_be_offset = False

    def __init__(self, weights, sigmas, g1=None, g2=None, du=None, dv=None, logger=None):
        weights = np.asarray(weights, dtype=float)
        sigmas = np.asarray(sigmas, dtype=float)
        if weights.ndim != 1 or len(weights) == 0 or sigmas.shape != weights.shape:
            raise ValueError('weights and sigmas must be nonempty vectors of equal length')
        if np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError('weights must be nonnegative and have positive sum')
        if np.any(sigmas <= 0):
            raise ValueError('sigmas must be positive')
        g1 = np.zeros_like(weights) if g1 is None else np.asarray(g1, dtype=float)
        g2 = np.zeros_like(weights) if g2 is None else np.asarray(g2, dtype=float)
        if g1.shape != weights.shape or g2.shape != weights.shape:
            raise ValueError('g1 and g2 must have one value per component')
        if np.any(g1**2 + g2**2 >= 1):
            raise ValueError('component reduced shears must have |g| < 1')
        du = np.zeros_like(weights) if du is None else np.asarray(du, dtype=float)
        dv = np.zeros_like(weights) if dv is None else np.asarray(dv, dtype=float)
        if du.shape != weights.shape or dv.shape != weights.shape:
            raise ValueError('du and dv must have one value per component')
        self.weights = weights / weights.sum()
        self.sigmas = sigmas
        self.g1 = g1
        self.g2 = g2
        self.du = du
        self.dv = dv
        self.kwargs = dict(weights=self.weights, sigmas=self.sigmas, g1=self.g1, g2=self.g2,
                           du=self.du, dv=self.dv)
        self.set_num(None)

    @classmethod
    def _fix_kwargs(cls, kwargs):
        # FITS reads length-one array columns as scalars for a single-component mixture.
        for name in ['weights', 'sigmas', 'g1', 'g2', 'du', 'dv']:
            kwargs[name] = np.atleast_1d(kwargs[name])

    def initialize(self, star, logger=None, default_init=None):
        """Initialize with the configured mixture, independently of the image data."""
        params = np.concatenate([self.weights, self.sigmas, self.g1, self.g2, self.du, self.dv])
        return Star(star.data, star.fit.newParams(params, num=self._num))

    def getProfile(self, params):
        """Construct the unit-flux mixture from its parameter vector."""
        weights, sigmas, g1, g2, du, dv = np.split(params, 6)
        components = [galsim.Gaussian(sigma=s, flux=w).shear(g1=_g1, g2=_g2).shift(_du,_dv)
                      for w, s, _g1, _g2, _du, _dv in zip(weights, sigmas, g1, g2, du, dv)]
        return galsim.Sum(components)

    def fit(self, star, logger=None, convert_func=None, draw_method=None):
        """GaussianMixture currently requires a Fixed interpolator."""
        raise NotImplementedError('GaussianMixture currently requires interp: Fixed')
