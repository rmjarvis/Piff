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
.. module:: interp
"""

import numpy as np
import scipy.linalg
import galsim
import warnings

from .interp import Interp
from .star import Star
from . import _piff

try: 
    import jax
    from jax import jit
    from jax import numpy as jnp
    from jax import vmap

except ImportError:
    CAN_USE_JAX = False
    # define dummy functions for jax
    def jit(f):
        return f
else:
    CAN_USE_JAX = True
    jax.config.update("jax_enable_x64", True)

# Bellow are implementations of _solve_direct using JAX.
# if jax.config.update("jax_enable_x64", True) it will give the
# same results as the original code in double precision, but will run
# slower, but still faster than the numpy/scipy version.

@jit
def jax_solve(ATA, ATb):
    # Original code:
    # dq = scipy.linalg.solve(ATA, ATb, assume_a='pos', check_finite=False)
    # New code:
    (factor, lower) = (jax.scipy.linalg.cholesky(ATA, overwrite_a=True, lower=False), False)
    dq = jax.scipy.linalg.cho_solve((factor, lower), ATb, overwrite_b=False)
    return dq

@jit
def build_ATA_ATb(alpha, beta, K):
    ATb = (beta[:, jnp.newaxis] * K).flatten()
    tmp1 = alpha[:, :, jnp.newaxis] * K
    ATA = K[jnp.newaxis, :, jnp.newaxis, jnp.newaxis] * tmp1[:, jnp.newaxis, :, :]
    return ATA, ATb

@jit
def vmap_build_ATA_ATb(Ks, alphas, betas):
    # Use vmap to vectorize build_ATA_ATb across the first dimension of Ks, alphas, and betas
    vmapped_build_ATA_ATb = vmap(build_ATA_ATb, in_axes=(0, 0, 0))
    # Get the vectorized results
    ATAs, ATbs = vmapped_build_ATA_ATb(alphas, betas, Ks)
    # Sum the results along the first axis
    ATb = jnp.sum(ATbs, axis=0)
    ATA = jnp.sum(ATAs, axis=0)
    return ATA, ATb


class BasisInterp(Interp):
    r"""An Interp class that works whenever the interpolating functions are
    linear sums of basis functions.  Does things the "slow way" to be stable to
    degenerate fits to individual stars, instead of fitting to parameter sets
    produced by single stars.

    First time coding this we will assume that each element of the PSF parameter
    vector p is a linear combination of the same set of basis functions across the
    focal plane,

    .. math::

        p_i = \sum_{j} q_{ij} K_j(u,v,other stellar params).

    The property degenerate_points is set to True to indicate that this interpolator
    uses the alpha/beta quadratic form of chisq for each sample, rather than assuming
    that a best-fit parameter vector is available at every sample.

    Internally we'll store the interpolation coefficients in a 2d array of dimensions
    (nparams, nbases)

    Note: This is an abstract base class.  The concrete class you probably want to use
    is BasisPolynomial.
    """
    _type_name = None

    def __init__(self):
        self.degenerate_points = True  # This Interpolator uses chisq quadratic forms
        self.solver = "scipy"  # The default.  May be overridden by subclasses.
        self.q = None
        self.set_num(None)

    def initialize(self, stars, logger=None):
        """Initialize both the interpolator to some state prefatory to any solve iterations and
        initialize the stars for use with this interpolator.

        This class will initialize everything to have constant PSF parameter vector taken
        from the first Star in the list.

        :param stars:       A list of Star instances to use to initialize.
        :param logger:      A logger object for logging debug info. [default: None]

        :returns:           A new list of Stars which have their parameters initialized.
        """
        c = np.mean([s.fit.get_params(self._num) for s in stars], axis=0)
        self.q = c[:,np.newaxis] * self.constant(1.)[np.newaxis,:]
        stars = self.interpolateList(stars)
        return stars

    def basis(self, star):
        """Return 1d array of polynomial basis values for this star

        :param star:   A Star instance

        :returns:      1d numpy array with values of u^i v^j for 0<i+j<=order
        """
        raise NotImplementedError("Cannot call `basis` for abstract base class BasisInterp. "
                                  "You probably want to use BasisPolynomial.")

    def constant(self, value=1.):
        """Return 1d array of coefficients that represent a polynomial with constant value.

        :param value:  The value to use as the constant term.  [default: 1.]

        :returns:      1d numpy array with values of u^i v^j for 0<i+j<=order
        """
        raise NotImplementedError("Cannot call `constant` for abstract base class BasisInterp. "
                                  "You probably want to use BasisPolynomial.")

    def solve(self, stars, logger=None):
        """Solve for the interpolation coefficients given some data.
        The StarFit element of each Star in the list is assumed to hold valid
        alpha and beta members specifying depending of chisq on differential
        changes to its parameter vector.

        :param stars:       A list of Star instances to interpolate between
        :param logger:      A logger object for logging debug info. [default: None]
        """
        logger = galsim.config.LoggerWrapper(logger)
        if self.q is None:
            raise RuntimeError("Attempt to solve() before initialize() of BasisInterp")

        # The inputs to this function are for each star i the design equation is
        #
        #   A_i p = b_i
        #
        # The parameters at each stars location are modeled as
        #
        #   p_i = q . K_i(u_i,v_i,...)
        #
        # where K is the basis vector for the particular location of this star and q is
        # our 2d array of fitting parameters.  There is a row in q for each element of p,
        # and a column in q for each element in K.
        #
        # Each star then gives us nparam equations, which become rows of our full design matrix.
        # Consider the first equation for one star:
        #
        #   A[0,:] p = b[0]
        #   A[0,:] q K = b[0]
        #
        # Each element in the A[0,:]T KT outer product is the coefficient of a single element q_mn.
        # So we can rewrite this as
        #
        #   (A[0,:,np.newaxis] * K[np.newaxis,:]).flatten() * q.flatten() = b[0]
        #
        # Thus, the equation we want for our big design matrix has rows with
        #
        #   (A_i[j,:,np.newaxis] * K_i[np.newaxis,:]).flatten()
        #
        # and the b vector has elements
        #
        #   b_i[j]
        #
        # Now, the typical usage of this class is such that the size of A is
        #
        #   (nstars * npixels x nparam * nbasis)
        #
        # Typical numbers are:
        #
        #   nstars ~ 100
        #   npixels ~ 500
        #   nparam ~ 400
        #   nbasis ~ 6
        #
        # So the size of A is ~ 50,000 x 2,400, which for double precision is about 1 GB.
        # Considering that a particular CCD in a dense field might have up to 10 times this many
        # stars, this is a very steep memory demand for this function.
        #
        # Therefore, we make the default behavior be to construct AT A and solve AT A dq = AT b.
        # But we have an option to use QR decomposition of A instead, which may have stability
        # advantages, since the condition of AT A is the square of the condition of A, so the
        # QR method often has fewer numerical problems than the direct method for large matrices.
        # It just requires a lot of memory.

        logger = galsim.config.LoggerWrapper(logger)
        if self.solver == "qr":
            self._solve_qr(stars, logger)
        else:
            self._solve_direct(stars, logger)


    def _solve_qr(self, stars, logger):
        """The implementation of solve() when use_qr = True.
        """

        # First, build up one chunk for each star.  We'll stack them later.
        A_chunks = []
        b_chunks = []
        for s in stars:
            # Get the basis function values at this star
            K = self.basis(s)
            b_chunks.append(s.fit.b)
            A_chunks.append((s.fit.A[:,:,np.newaxis] * K[np.newaxis,:]).reshape(
                    s.fit.A.shape[0], s.fit.A.shape[1] * len(K)))

        # Now stack the chunks into a single A and b.
        A = np.vstack(A_chunks)
        b = np.concatenate(b_chunks)

        if len(stars) < len(K) or A.shape[0] < A.shape[1]:
            raise RuntimeError("Too few constraints for solution. (Probably too few stars)")

        # Note: The following snippet is the straightforward way to do this using the
        #       scipy qr function.  However, it generates the full Q matrix, which is slow.
        #       Using the lapack functions directly is slightly more obfuscated, but faster.
        #Q,R = scipy.linalg.qr(A, mode='economic', overwrite_a=True)
        #dq = Q.T.dot(b)
        #dq = scipy.linalg.solve_triangular(R, dq, overwrite_b=True, check_finite=False)

        # This computes A -> Q R, where QR are stored in a single matrix along with an
        # ancillary tau matrix that helps define the Householder matrices used to build
        # the real Q.
        QR, tau, work, info = scipy.linalg.lapack.dgeqrf(A)
        m = A.shape[1]

        # Check the diagonal values of the R matrix.  The following calculation isn't a
        # real condition number, since the regular QR decomposition isn't actually rank
        # revealing.  However, we only get problems with the normal QR decomposition if
        # this number is very small, in which case we should switch to a QRP decomposition.
        abs_Rdiag = np.abs(np.diag(QR))
        cond = np.min(abs_Rdiag) / np.max(abs_Rdiag)
        if cond < 1.e-12:
            # Note: this calculation is much slower, but it is safe to use even for
            # singular inputs, so it will always produce a valid answer.
            logger.info('Nominal condition is %s (min, max = %s, %s)', cond,
                        np.min(abs_Rdiag), np.max(abs_Rdiag))
            logger.info('Switching to QRP solution')
            QR, P, tau, work, info = scipy.linalg.lapack.dgeqp3(A, overwrite_a=True)
            P[:] -= 1  # Switch to python 0-based indexing.
            abs_Rdiag = np.abs(np.diag(QR))
            cond = np.min(abs_Rdiag) / np.max(abs_Rdiag)
            logger.info('Condition for QRP is %s (min, max = %s, %s)', cond,
                        np.min(abs_Rdiag), np.max(abs_Rdiag))
            # Skip any rows of R that have essentially 0 on the diagonal.
            k = np.sum(abs_Rdiag > 1.e-15 * np.max(abs_Rdiag))
            logger.debug('k = %d, m = %d',k,m)
        else:
            P = None
            k = m

        # The next steps are the same regardless of whether we pivoted or not.
        # This computes y = Q.T b
        dq, work, info = scipy.linalg.lapack.dormqr('L', 'T', QR, tau, b, work)
        # Cut dq down to the first m elements, since it is still the size of b here.
        dq = dq[:m]
        # Solve R dq = y (in place)
        scipy.linalg.lapack.dtrtrs(QR[:k,:k], dq[:k], overwrite_b=True)

        if P is not None:
            # Apply the permuation if we have one.
            dq1 = dq
            dq1[k:m] = 0.
            dq = np.empty(m)
            dq[P] = dq1[:m]

        logger.debug('...finished solution')
        # Reshape dq back into a 2d array and add it to the current solution.
        self.q += dq.reshape(self.q.shape)

    def _solve_direct(self, stars, logger):
        if self.solver == "cpp":
            self._solve_direct_cpp(stars, logger)
        else:
            self._solve_direct_python(stars, logger)

    def _solve_direct_cpp(self, stars, logger):
        """The implementation in C++ of solve() when use_qr = False.
        """

        Ks = []
        As = []
        bs = []
        for s in stars:
            # Get the basis function values at this star
            K = self.basis(s)
            Ks.append(K)
            As.append(s.fit.A)
            bs.append(s.fit.b)
        import pickle
        import glob
        dic = {"As": As,
               "bs": bs,
               "Ks": Ks}
        pkls = glob.glob('/sdf/data/rubin/user/leget/lsst_dev/tickets/DM-49086/As_bs_Ks*')
        fpkl = open(f'/sdf/data/rubin/user/leget/lsst_dev/tickets/DM-49086/As_bs_Ks_{len(pkls)}.pkl','wb')
        pickle.dump(dic, fpkl)
        fpkl.close()
        dq = _piff._solve_direct_cpp(bs, As, Ks)
        self.q += dq.reshape(self.q.shape)

    def _solve_direct_python(self, stars, logger):
        """The implementation in python of solve() when use_qr = False.
        """

        # Build ATA and ATb by accumulating the chunks for each star as we go.
        nq = np.prod(self.q.shape)
        ATA = np.zeros((nq, nq), dtype=float)
        ATb = np.zeros(nq, dtype=float)

        if self.solver == "jax":
            Ks = []
            alphas = []
            betas = []
            for s in stars:
                # Get the basis function values at this star
                K = self.basis(s)
                Ks.append(K)
                alphas.append(s.fit.alpha)
                betas.append(s.fit.beta)
            alphas = np.array(alphas).reshape((len(alphas), alphas[0].shape[0], alphas[0].shape[1]))
            betas = np.array(betas).reshape((len(betas), betas[0].shape[0]))
            Ks = np.array(Ks).reshape((len(Ks), Ks[0].shape[0]))
            import pickle
            import glob
            dic = {"alphas": alphas,
                   "betas": betas,
                   "Ks": Ks,
                   "nq": nq}
            pkls = glob.glob('/sdf/data/rubin/user/leget/lsst_dev/tickets/DM-49086/alphas_betas_ks*')
            fpkl = open(f'/sdf/data/rubin/user/leget/lsst_dev/tickets/DM-49086/alphas_betas_ks_{len(pkls)}.pkl','wb')
            pickle.dump(dic, fpkl)
            fpkl.close()
            ATA, ATb = vmap_build_ATA_ATb(Ks, alphas, betas)
            ATA = ATA.reshape(nq,nq)
        else:
            for s in stars:
                # Get the basis function values at this star
                K = self.basis(s)
                # Sum contributions into ATA, ATb

                if True:
                    alpha = s.fit.alpha
                    beta = s.fit.beta
                    ATb += (beta[:,np.newaxis] * K).flatten()
                    tmp1 = alpha[:,:,np.newaxis] * K
                    tmp2 = K[np.newaxis,:,np.newaxis,np.newaxis] * tmp1[:,np.newaxis,:,:]
                    ATA += tmp2.reshape(nq,nq)
                else:  # pragma: no cover
                    # This is equivalent, but slower.
                    # It is here to make more explicit the connection between this calculation
                    # and the corresponding part of the QR code above.
                    A1 = (s.fit.A[:,:,np.newaxis] * K[np.newaxis,:]).reshape(
                            s.fit.A.shape[0], s.fit.A.shape[1] * len(K))
                    ATb += A1.T.dot(s.fit.b)
                    ATA += A1.T.dot(A1)

        logger.info('Beginning solution of matrix size %s',ATA.shape)
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                # Note: It is usually ok to assume positive definite.  It is pretty rare that
                # assuming just 'sym' instead (which does an LDL decomposition rather than
                # Cholesky) would help.  If this fails, the matrix is usually high enough
                # condition that it is functionally singular, and switching to SVD is warranted.
                if self.solver == "jax":
                    dq = jax_solve(ATA, ATb)
                else:
                    dq = scipy.linalg.solve(ATA, ATb, assume_a='pos', check_finite=False)

            if len(w) > 0:
                # scipy likes to warn about high condition.  They aren't actually a problem
                # though, and in practice, we found in DES data that switching to SVD whenever
                # scipy thought the condition was high led to a significant increase in the mean
                # size rediduals.  We don't have a unit test that would catch this, so be careful
                # about changing the behavior of this part of the code!  For now, we just go to
                # the svd solution when ATA is fully singular.
                logger.info(w[0].message)
                logger.debug('norm(ATA dq - ATb) = %s',scipy.linalg.norm(ATA.dot(dq) - ATb))
                logger.debug('norm(dq) = %s',scipy.linalg.norm(dq))

        except (np.linalg.LinAlgError, scipy.linalg.LinAlgError) as e:
            logger.info('Caught %s',str(e))
            logger.info('Switching to svd solution')
            Sd,U = scipy.linalg.eigh(ATA)
            nsvd = np.sum(np.abs(Sd) > 1.e-15 * np.abs(Sd[-1]))
            logger.info('2-condition is %e',np.abs(Sd[-1]/Sd[0]))
            logger.info('nsvd = %d of %d',nsvd,len(Sd))
            # Note: unlike scipy.linalg.svd, the Sd here is in *ascending* order, not descending.
            Sd[-nsvd:] = 1./Sd[-nsvd:]
            Sd[:-nsvd] = 0.
            S = np.diag(Sd)
            dq = U.dot(S.dot(U.T.dot(ATb)))
            logger.info('norm(ATA dq - ATb) = %s',scipy.linalg.norm(ATA.dot(dq) - ATb))
            logger.info('norm(dq) = %s',scipy.linalg.norm(dq))
            logger.info('norm(q) = %s',scipy.linalg.norm(self.q))

        logger.debug('...finished solution')
        # Reshape dq back into a 2d array and add it to the current solution.
        self.q += dq.reshape(self.q.shape)

    def interpolate(self, star, logger=None):
        """Perform the interpolation to find the interpolated parameter vector at some position.

        :param star:        A Star instance to which one wants to interpolate
        :param logger:      A logger object for logging debug info. [default: None]

        :returns: a new Star instance holding the interpolated parameters
        """
        if self.q is None:
            raise RuntimeError("Attempt to interpolate() before initialize() of BasisInterp")

        K = self.basis(star)
        p = np.dot(self.q,K)
        fit = star.fit.newParams(p, num=self._num)
        return Star(star.data, fit)


class BasisPolynomial(BasisInterp):
    """A version of the Polynomial interpolator that works with BasisModels and can use the
    quadratic form of the chisq information it calculates.  It works better than the regular
    Polynomial interpolator when there is missing or degenerate information.

    The order is the highest power of a key to be used.  This can be the same for all keys
    or you may provide a list of separate order values to be used for each key.  (e.g. you
    may want to use 2nd order in the positions, but only 1st order in the color).

    All combinations of powers of keys that have total order <= max_order are used.
    The maximum order is normally the maximum order of any given key's order, but you may
    specify a larger value.  (e.g. to use 1, x, y, xy, you would specify order=1, max_order=2.)

    There are several options for what code to use for doing the linear algebra, controlled
    by the ``solver`` parameter, which can take one of the following values:

    1. "scipy" uses regular numpy array functionality to build the matrices, and then uses
       ``scipy.linalg.solve`` to find the solution.  It starts by assuming the matrix
       is positive definite, and it falls back to an SVD solution when that is not the case.
    2. "qr" will also use numpy to build the matrices, but then it uses QR decomposition
       for the solution rather than the more direct least squares solution.
       QR decomposition requires more memory than the default and is somewhat slower
       (nearly a factor of 2); however, it is significantly less susceptible to
       numerical errors from high condition matrices.
    3. "jax" uses the JAX module for building and solving the linear algebra equations
       rather than numpy/scipy. It should be equivalent in its results to the "scipy" option,
       but it may be faster if a multi-core cpu or gpu is available.
    4. "cpp" uses the Eigen linear algebra package in C++ to build and solve the linear
       algebra equations rather than numpy/scipy.  On a single core cpu (and more), it
       will be faster than default "scipy" solver if the number of training stars is more
       than ~30. With a Piff config using `PixelGrid` with ``size=25`` and ``interp="Lanczos(11)"``
       and a second order polynomial for interpolation, and running on ~O(100) PSFs reserved
       stars on a 4GB single core CPU, "cpp" solver is 60% faster than "scipy" solver.

    Use type name "BasisPolynomial" in a config field to use this interpolant.

    :param order:       The order to use for each key.  Can be a single value (applied to all
                        keys) or an array matching number of keys.
    :param keys:        List of keys for properties that will be used as the polynomial arguments.
                        [default: ('u','v')]
    :param max_order:   The maximum total order to use for cross terms between keys.
                        [default: None, which uses the maximum value of any individual key's order]
    :param solver:      Which solver to use.  Solvers available are "scipy", "qr", "jax",
                        "cpp". See above for details.
    :param logger:      A logger object for logging debug info. [default: None]
    """
    _type_name = 'BasisPolynomial'

    def __init__(
            self,
            order,
            keys=('u','v'),
            max_order=None,
            solver="scipy",
            use_qr=False,
            logger=None
        ):
        super(BasisPolynomial, self).__init__()

        logger = galsim.config.LoggerWrapper(logger)

        self._keys = keys
        if hasattr(order,'__len__'):
            if not len(order)==len(keys):
                raise ValueError('Number of provided orders does not match number of keys')
            self._orders = order
        else:
            self._orders = (order,) * len(keys)

        if max_order is None:
            self._max_order = np.max(self._orders)
        else:
            self._max_order = max_order

        self.solver = solver

        valid_solver = ["scipy", "qr", "jax", "cpp"]

        if solver not in valid_solver:
            raise ValueError(f"{solver} is not a valid solver. Valid solver are {valid_solver}")

        # To match old API when jax and cpp were not part of solving
        # basis interp.
        if use_qr and solver not in ["scipy", "qr"]:
            raise NotImplementedError(f"use_qr and {solver} are not compatible")
        if use_qr:
            logger.error("WARNING: use_qr=True is deprecated. "
                         "Use solver='qr' instead.")
            self.solver = "qr"

        if not CAN_USE_JAX and self.solver == "jax":
            logger.warning("JAX not installed. Reverting to numpy/scipy.")
            self.solver = "scipy"

        if self._max_order<0 or np.any(np.array(self._orders) < 0):
            # Exception if we have any requests for negative orders
            raise ValueError('Negative polynomial order specified')

        self.kwargs = {
            'order' : order,
            'keys' : keys,
            'solver': solver,
        }

        # Now build a mask that picks the desired polynomial products
        # Start with 1d arrays giving orders in all dimensions
        ord_ranges = [np.arange(order+1,dtype=int) for order in self._orders]
        # Nifty trick to produce n-dim array holding total order
        #sumorder = np.sum(np.ix_(*ord_ranges))  # This version doesn't work in numpy 1.19
        sumorder = np.sum(np.meshgrid(*ord_ranges, indexing='ij'), axis=0)
        self._mask = sumorder <= self._max_order

    def getProperties(self, star):
        return np.array([star.data[k] for k in self._keys], dtype=float)

    @property
    def property_names(self):
        """List of properties used by this interpolant.
        """
        return self._keys

    def basis(self, star):
        """Return 1d array of polynomial basis values for this star

        :param star:   A Star instance

        :returns:      1d numpy array with values of u^i v^j for 0<i+j<=order
        """
        # Get the interpolation key values
        vals = self.getProperties(star)
        # Make 1d arrays of all needed powers of keys
        pows1d = []
        for i,o in enumerate(self._orders):
            p = np.ones(o+1,dtype=float)
            p[1:] = vals[i]
            pows1d.append(np.cumprod(p))
        # Use trick to produce outer product of all these powers
        #pows2d = np.prod(np.ix_(*pows1d))
        pows2d = np.prod(np.meshgrid(*pows1d, indexing='ij'), axis=0)
        # Return linear array of terms making total power constraint
        return pows2d[self._mask]

    def constant(self, value=1.):
        """Return 1d array of coefficients that represent a polynomial with constant value.

        :param value:  The value to use as the constant term.  [default: 1.]

        :returns:      1d numpy array with values of u^i v^j for 0<i+j<=order
        """
        out = np.zeros( np.count_nonzero(self._mask), dtype=float)
        out[0] = value  # The constant term is always first.
        return out

    def _finish_write(self, writer):
        """Write the solution.

        :param writer:      A writer object that encapsulates the serialization format.
        """
        if self.q is None:
            raise RuntimeError("Solution not set yet.  Cannot write this BasisPolynomial.")

        dtypes = [ ('q', float, self.q.shape) ]
        data = np.zeros(1, dtype=dtypes)
        data['q'] = self.q
        writer.write_table('solution', data)

    def _finish_read(self, reader):
        """Read the solution.

        :param reader:      A reader object that encapsulates the serialization format.
        """
        data = reader.read_table('solution')
        assert data is not None
        self.q = data['q'][0]

