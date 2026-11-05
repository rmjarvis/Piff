# Copyright (c) 2026 by Mike Jarvis and the other collaborators on GitHub at
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

import numpy as np
import galsim
from scipy.fft import fftn, ifftn
from scipy.ndimage import uniform_filter

from ..input import InputFiles, ImageMetadata
from .gwcs import RomanGWCS


class RomanFiles(InputFiles):
    """Read calibrated Roman ASDF images and their ramp-fit noise information.

    This automatically reads the correct data arrays from the Roman image files
    without the need for explicit ext values for the mask, image, wcs, etc.

    It is recommended to use ``sky: sep`` to determine a good background to use
    both for sky subtraction and to determine the appropriate background variance
    for the weight map.

    Additional input options are:

    :ramp_fit:      ``optimal`` (default) uses the calibrated data and ERR variance.
                    ``uniform`` uses data + DUMO and infers read variance and gain
                    from clean optimal-ramp background pixels and the read pattern.
    :good_dq_bits:  Integer bit mask of DQ flags to allow. Default 0 rejects every
                    nonzero DQ value. To allow JUMP_DET and UNRELIABLE_SLOPE, use
                    ``good_dq_bits: 16777220`` (4 | 16777216).
    :smoothing_scale: Gaussian standard deviation in pixels for smoothing the uniform
                    gain and non-Poisson variance estimates. [default: 64]
    :mask_dumo_spikes: For uniform ramps, whether to reject DUMO values that spike
                    up to high values (>10 x ERR) at JUMP_DET pixels unless their
                    eight neighbor data pixels have summed S/N > 10. This empirical
                    screen is optional and only matters when JUMP_DET is allowed.
                    [default: False]

    Uniform-noise calibration selects pixels within twice ERR of the median in both
    ramp images. See `load_gain` and `estimate_uniform_noise` for more details.

    :param config:  The input configuration dictionary.
    :param base:    The base configuration dictionary. [default: None]
    :param logger:  Optional logger. [default: None]
    """
    _type_name = 'RomanFiles'
    _opt_params = InputFiles._opt_params | {
        'ramp_fit': str,
        'good_dq_bits': int,
        'smoothing_scale': float,
        'mask_dumo_spikes': bool,
    }

    def __init__(self, config, base=None, logger=None):
        for key in ['image_ext', 'weight_file_name', 'weight_ext',
                    'badpix_file_name', 'badpix_ext', 'fix_ushort_badpix',
                    'gain_file_name', 'gain_ext', 'gain', 'gain_col', 'noise', 'invert_weight']:
            if key in config:
                raise ValueError('RomanFiles constructs its own image, weight, mask, and gain; '
                                 '%s is not supported' % key)
        config = dict(config)
        config['image_ext'] = 'roman.data'
        config.setdefault('remove_signal_from_weight', True)
        super().__init__(config, base=base, logger=logger)

    def read_metadata(self, file_name, ext):
        """Read Roman image bounds and native WCS."""
        import asdf

        with asdf.open(file_name) as af:
            roman = af['roman']
            ny, nx = roman['data'].shape
            return ImageMetadata(galsim.BoundsI(1, nx, 1, ny),
                                 RomanGWCS(roman['meta']['wcs']), None)

    def load_image(self, data, logger):
        """Load Roman science and auxiliary ramp data from the ASDF file.

        Input data: image_num, config, base.
        This function sets: image, ramp_fit, optimal, dumo

        The latter 3 are used by later stages to compute the primary data products.
        """
        import asdf

        data.ramp_fit = self._parse_params(data, {'ramp_fit': 'optimal'})['ramp_fit']
        if data.ramp_fit not in ('optimal', 'uniform'):
            raise ValueError('ramp_fit must be optimal or uniform')
        file_name = self._parse_params(data, {'image_file_name': None})['image_file_name']
        logger.info('Reading Roman %s ramp image from %s', data.ramp_fit, file_name)
        with asdf.open(file_name) as af:
            roman = af['roman']
            data.optimal = np.array(roman['data'], dtype=np.float32)
            if data.ramp_fit == 'uniform':
                data.dumo = np.array(roman['dumo'], dtype=np.float32)
                image = data.optimal + data.dumo
            else:
                image = data.optimal
        data.image = galsim.Image(image, wcs=self.metadata_list[data.image_num].wcs)
        data.image.header = None
        data.base['_current_image'] = data.image

    def load_weight(self, data, logger):
        """Read ERR and photon variance, and derive the total non-Poisson variance.

        Input data: image_num, image, config, base.
        This function sets: err, var_rnoise, var_poisson
        """
        import asdf

        file_name = self._parse_params(data, {'image_file_name': None})['image_file_name']
        with asdf.open(file_name) as af:
            roman = af['roman']
            data.err = np.array(roman['err'], dtype=float)
            data.var_poisson = np.array(roman['var_poisson'], dtype=float)
        # Treat all non-Poisson contributions together as the effective total read noise.
        data.var_rnoise = data.err**2 - data.var_poisson

    def load_gain(self, data, logger):
        """Read DQ and derive the gain for the selected ramp fit.

        Input data: image, optimal, ramp_fit, err, var_rnoise, var_poisson, config, base.
        This function sets: gain_image, dq, and updates: var_rnoise.
    
        For ramp_fit=optimal, gain = optimal / var_poisson
        For ramp_fit=uniform, the procedure is more complicate. The gain and read_variance
        are inferred together from unflagged background pixels. See estimate_uniform_noise
        for details.

        var_rnoise represents all non-Poisson variance, without separating read noise
        from flat-field or other contributions.
        """
        import asdf

        file_name = self._parse_params(data, {'image_file_name': None})['image_file_name']
        with asdf.open(file_name) as af:
            roman = af['roman']
            data.dq = np.array(roman['dq'])
        if data.ramp_fit == 'optimal':
            gain = np.zeros(data.image.array.shape)
            np.divide(data.optimal, data.var_poisson, out=gain,
                      where=np.isfinite(data.var_poisson) & (data.var_poisson > 0))
        else:
            data.var_rnoise, gain = self.estimate_uniform_noise(data)
        data.gain_image = galsim.Image(gain, bounds=data.image.bounds)

    def build_mask(self, data, logger):
        """Build a mask of which pixels Piff should not use.

        Input data: image, weight, gain_image, optimal, dumo, err, var_poisson, dq,
                    ramp_fit, config, base
        This function sets: mask
        """
        super().build_mask(data, logger)
        options = self._parse_params(data, {'good_dq_bits': 0, 'mask_dumo_spikes': False})
        good_bits = options['good_dq_bits']
        if good_bits < 0 or good_bits > np.iinfo(np.uint32).max:
            raise ValueError('good_dq_bits must be an unsigned 32-bit bit mask')

        # Mask dq pixels not marked as "good_dq_bits".
        data.mask |= (data.dq & ~np.uint32(good_bits)) != 0

        # Mask any non-finite or otherwise invalid values.
        data.mask |= ~np.isfinite(data.optimal) | (data.optimal <= 0)
        data.mask |= ~np.isfinite(data.err) | (data.err <= 0)
        data.mask |= ~np.isfinite(data.var_poisson) | (data.var_poisson <= 0)

        # If requested mask DUMO pixels with clear spike values.
        # This looks for dumo pixels where:
        # 1. dq = JUMP_DET (=4)
        # 2. dumo > 10 * err (i.e. the pixel's own S/N > 10)
        # 3. total S/N in surrounding 8 pixels < 10
        # This is a rough proxy for pixels where the JUMP_DET is doing the right thing rather
        # than spuriously triggering due to the guider problems in commissioning data.
        if data.ramp_fit == 'uniform' and options['mask_dumo_spikes']:
            background = np.median(data.optimal[~data.mask])
            signal = np.where(data.mask, 0, data.optimal - background)
            variance = np.where(data.mask, 0, data.err**2)
            neighbor_signal = 9 * uniform_filter(signal, size=3, mode='constant') - signal
            neighbor_variance = 9 * uniform_filter(variance, size=3, mode='constant') - variance
            supported = neighbor_signal > 10 * np.sqrt(np.maximum(neighbor_variance, 0))
            data.mask |= ((data.dq & 4) != 0) & (data.dumo > 10 * data.err) & ~supported

    def finalize_weight(self, data, logger):
        """Finalize the weight image to be inverse variance weight without source poisson noise.

        Note: technically, the user can select remove_signal_from_weight=False to not remove
        the source Poisson noise, but this is not advised.

        Input data: image, var_rnoise, var_poisson, ramp_fit, mask, gain_image, sky, extra_props,
                    config, base.
        Uniform ramps also use: optimal, dumo.
        This function updates: weight, mask
        """
        params = self._parse_params(data, {'remove_signal_from_weight': True})
        gain = data.gain_image.array
        # Start with the read noise.
        variance = data.var_rnoise.copy()
        if params['remove_signal_from_weight']:
            # Add in the poisson noise from the background photons.
            sky = data.sky
            if 'sky' in data.extra_props:
                sky = np.mean(data.extra_props['sky'])
            if sky is not None:
                sky = sky.array if isinstance(sky, galsim.Image) else sky
                variance += sky / gain
            logger.verbose("Removed source variance.")
        elif data.ramp_fit == 'uniform':
            # Add in the poisson noise from all photons.
            variance += (data.optimal + data.dumo) / gain
        else:
            variance += data.var_poisson
        invalid = ~data.mask & (~np.isfinite(variance) | (variance <= 0))
        data.mask |= invalid
        if np.any(invalid):
            logger.warning("Masked %d pixels with invalid variance in Roman weight calculation",
                           np.count_nonzero(invalid))
        # weight is 1/variance
        # Do this carefully to keep masked pixels = 0.
        data.weight = galsim.Image(variance, bounds=data.image.bounds)
        data.weight.array[data.mask] = 0
        data.weight._invertSelf()
        data.weight._piff_default_weight = False

    def estimate_uniform_noise(self, data):
        """Estimate uniform-ramp noise and gain.

        Input data: image, optimal, err, var_rnoise, var_poisson, dq, config, base.
        Return estimate of var_rnoise, gain

        Uniform ramp fitting uses fixed slope coefficients. For illumination that is
        constant during the ramp, photon variance is proportional to the signal with a
        brightness-independent coefficient: variance = signal / gain. Bright sources
        therefore have the same effective gain as other nearby pixels. (We do see evidence
        of spatial structure in the gain map, so it's not just a single constant value.)
        Optimal fitting instead changes its coefficients with the photon/read-noise
        balance, so its effective gain varies significantly with brightness.

        The project supplies dumo (difference uniform minus optimal) to let us calculate
        the uniform ramp data value. But they don't provide the necessary information
        to directly determine either the gain or the noise that one would get with the
        uniform-weight ramp fits.

        They do, however, provide the read times for the ramp samples (the resultants).
        From this we can calculate the expected correlation between samples assuming
        uniform illumination during an exposure. We can propagate this covariance to
        estimate the variance in the measurements for a given weight vector.

        The project tells us the poisson noise variance on each pixel and the total error
        from which we can infer the read noise. (Technically, we use ERR^2 - VAR_POISSON,
        which includes all non-poisson sources of noise, including flat fielding noise.
        We take this all to be read noise, which is probably not 100% accurate.)

        We then make the assumption that the background pixels have uniform illumination
        (i.e. they are not affected by any guiding problems). From this we can estimate the
        optimal weight vector for a typical background pixel, which gives us a scaling relation
        for the read noise and gain between optimal and uniform weights.

        Finally, we use those scalings to get the read noise and gain for the background
        pixels and smooth/interpolate these to the rest of the pixels in the image.
        This is valid for uniform weights, since the read noise and gain are expected to be
        independent of the pixel flux.

        There are inline code comments about each step with further details about how
        this works, including the math involved.

        Returns the estimated read noise variance and gain as numpy arrays.
        """
        from astropy.convolution import Gaussian2DKernel, convolve_fft
        import asdf

        smoothing_scale = self._parse_params(data, {'smoothing_scale': 64})['smoothing_scale']
        if smoothing_scale <= 0:
            raise ValueError('smoothing_scale must be positive')

        # We first pick out pixels that are basically just background. Our calculations will
        # assume uniform illumination, which isn't true for bright pixels when there are
        # guiding issues (the main reason to use this). We do this by selecting pixels that
        # are within 2 sigma of the median for both data and data+dumo images.
        samples = data.dq == 0
        # DQ=0 excludes most problematic pixels, but weirdly some still have var_poisson=0.
        samples &= (data.optimal > 0) & (data.var_poisson > 0)
        if np.any(samples):
            background = np.median(data.optimal[samples])
            samples &= np.abs(data.optimal - background) < 2 * data.err
            uniform_background = np.median(data.image.array[samples])
            samples &= np.abs(data.image.array - uniform_background) < 2 * data.err
        if not np.any(samples):
            raise ValueError('No clean background samples for uniform-noise calibration')

        # Use the provided read times to calculate P.
        #
        # Each read is N(t_i) with expectation value E(N(t_i)) = f t_i, where f is the rate of
        # photon arrival, photons/sec. Its noise includes both read noise and poisson noise on
        # the photon counts. We take the read noise variance, g, to be constant for each read.
        #
        # The covariance between two reads (N(t_1) and N(t_2)) is simple -- they share the
        # photons that arrived before the earlier read, and not anything between then and the
        # later read:
        #
        #   Cov(N(t_i), N(t_j)) = f min(t_i, t_j) + delta_ij g
        #
        # Each resultant, R_a, is the average of several such reads at a set of times.
        # E.g. the first resultant might be the average of reads at times [1, 2]
        # Then the next could be at times [3, 4, 5], etc.
        #
        # The covariance between two resultants is then the average of the covariances of all
        # the pairs of reads they comprise:
        #
        #   Cov(R_a, R_b) = f Sum_(i in a) Sum_(j in b) min(t_i, t_j)/(n_a n_b) + delta_ab g/n_a
        #
        # We define P_ab and R_ab matrices such that
        #
        #   Cov(R_a, R_b) = f P_ab + g R_ab

        file_name = self._parse_params(data, {'image_file_name': None})['image_file_name']
        with asdf.open(file_name) as af:
            exposure = af['roman']['meta']['exposure']
            read_times = [np.array(t) * exposure['frame_time']
                          for t in exposure['read_pattern']]
        nreads = np.array([len(t) for t in read_times])
        times = np.array([np.mean(t) for t in read_times])
        R = np.diag(1.0 / nreads)
        P = np.array([[np.minimum(a[:, None], b[None, :]).mean()
                       for b in read_times]
                      for a in read_times])

        # For a given pixel with resultants R_i, the expected values along the ramp are
        # R_i = f t_i where t_i is the average of the individual reads for that resultant.
        # There may also be an initial value before the exposure starts, which we want to
        # ignore, so E(R_i) = b + f t_i in general.
        #
        # A uniform-weight fit of the slope, f, is uniform in the reads, but since resultants
        # can each have a different number of reads, we actually weight by n_i. This is just
        # the normal inverse-variance weighting, since the read noise variance is g/n_i.
        # The uniform here means we don't account for the varying poisson noise in the weights.
        #
        # The least squares solution can be written as:
        #
        #   f^ = Sum_i u_i R_i
        #
        #   u_i = n_i (t_i W - T) / (U W - T^2)
        #
        # where W = Sum_i n_i, T = Sum_i n_i t_i, and U = Sum_i n_i t_i^2.
        #
        # These weights can be used to calculate the poisson noise contribution to the
        # pixel variance. Along with the read noise, the total variance is:
        #
        #   Var(f^) = f u^T P u + g u^T R u

        W = nreads.sum()
        T = nreads @ times
        U = nreads @ times**2
        u = nreads * (times * W - T) / (U * W  - T**2)
        uTPu = u @ P @ u
        uTRu = u @ R @ u

        # The optimal weights are different depending on the brightness. Specifically,
        # they depend on the ratio f/g. For the background pixels we're considering here,
        # the average ratio is found from the read noise and poisson noise.
        # We make the approximation that the photon/read variance ratio is equal for optimal
        # and uniform fits, allowing us to estimate f/g using the uniform noise factors.
        # That's not exactly true, and we'll have better estimate of the difference by the
        # end of this process, but it's close enough for this starting value.
        ratio = np.median(data.var_poisson[samples] / data.var_rnoise[samples])
        q = ratio * uTRu / uTPu

        # The optimal weights for a uniform illumination can be thought of as finding the
        # minimum variance linear estimate of the slope between successive resultants.
        # I.e. the differences divided by the time. We make d as this difference operator, so
        # d @ resultants would be the slopes to be fit for a given pixel.
        d = np.diff(np.eye(len(times)), axis=0) / np.diff(times)[:, None]
        R_d = d @ R @ d.T
        P_d = d @ P @ d.T
        cov_d = R_d + q * P_d  # Ignore the overall factor of g, which drops out.

        # Solve for w given that the expectation value of the slopes are all equal.
        w = np.linalg.solve(cov_d, np.ones(len(d)))
        w /= w.sum()

        # This now gives us a scaling we can apply to convert var_rnoise and var_poisson from
        # the optimal weight to uniform.
        # However, we actually want the gain, which scales inversely with var_poisson for
        # these background images. We can then assume that gain pattern also applies to the
        # brighter objects for uniform weights. (That's not true for optimal.)
        R_scale = uTRu / (w @ R_d @ w)
        gain_scale = (w @ P_d @ w) / uTPu

        # The read noise and gain both seem to have real structure across the SCA, so we
        # don't just use a single mean value and apply it everywhere. Instead, use a
        # 2D Gaussian smoothing kernel from astropy to smooth the raw values over the whole
        # image and interpolate to the non-background pixels.
        # Also, we want the final maps to be read noise and gain. At fixed signal, gain is
        # inversely proportional to the photon variance. We choose to interpolate 1/gain
        # rather than gain, since that's the quantity we eventually use, but it probably
        # doesn't matter much either way.
        kernel = Gaussian2DKernel(smoothing_scale)
        kwargs = {
            'mask': ~samples,
            'fill_value': np.nan,
            'nan_treatment': 'interpolate',
            'preserve_nan': False,
            'min_wt': 1.e-6,
            'complex_dtype': np.complex64,
            'fftn': fftn,
            'ifftn': ifftn,
        }
        read_map = convolve_fft(data.var_rnoise, kernel, **kwargs)
        read_map *= R_scale
        inverse_gain = np.zeros(data.optimal.shape, dtype=np.float32)
        np.divide(data.var_poisson, data.optimal, out=inverse_gain, where=samples)
        inverse_gain = convolve_fft(inverse_gain, kernel, **kwargs)
        gain_map = gain_scale / inverse_gain
        return read_map, gain_map
