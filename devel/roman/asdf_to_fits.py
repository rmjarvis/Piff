#!/usr/bin/env python3
"""Convert calibrated Roman ASDF images to ordinary multi-extension FITS.

Simple usage
------------

python asdf_to_fits.py path/to/*.asdf

This creates fits files with the same name, but a .fits extension alongside the asdf files.

Options:

    --output-dir selects a different output directory for the fits files.
    --no-wcs skips the WCS conversion and does not put WCS information into the FITS files.
    --gain-map adds the GAIN HDU to the output.
    --background-weights uses sep to determine a good background to compute accurate weights.
    --uniform adds additional HDUs (see Uniform Weights below).
    --mesh-size sets a tile size in pixels for the uniform noise calibration. (default: 128)
    --overwrite lets the script replace existing output files.

Output HDUs
-----------

The output fits files include the following HDUs copied from the asdf data (whenever the
asdf file includes these arrays):

SCI: The science image. Called data in the asdf file. It is the "optimal" pixel values
     favored by the project. It includes the WCS information in the fits header.

DQ: Data quality flags. Most non-zero values indicate pixels to mask. Some that we
    think are ok (at least in early commissioning data) are JUMP_DET (4) and
    UNRELIABLE_SLOPE (16777216). We looked at LOW_QE (8192), and most of these are
    ok too, but they include a population of pixels that appear more like HOT
    pixels, so we reject this one.

ERR: An estimate of the error in the SCI values. It is a sigma value, so ERR^2 is the
     variance. It includes both read noise and poisson noise from source and background.

VAR_POISSON: The project estimate of the poisson noise. It includes noise from both
             background and sources.

VAR_RNOISE: The project estimate of the read noise.

VAR_FLAT: An estimate of the noise from flat field uncertainty.

CHISQ: The per-pixel ramp-fit chi-squared statistic. Not sure if this is useful.

DUMO: Difference Uniform Minus Optimal. It supplies a way to convert from the project "optimal"
      weight to a uniform weight ramp slope.


The following hdus are constructed from the above for convenience when used with Piff:

MASK: A mask plane usable by Piff. This converts the DQ values into either 1 or 0.
      1 for DQ values other than JUMP_DET and UNRELIABLE_SLOPE, 0 otherwise.
      We also set this to 1 for pixels where we obtain an invalid value for something
      such as the GAIN.

IVAR: A normal inverse variance weight that Piff prefers. If the option --background-weights
      is used, they will have the Poisson noise from sources removed. Note: for Piff, if you
      don't use --background-weights, then you should use `remove_signal_from_weight: true`
      to get Piff to remove the source Poisson noise from the variance. Otherwise, this
      script will already have done that for you.

SCI_UNIF: The sum of data + dumo; i.e. the uniform-weight ramp slope for each pixel.
          When using this, you probably also want to run with --uniform. See the next section.

GAIN: The ratio SCI / VAR_POISSON, which is the effective gain in each pixel. Because of how
      the pixel weighting and cosmic ray removal is done, each pixel effectively has a different
      gain. This map lets Piff use that information correctly.


Uniform Weights
---------------

The "optimal" weights give pretty bad results on early commissioning data with the guider
problems. Uniform turns out to work a lot better, so we recommend using SCI_UNIF rather
than the regular SCI. This may change when the guiding improves, but for the images with
very bad jitter, the uniform weight is a big improvement. However, to use it properly, we
also need to compute a few additional things.

The following additional HDUs are output when using --uniform:

MASK_UNIF: This keeps most of the JUMP_DET pixels, but masks isolated pixels marked
           as JUMP_DET with large DUMO values.

IVAR_UNIF: The inverse variance estimate for the UNIF image. This involves recalculating
           both the read variance and the background Poisson variance for uniform ramp weights,
           and it uses GAIN_UNIF rather than the normal GAIN when computing the Poisson noise
           from the background.

GAIN_UNIF: An estimate of the effective gain in the UNIF image.


World Coordinate System
-----------------------

The asdf file includes the WCS in GWCS format, which doesn't directly work in a FITS
file, so we convert it to a TAN-SIP WCS, targeting 0.01 pixel accuracy.
The sampled maximum residual is recorded in the FITS header of the SCI HDU as WCSERR.


Dependencies
------------

The code requires the following modules to be installed:

numpy, scipy, astropy, gwcs,
asdf, asdf-astropy, roman_datamodels

The ASDF-related packages need to be fairly recent to read the input files from Roman,
including roman_datamodels, which is used implicitly by these packages.

Additionally, sep is required for --uniform or --background-weights.
"""

import argparse
from pathlib import Path
import warnings

import asdf
from asdf.exceptions import AsdfConversionWarning
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_dilation, uniform_filter


def estimate_uniform_noise(science, read_variance, poisson_variance, samples, read_times,
                           *, mesh_size):
    """Infer uniform-ramp read variance and effective gain from clean background pixels.

    ``read_times`` contains individual read times in seconds, grouped by resultant.
    ``samples`` selects finite, positive, unflagged source-free measurements. Input
    variances describe the optimal full-ramp fit. The returned read variance and
    gain maps have the input image shape; metadata records the temporal factors and
    the number of calibration samples. The gain converts model intensity into
    uniform-ramp Poisson variance via ``model / gain``.

    We model the covariance of the calibrated resultants as A*R + C*P. R has
    diagonal 1/N_i for N_i independent reads averaged into resultant i. P averages
    min(t_a, t_b) over read pairs, since cumulative Poisson counts share all earlier
    arrivals. A is the calibrated per-read variance; C is the calibrated photon
    covariance amplitude for the local illumination. The image is a slope, so its
    read and photon variances also contain temporal factors from the slope weights.

    The optimal fitter's separate variance planes identify A and C by inverting
    the predicted variance ratio under full-ramp GLS. This assumes the unflagged
    optimal fit uses that covariance model. For the uniform slope coefficient
    vector w (sum(w)=0, w dot resultant_times=1), the new variances are
    A*(w^T R w) and C*(w^T P w). C is proportional to illumination, so C/SCI on
    background pixels provides an inverse gain amplitude. The returned gain also
    includes the uniform temporal factor; it is NOT the detector gain in e-/DN.
    Calibration/flat/photometric scaling is inherited from the calibrated planes.

    Only clean background samples are inverted: jump-affected optimal fits can use
    different read subsets, so their variances cannot reliably be propagated this
    way. Robust tile medians and interpolation supply amplitudes at sources and
    flagged pixels. This is an inferred noise model, not a rerun of calibration.
    """
    if mesh_size <= 0:
        raise ValueError('mesh_size must be positive')
    nreads = np.array([len(t) for t in read_times])
    times = np.array([np.mean(t) for t in read_times])
    read_covariance = np.diag(1.0 / nreads)
    photon_covariance = np.array([
        [np.minimum(a[:, None], b[None, :]).mean() for b in read_times]
        for a in read_times])
    total_reads = nreads.sum()
    nt = nreads @ times
    # Equal per-read weights become N_i weights for averaged resultants. These
    # slope coefficients cancel a constant intercept and give unit response to t.
    uniform = (total_reads * nreads * times - nt * nreads) / (
        total_reads * (nreads @ times**2) - nt**2)
    uniform_read = uniform @ read_covariance @ uniform
    uniform_photon = uniform @ photon_covariance @ uniform

    # Differencing removes the nuisance intercept. At each photon/read amplitude
    # ratio q=C/A, GLS predicts the reported slope variance ratio as
    # VAR_POISSON/VAR_READ = q * photon_factor(q)/read_factor(q). Tabulate this
    # monotonic relation so we can invert it for every background sample.
    difference = np.diff(np.eye(len(times)), axis=0) / np.diff(times)[:, None]
    read_diff = difference @ read_covariance @ difference.T
    photon_diff = difference @ photon_covariance @ difference.T
    log_ratio = np.linspace(-8 * np.log(10), 8 * np.log(10), 4097)
    ratio = np.exp(log_ratio)
    covariance = read_diff[None, :, :] + ratio[:, None, None] * photon_diff[None, :, :]
    weights = np.linalg.solve(covariance, np.ones((len(ratio), len(times) - 1, 1)))[..., 0]
    weights /= weights.sum(axis=1)[:, None]
    read_factor = np.einsum('ni,ij,nj->n', weights, read_diff, weights)
    photon_factor = np.einsum('ni,ij,nj->n', weights, photon_diff, weights)
    predicted_ratio = np.log(ratio * photon_factor / read_factor)

    # Infer calibration amplitudes only on clean background pixels, not on the
    # jump-affected stellar pixels whose optimal-fit uncertainties are unsuitable.
    ny, nx = science.shape
    x_edges = list(range(0, nx, mesh_size)) + [nx]
    y_edges = list(range(0, ny, mesh_size)) + [ny]
    x_centers = (np.array(x_edges[:-1]) + np.array(x_edges[1:]) - 1) / 2.0
    y_centers = (np.array(y_edges[:-1]) + np.array(y_edges[1:]) - 1) / 2.0
    mesh = np.empty((len(y_centers), len(x_centers), 2))
    for j, (y0, y1) in enumerate(zip(y_edges[:-1], y_edges[1:])):
        for i, (x0, x1) in enumerate(zip(x_edges[:-1], x_edges[1:])):
            region = np.s_[y0:y1, x0:x1]
            use = samples[region]
            if not np.any(use):
                raise ValueError(f'No clean background samples in tile ({i}, {j}); '
                                 'increase --mesh-size')
            read = read_variance[region][use]
            photon = poisson_variance[region][use]
            inferred_ratio = np.interp(np.log(photon / read), predicted_ratio, log_ratio)
            read_amplitude = read / np.interp(inferred_ratio, log_ratio, read_factor)
            photon_amplitude = photon / np.interp(inferred_ratio, log_ratio, photon_factor)
            mesh[j, i, 0] = np.median(read_amplitude)
            mesh[j, i, 1] = np.median(photon_amplitude) / np.median(science[region][use])

    # Interpolate inverse gain, so interpolation is linear in the noise amplitudes.
    # Clip at outer tile centers to avoid extrapolating to negative variances.
    interpolate = RegularGridInterpolator((y_centers, x_centers), mesh)
    read_map = np.empty(science.shape, dtype=np.float32)
    gain_map = np.empty(science.shape, dtype=np.float32)
    x = np.clip(np.arange(nx), x_centers[0], x_centers[-1])
    for start in range(0, ny, mesh_size):
        end = min(start + mesh_size, ny)
        y = np.clip(np.arange(start, end), y_centers[0], y_centers[-1])
        xx, yy = np.meshgrid(x, y)
        amplitudes = interpolate(np.stack([yy, xx], axis=-1))
        read_map[start:end] = amplitudes[:, :, 0] * uniform_read
        gain_map[start:end] = 1.0 / (amplitudes[:, :, 1] * uniform_photon)
    metadata = {
        'MESH': mesh_size,
        'NSAMP': int(np.count_nonzero(samples)),
        'UNIRFAC': float(uniform_read),
        'UNIPFAC': float(uniform_photon),
    }
    return read_map, gain_map, metadata


def source_mask(science, error, mask, background):
    """Find source footprints for noise calibration, using our SEP detection settings.

    ERR sets a reference threshold here, not the final uniform-ramp noise model.
    Use the union of optimal and uniform footprints so jumps and source cores in
    either estimate cannot enter the background calibration sample.
    """
    import sep
    image = np.array(science, dtype=np.float32, order='C')
    noise = np.array(error, dtype=np.float32, order='C')
    image[mask] = 0
    noise[mask] = 1
    image -= background
    sep.set_extract_pixstack(1000000)
    _, segmentation = sep.extract(
        image, 1.5, err=noise, mask=mask, minarea=4, deblend_nthresh=32,
        deblend_cont=0.005, filter_type='conv', segmentation_map=True)
    # Omitting filter_kernel uses SEP's compact built-in 3x3 detection filter.
    return segmentation != 0


def uniform_hdus(roman, header, original_mask, *, mesh_size):
    """Build the extra mask and noise planes for SCI_UNIF, without companion files."""
    science = np.asarray(roman['data'], dtype=np.float32)
    error = np.asarray(roman['err'], dtype=np.float32)
    poisson = np.asarray(roman['var_poisson'], dtype=np.float32)
    dumo = np.asarray(roman['dumo'], dtype=np.float32)
    uniform = science + dumo
    dq = np.asarray(roman['dq'])
    background = header['SKYBKG']

    # Guiding makes many real stellar cores JUMP_DET. Rejecting that bit wholesale
    # destroys the stars. Instead flag a >10 ERR positive DUMO excursion only when
    # its eight SCI neighbors lack 10-sigma summed support. Exclude the tested
    # pixel itself and masked neighbors: an isolated spike cannot support itself.
    # This empirical screen can retain a real jump on a bright source; it is not
    # a calibrated cosmic-ray classifier. WARM/LOW_QE stay in original_mask.
    signal = np.where(original_mask, 0, science - background)
    variance = np.where(original_mask, 0, error**2)
    neighbor_signal = 9 * uniform_filter(signal, size=3, mode='constant') - signal
    neighbor_variance = 9 * uniform_filter(variance, size=3, mode='constant') - variance
    supported = neighbor_signal > 10 * np.sqrt(np.maximum(neighbor_variance, 0))
    added_mask = ~original_mask & ((dq & 4) != 0) & (dumo > 10 * error) & ~supported
    mask = original_mask | added_mask | ~np.isfinite(uniform)

    sources = source_mask(science, error, original_mask, background)
    sources |= source_mask(uniform, error, mask, background)
    sources = binary_dilation(sources, iterations=3)
    # ERR^2 includes source AND background Poisson noise. Remove that entire
    # component (and flat variance if supplied) to isolate read variance. Never
    # clip negative estimates into the calibration sample: use finite positive,
    # completely unflagged background pixels only, then interpolate amplitudes.
    if 'var_rnoise' in roman:
        read = np.asarray(roman['var_rnoise'], dtype=float)
        read_source = 'VAR_RNOISE'
    else:
        read = error.astype(float)**2 - poisson
        if 'var_flat' in roman:
            read -= np.asarray(roman['var_flat'])
        read_source = 'ERR^2 minus other variances'
    samples = ~mask & ~sources & (dq == 0)
    samples &= np.isfinite(science) & (science > 0)
    samples &= np.isfinite(read) & (read > 0) & np.isfinite(poisson) & (poisson > 0)
    exposure = roman['meta']['exposure']
    read_times = [np.asarray(t) * exposure['frame_time'] for t in exposure['read_pattern']]
    read_map, gain, info = estimate_uniform_noise(
        science, read, poisson, samples, read_times, mesh_size=mesh_size)
    # Add back ONLY the background photon variance. Source noise must be added
    # later from the fitted source model, not the observed noisy pixel intensity.
    variance = read_map + background / gain
    ivar = np.zeros_like(read_map)
    np.divide(1.0, variance, out=ivar, where=~mask)

    mask_header = header.copy()
    mask_header['DUMOSIG'] = (10.0, 'Minimum positive DUMO/ERR for rejection')
    mask_header['SUPPSNR'] = (10.0, 'Required SCI neighbor S/N to retain excursion')
    mask_header.add_history('Empirical DUMO screen; no science pixel values replaced.')
    noise_header = header.copy()
    noise_header.update(info)
    noise_header['WGTTYPE'] = ('BACKGROUND', 'IVAR excludes source Poisson variance')
    noise_header['RNMETHOD'] = read_source
    noise_header['NOISEMOD'] = 'UNFLAGGED_GLS'
    noise_header['SRCDIL'] = (3, 'Segmentation exclusion buffer in pixels')
    noise_header.add_history('Uniform noise inferred from clean optimal-fit variances.')
    noise_header.add_history('GAIN_UNIF assumes stationary illumination for source noise.')
    print(f'Uniform: {added_mask.sum()} additional masked pixels; {info["NSAMP"]} '
          f'clean samples; median background variance={np.median(variance[~mask]):.7g}, '
          f'median gain={np.median(gain[~mask]):.7g}', flush=True)
    return [fits.ImageHDU(mask.astype(np.int16), mask_header, name='MASK_UNIF'),
            fits.ImageHDU(ivar, noise_header, name='IVAR_UNIF'),
            fits.ImageHDU(gain, noise_header, name='GAIN_UNIF')]


def convert(input_path, output_path, *, include_wcs=True, overwrite=False, include_gain=False,
            background_weights=False, uniform_weights=False, mesh_size=128):
    """Write one calibrated Roman image to FITS, returning the output path."""
    background_weights = background_weights or uniform_weights
    include_gain = include_gain or background_weights
    input_path = Path(input_path)
    output_path = Path(output_path)
    # Reject unrecognized tags, but allow package-version provenance warnings
    # when the installed readers can deserialize all objects successfully.
    with (warnings.catch_warnings(action='error', category=AsdfConversionWarning),
          asdf.open(input_path) as af):
        roman = af['roman']
        if 'data' not in roman:
            raise ValueError(f'{input_path} has no science image; use a *_cal.asdf file')
        if uniform_weights and 'dumo' not in roman:
            raise ValueError('--uniform requires the DUMO plane')
        meta = roman['meta']
        header = fits.Header()
        header['TELESCOP'] = meta['telescope']
        header['INSTRUME'] = meta['instrument']['name']
        header['DETECTOR'] = meta['instrument']['detector']
        header['FILTER'] = meta['instrument']['optical_element']
        header['EXPTIME'] = (meta['exposure']['exposure_time'], 'Exposure time in seconds')
        header['EFFEXPTM'] = meta['exposure']['effective_exposure_time']
        header['ORIGFILE'] = input_path.name
        header.add_history('Converted from Roman ASDF; pixel values copied without rescaling.')

        image_header = header.copy()
        if include_wcs:
            ny, nx = roman['data'].shape
            gwcs = meta['wcs']
            image_header.update(gwcs.to_fits_sip(
                bounding_box=((0, nx - 1), (0, ny - 1)),
                max_pix_error=0.01, max_inv_pix_error=0.01))
            # Check a separate grid, including the science image edges. GWCS uses
            # zero-based pixels; so does Astropy with origin=0 below.
            x, y = np.meshgrid(np.linspace(0, nx - 1, 17), np.linspace(0, ny - 1, 17))
            ra, dec = gwcs(x, y)
            fit_x, fit_y = WCS(image_header).all_world2pix(ra, dec, 0)
            error = np.max(np.hypot(fit_x - x, fit_y - y))
            if not np.isfinite(error) or error > 0.01:
                raise ValueError(f'FITS WCS residual {error:.6g} exceeds 0.01 pixel')
            image_header['WCSERR'] = (float(error), 'Max pixel residual on 17x17 check grid')
            image_header.add_history('TAN-SIP approximation to original ASDF GWCS.')
            print(f'{input_path.name}: maximum sampled WCS error = {error:.6g} pixel')

        hdus = [fits.PrimaryHDU(header=header)]
        for key, name in [('data', 'SCI'), ('dq', 'DQ'), ('err', 'ERR'),
                          ('var_poisson', 'VAR_POISSON'), ('var_rnoise', 'VAR_RNOISE'),
                          ('var_flat', 'VAR_FLAT'), ('chisq', 'CHISQ'), ('dumo', 'DUMO')]:
            if key in roman:
                data = np.asarray(roman[key])
                if data.dtype == np.float16:
                    data = data.astype(np.float32)
                hdus.append(fits.ImageHDU(data=data, header=image_header.copy(), name=name))
        error = np.asarray(roman['err'], dtype=np.float32)
        good_dq_bits = np.uint32(4 | 16777216)  # JUMP_DET | UNRELIABLE_SLOPE
        mask = (((np.asarray(roman['dq']) & ~good_dq_bits) != 0) |
                ~np.isfinite(roman['data']) |
                ~np.isfinite(error) | (error <= 0))
        if include_gain:
            poisson = np.asarray(roman['var_poisson'], dtype=np.float32)
            science = np.asarray(roman['data'])
            gain = np.zeros_like(error)
            valid_gain = np.isfinite(science) & (science > 0)
            valid_gain &= np.isfinite(poisson) & (poisson > 0)
            # SCI/VAR_POISSON estimates the optimal-fit effective gain, including
            # temporal weighting. At jump-affected pixels this can differ greatly
            # from uniform-ramp gain; do not reuse it for SCI_UNIF source noise.
            np.divide(science, poisson, out=gain, where=valid_gain)
            mask |= ~valid_gain | ~np.isfinite(gain) | (gain <= 0)
        variance = error.astype(float)**2
        if background_weights:
            import sep
            # Use the same mask and scalar SEP estimator as detection. Preserve SCI/ERR.
            background_image = np.array(roman['data'], dtype=np.float32, order='C')
            background_image[mask] = 0
            background = float(sep.Background(background_image, mask=mask).globalback)
            if 'var_rnoise' in roman:
                read_variance = np.array(roman['var_rnoise'], dtype=float)
                read_source = 'VAR_RNOISE'
            else:
                read_variance = np.zeros_like(variance)
                np.subtract(variance, poisson, out=read_variance, where=~mask)
                read_source = 'ERR^2-VAR_POISSON'
                if 'var_flat' in roman:
                    read_variance -= np.asarray(roman['var_flat'])
                    read_source = 'ERR^2-VAR_POISSON-VAR_FLAT'
            usable_read = ~mask & (np.asarray(roman['dq']) == 0)
            usable_read &= np.isfinite(read_variance) & (read_variance > 0)
            if not usable_read.any():
                raise ValueError('No usable unflagged pixels to estimate median read variance')
            median_read = float(np.median(read_variance[usable_read]))
            replace = ~mask & (read_variance < 0)
            # This repair is only for the legacy optimal-image weights. The
            # uniform model below discards these pixels from its calibration.
            read_variance[replace] = median_read
            mask |= ~np.isfinite(read_variance)
            variance = read_variance.copy()
            variance[~mask] += background / gain[~mask]
            image_header['SKYBKG'] = (background, 'Scalar SEP background in SCI units')
            image_header['WGTTYPE'] = ('BACKGROUND', 'IVAR excludes source Poisson noise')
            image_header['RNMETHOD'] = (read_source, 'Read variance estimate')
            image_header['RNVAR'] = (median_read, 'Median unflagged read variance')
            image_header['RNREPLAC'] = (int(replace.sum()), 'Negative read variances replaced')
            # Detection and Piff must subtract the same background used for these weights.
            for hdu in hdus:
                hdu.header['SKYBKG'] = image_header['SKYBKG']
            print(f'{input_path.name}: background={background:.7g}, '
                  f'median read variance={median_read:.7g}, '
                  f'replaced {replace.sum()} negative read variances')
        image_header['DQGOOD'] = (int(good_dq_bits), 'DQ bits allowed in MASK/IVAR')
        hdus.insert(4, fits.ImageHDU(data=mask.astype(np.int16),
                                     header=image_header.copy(), name='MASK'))
        ivar = np.zeros_like(error)
        np.divide(1.0, variance, out=ivar, where=~mask)
        hdus.insert(5, fits.ImageHDU(data=ivar, header=image_header.copy(), name='IVAR'))
        if include_gain:
            gain_header = image_header.copy()
            gain_header['GAINMETH'] = ('SCI/VAR_POISSON', 'Estimated effective Poisson gain')
            gain_ref = meta.get('ref_file', {}).get('gain')
            if gain_ref is not None:
                gain_header['GAINREF'] = (str(gain_ref), 'Detector gain reference')
            hdus.append(fits.ImageHDU(data=gain, header=gain_header, name='GAIN'))
        if 'dumo' in roman:
            uniform = np.asarray(roman['data'], dtype=np.float32) + np.asarray(
                roman['dumo'], dtype=np.float32)
            uniform_header = image_header.copy()
            uniform_header['RAMPFIT'] = ('UNIFORM', 'SCI_UNIF = original data + DUMO')
            uniform_header.add_history('ERR, IVAR, GAIN describe the original optimal ramp fit.')
            hdus.append(fits.ImageHDU(data=uniform, header=uniform_header, name='SCI_UNIF'))
            if uniform_weights:
                hdus.extend(uniform_hdus(roman, uniform_header, mask, mesh_size=mesh_size))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with fits.HDUList(hdus) as hdul:
            hdul.writeto(output_path, overwrite=overwrite, checksum=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inputs', type=Path, nargs='+', help='Calibrated *_cal.asdf files')
    parser.add_argument('--output-dir', type=Path,
                        help='Output directory (default: alongside each input ASDF)')
    parser.add_argument('--no-wcs', action='store_true', help='Export arrays without sky WCS')
    parser.add_argument('--gain-map', action='store_true', help='Export effective Poisson GAIN map')
    parser.add_argument('--background-weights', action='store_true',
                        help='Write read+background IVAR and effective GAIN (requires SEP)')
    parser.add_argument('--uniform', action='store_true',
                        help='Also build the uniform-ramp mask, IVAR and GAIN (requires SEP)')
    parser.add_argument('--mesh-size', type=int, default=128,
                        help='Uniform noise calibration tile size in pixels (default: 128)')
    parser.add_argument('--overwrite', action='store_true', help='Replace existing output files')
    args = parser.parse_args()
    for path in args.inputs:
        output_dir = path.parent if args.output_dir is None else args.output_dir
        output = convert(path, output_dir / path.with_suffix('.fits').name,
                         include_wcs=not args.no_wcs, overwrite=args.overwrite,
                         include_gain=args.gain_map, background_weights=args.background_weights,
                         uniform_weights=args.uniform, mesh_size=args.mesh_size)
        print(f'Wrote {output}')


if __name__ == '__main__':
    main()
