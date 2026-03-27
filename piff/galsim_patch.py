import numpy as np
import galsim
from astropy import units
from galsim import utilities
from galsim.table import LookupTable, _LookupTable
from galsim.sed import SED
from galsim.errors import GalSimError

# Temporary compatibility patch for GalSim < 2.9.
# Remove this module once Piff can depend on GalSim 2.9+, which includes thin(bandpass=...).

def _piff_sed_thin(self, rel_err=1.e-4, trim_zeros=True, preserve_range=True, fast_search=True,
                   bandpass=None):  # pragma: no cover
    def _bandpass_native_waves(waves):
        # The thinning algorithm runs on the SED's native tabulation grid, but when a
        # bandpass target is provided we need to evaluate the throughput at the same physical
        # observed wavelengths. This helper converts SED-native wavelengths into the native
        # wavelength units used by bandpass._tp.
        if self.wave_factor:
            observed_nm = waves / self.wave_factor
        else:
            observed_nm = (waves * self.wave_type).to(units.nm, units.spectral()).value
        observed_nm *= (1.0 + self.redshift)
        if bandpass.wave_factor:
            return observed_nm * bandpass.wave_factor
        else:
            return (observed_nm * units.nm).to(bandpass.wave_type, units.spectral()).value

    if bandpass is not None:
        if self.blue_limit > bandpass.red_limit or self.red_limit < bandpass.blue_limit:
            raise GalSimError("Bandpass does not overlap the SED wavelength range.")
        wave_list, _, _ = utilities.combine_wave_list(self, bandpass)
        if preserve_range and not trim_zeros:
            # If we want to preserve the range, add back the limits to each end.
            front = [self.blue_limit] if self.blue_limit < wave_list[0] else []
            back = [self.red_limit] if self.red_limit > wave_list[-1] else []
            if front or back:
                wave_list = np.concatenate((front, wave_list, back))
    else:
        wave_list = self.wave_list

    if len(wave_list) > 0:
        rest_wave_native = self._get_rest_native_waves(wave_list)
        spec_native = self._spec(rest_wave_native)

        if bandpass is not None:
            # Identify the overlapping region in the SED's native wavelength units to
            # determine which portion of the wave_list is within the bandpass limits.
            band_native_limits = self._get_rest_native_waves(
                np.array([bandpass.blue_limit, bandpass.red_limit])
            )
            native_blue_limit = np.min(band_native_limits)
            native_red_limit = np.max(band_native_limits)
            in_band = np.logical_and(
                rest_wave_native >= native_blue_limit,
                rest_wave_native <= native_red_limit,
            )

            # Compute the product SED * bandpass, since that is the quantity whose integral we
            # want to preserve for this observation.
            tp_native = np.zeros_like(rest_wave_native, dtype=float)
            bp_wave_native = _bandpass_native_waves(rest_wave_native[in_band])
            tp_native[in_band] = bandpass._tp(bp_wave_native) / bandpass.wave_factor
            spec_native *= tp_native

        # Note that this is thinning in native units, not nm and photons/nm.
        interpolant = (self.interpolant if not isinstance(self._spec, LookupTable)
                       else self._spec.interpolant)
        newx, newf = utilities.thin_tabulated_values(
                rest_wave_native, spec_native, rel_err=rel_err,
                trim_zeros=trim_zeros, preserve_range=preserve_range,
                fast_search=fast_search, interpolant=interpolant)

        if bandpass is not None:
            # Convert the thinned product back into an SED by dividing out the bandpass
            # wherever the throughput is non-zero.
            in_band = np.logical_and(newx >= native_blue_limit, newx <= native_red_limit)
            tp_native = np.zeros_like(newx, dtype=float)
            bp_wave_native = _bandpass_native_waves(newx[in_band])
            tp_native[in_band] = bandpass._tp(bp_wave_native) / bandpass.wave_factor
            nz = tp_native != 0.  # Don't divide by 0.
            assert np.all(newf[~nz] == 0.)
            newf[nz] /= tp_native[nz]

        newspec = _LookupTable(newx, newf, interpolant=interpolant)
        return SED(newspec, self.wave_type, self.flux_type, redshift=self.redshift,
                   fast=self.fast)
    else:
        return self


galsim.SED.thin = _piff_sed_thin
