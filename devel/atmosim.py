import multiprocessing as mp
import numpy as np
from scipy.optimize import bisect
import galsim

def _vk_seeing(r0_500, wavelength, L0):
    # von Karman profile FWHM from Tokovinin fitting formula
    kolm_seeing = galsim.Kolmogorov(r0_500=r0_500, lam=wavelength).fwhm
    r0 = r0_500 * (wavelength/500)**1.2
    arg = 1. - 2.183*(r0/L0)**0.356
    factor = np.sqrt(arg) if arg > 0.0 else 0.0
    return kolm_seeing*factor


def _seeing_resid(r0_500, wavelength, L0, target_seeing):
    return _vk_seeing(r0_500, wavelength, L0) - target_seeing


def _r0_500(wavelength, L0, target_seeing):
    """Returns r0_500 to use to get target seeing."""
    r0_500_max = min(1.0, L0*(1./2.183)**(-0.356)*(wavelength/500.)**1.2)
    r0_500_min = 0.01
    return bisect(
        _seeing_resid,
        r0_500_min,
        r0_500_max,
        args=(wavelength, L0, target_seeing)
    )


def make_atmosphere(airmass,raw_seeing,wavelength,rng,L0_input=None,kcrit=0.2,screen_size=819.2,screen_scale=0.1,nproc=6,verbose=False):
    """
    Makes a Galsim Atmosphere class 
    ----------
    :param airmass: Value of airmass
    :param raw_seeing: Seeing at 500nm at zenith, with galsim.Angle units
    :param wavelength: Value in nm
    :param rng: Galsim random number generator
    :param L0_input: input L0 value in meters [default: None]
    
    Returns
    -------
    atm : galsim.PhaseScreenList
    target_FWHM : float
        Seeing at wavelength and airmass in arcseconds
    r0_500 : float
        Fried parameter at 500nm in meters
    L0 : float
        Outer scale in meters    
    """

    ud = galsim.UniformDeviate(rng)
    gd = galsim.GaussianDeviate(rng)

    target_FWHM = (
        raw_seeing/galsim.arcsec *
        airmass**0.6 *
        (wavelength/500.0)**(-0.3)
    )

    if verbose:
        print(f"raw seeing = {raw_seeing/galsim.arcsec}")
        print(f"airmass factor = {airmass**0.6}")
        print(f"wavelength factor = {(wavelength/500.0)**(-0.3)}")
        print(f"target FWHM = {target_FWHM}")

    # Use values measured from Ellerbroek 2008.
    altitudes = [0.0, 2.58, 5.16, 7.73, 12.89, 15.46]
    # Elevate the ground layer though.  Otherwise, PSFs come out too correlated
    # across the field of view.
    altitudes[0] = 0.2

    # Use weights from Ellerbroek too, but add some random perturbations.
    weights = [0.652, 0.172, 0.055, 0.025, 0.074, 0.022]
    weights = [np.abs(w*(1.0 + 0.1*gd())) for w in weights]
    weights = np.clip(weights, 0.01, 0.8)  # keep weights from straying too far.
    weights /= np.sum(weights)  # renormalize
    print("Weights = ",weights)

    # Draw outer scale from truncated log normal
    if L0_input == None:
        L0 = 0
        while L0 < 10.0 or L0 > 100:
            L0 = np.exp(gd() * 0.6 + np.log(25.0))
    else:
        L0 = L0_input
        
    # Given the desired targetFWHM and randomly selected L0, determine
    # appropriate r0_500
    r0_500 = _r0_500(wavelength, L0, target_FWHM)

    # Broadcast common outer scale across all layers
    L0 = [L0]*6

    # Uniformly draw layer speeds between 0 and max_speed.
    max_speed = 20.0
    speeds = rng.np.uniform(0, max_speed, 6)
    # Isotropically draw directions.
    directions = [rng.np.uniform(0, 360)*galsim.degrees for _ in range(6)]
    print("speeds, directions = ",speeds,directions)

    atm_kwargs = dict(
        r0_500=r0_500,
        L0=L0,
        speed=speeds,
        direction=directions,
        altitude=altitudes,
        r0_weights=weights,
        rng=rng,
        screen_size=screen_size,
        screen_scale=screen_scale
    )

    ctx = mp.get_context('fork')
    atm = galsim.Atmosphere(mp_context=ctx, **atm_kwargs)

    r0_500_new = atm.r0_500_effective
    r0 = r0_500_new * (wavelength/500.0)**(6./5)
    kmax = kcrit/r0

    with ctx.Pool(
        nproc,
        initializer=galsim.phase_screens.initWorker,
        initargs=galsim.phase_screens.initWorkerArgs()
    ) as pool:
        atm.instantiate(pool=pool, kmax=kmax, check='phot')

    return atm, target_FWHM, r0_500, L0
