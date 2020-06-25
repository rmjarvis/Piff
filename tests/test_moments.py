import numpy as np
import piff
import galsim
import copy
from piff.star import Star, StarFit, StarData
from piff.util import calculate_moments

from piff_test_helper import timer
from piff.util import hsm




def adrawProfile(star, prof, params, use_fit=True, copy_image=True):
    """Generate PSF image for a given star and profile

    :param star:        Star instance holding information needed for
                        interpolation as well as an image/WCS into which
                        PSF will be rendered.
    :param profile:     A galsim profile
    :param params:      Params associated with profile to put in the star.
    :param use_fit:     Bool [default: True] shift the profile by a star's
                        fitted center and multiply by its fitted flux

    :returns:   Star instance with its image filled with rendered PSF
    """
    # use flux and center properties
    if use_fit:
        prof = prof.shift(star.fit.center) * star.fit.flux
    image, weight, image_pos = star.data.getImage()
    if copy_image:
        image_model = image.copy()
    else:
        image_model = image
    prof.drawImage(image_model, method='auto', center=star.image_pos)
    properties = star.data.properties.copy()
    for key in ['x', 'y', 'u', 'v']:
        # Get rid of keys that constructor doesn't want to see:
        properties.pop(key, None)
    data = StarData(image=image_model,
                        image_pos=star.data.image_pos,
                        weight=star.data.weight,
                        pointing=star.data.pointing,
                        field_pos=star.data.field_pos,
                        orig_weight=star.data.orig_weight,
                        properties=properties)
    fit = StarFit(params,
                      flux=star.fit.flux,
                      center=star.fit.center)
    return Star(data, fit)


def makeStarsMoffat(nstar=100, beta=5., forcefail=False, test_return_error=False):

    # Moffat
    psf = piff.Moffat(beta)
    rng = galsim.BaseDeviate(12345)

    # make some stars w/o shot noise
    noiseless_stars = []
    noisy_stars = []
    all_star_moments = []
    pixel_sum = 1.e6
    foreground = 200.0

    # all stars in the same CCD
    ccdnum = 10  # also just put them in the same sensor
    decaminfo = piff.des.DECamInfo()
    wcs = decaminfo.get_nominal_wcs(ccdnum)

    # set the pixel index randomly x,y
    x = np.random.uniform(0,2048,nstar)
    y = np.random.uniform(0,4096,nstar)

    # pick random optics size
    optics_size = np.linspace(0.7,1.2,nstar)
    g1_lo = -0.1
    g1_hi = 0.1
    optics_g1 = np.linspace(g1_lo,g1_hi,nstar)
    optics_g2 = np.linspace(g1_lo,g1_hi,nstar)

    fit_params = np.array([1.0,0.0,0.0])
    idx_optics_size = 0
    idx_optics_g1 = 1
    idx_optics_g2 = 2

    vars = ['optics_size', 'optics_g1', 'optics_g2']

    moment_str = ['M00','M10','M01','M11','M20','M02',
                  'M21', 'M12', 'M30', 'M03',
                  'M31','M13','M40','M04',
                  'M22dup', 'M22n','M33n','M44n',
                  'varM00','varM10','varM01','varM11','varM20','varM02',
                  'varM21', 'varM12', 'varM30', 'varM03',
                  'varM31','varM13','varM40','varM04',
                  'varM22dup','varM22n','varM33n','varM44n']

    # build names of columns
    moments_names = [s + "_nonoise" for s in moment_str]
    moments_noise_names = [s + "_noise" for s in moment_str]
    moffat_names = moments_names + moments_noise_names + vars

    for i in range(nstar):

        fit_params[idx_optics_size] = optics_size[i]
        fit_params[idx_optics_g1] = optics_g1[i]
        fit_params[idx_optics_g2] = optics_g2[i]

        # make the star, its an empty vessel, with just position and wcs
        properties_in = {'chipnum': ccdnum}
        blank_star = piff.Star.makeTarget(x=x[i], y=y[i], wcs=wcs, stamp_size=19, properties=properties_in)

        prof = psf.getProfile(fit_params)
        noiseless_star = adrawProfile(blank_star, prof, fit_params)

        noiseless_stars.append(noiseless_star)

        # scale the image's pixel_sum, work with a copy
        im = copy.deepcopy(noiseless_star.image) * pixel_sum

        # Generate a Poisson noise model, with some foreground (assumes that this foreground was already subtracted)
        poisson_noise = galsim.PoissonNoise(rng,sky_level=foreground)
        im.addNoise(poisson_noise)  # adds in place

        # get new weight in photo-electrons (not an array)
        inverse_weight = im + foreground
        weight = 1.0/inverse_weight

        # make new noisy star by resetting data in the noiseless star
        noisy_star = piff.Star(noiseless_star.data.setData(im.array.flatten(),True), noiseless_star.fit)
        noisy_star.data.weight = weight   # overwrite weight inside star

        if forcefail:
            noisy_star.data.weight *= 0.

        noisy_stars.append(noisy_star)

        # moments
        moments = calculate_moments(star=noiseless_stars[i],errors=True, third_order=True, fourth_order=True, radial=True)
        moments_noise = calculate_moments(star=noisy_stars[i],errors=True, third_order=True, fourth_order=True, radial=True)

        if test_return_error:
            moments_check = calculate_moments(star=noiseless_stars[i],errors=False, third_order=True, fourth_order=True, radial=True)
            nval = len(moments_check)
            np.testing.assert_equal(np.array(moments)[0:nval], np.array(moments_check))


        all_moments =  moments + moments_noise + tuple(fit_params)
        all_star_moments.append(all_moments)

    # Work it, put it down, flip it and reverse it.
    full_array = np.vstack(all_star_moments)
    df = {}
    for key, val in zip(moffat_names, full_array.T):
        df[key] = np.array(val)

    return df


def makepulldist(dft, beta, vname):

    name_noise = "%s_noise" % (vname)
    name_nonoise = "%s_nonoise" % (vname)
    name_sigma = "var%s_noise" % (vname)

    try:
        diff = dft[name_noise] - dft[name_nonoise]
    except KeyError:
        print (name_noise, name_nonoise, dft.keys())
    pull = diff/np.sqrt(dft[name_sigma])

    return pull



@timer
def test_moments_return():

    np.random.seed(12345)
    rng = galsim.BaseDeviate(12345)
    dft = makeStarsMoffat(nstar=1,beta=5.,test_return_error=True)



@timer
def test_moments_fail():

    np.random.seed(12345)
    rng = galsim.BaseDeviate(12345)
    try:
        dft = makeStarsMoffat(nstar=1,beta=5.,forcefail=True)
        assert False
    except galsim.errors.GalSimHSMError:
        pass


@timer
def test_moments(dftlist=None):

    # always run this
    np.random.seed(12345)
    rng = galsim.BaseDeviate(12345)

    betalist = [1.5, 2.5, 5.]
    keylist = ["1p5", "2p5", "5"]
    if dftlist is None:
        dftlist = [makeStarsMoffat(nstar=1000,beta=betaval) for betaval in betalist]
    momentlist = ['M10','M01','M11','M20','M02','M21','M12','M30','M03','M31','M13','M40','M04','M22n','M33n','M44n']

    rmsval_dict = dict(M10=[1.024323, 0.996639,0.986769],
                       M01=[0.981495, 0.997452, 0.962892],
                       M11=[1.329671, 1.045360, 0.950834],
                       M20=[1.099344, 1.029830, 0.931873],
                       M02=[1.112284, 1.023089, 0.979789],
                       M21=[0.930467, 0.985090, 0.973814],
                       M12=[0.927560, 0.999851, 1.044756],
                       M30=[0.994757, 0.997164, 0.967364],
                       M03=[0.941321, 1.015403, 1.003081],
                       M31=[1.257617, 1.082664, 0.923452],
                       M13=[1.287733, 1.088511, 0.995732],
                       M40=[1.199421, 1.136400, 1.049415],
                       M04=[1.250599, 1.169380, 1.106795],
                       M22n=[0.879955, 0.985517, 1.017496],
                       M33n=[0.835669, 0.999379, 1.065365],
                       M44n=[0.809727, 1.021675, 1.119339])

    odict = {}
    for i, (dft, beta, key) in enumerate(zip(dftlist, betalist, keylist)):
        stacked = np.vstack([ makepulldist(dft, beta, amoment) for amoment in momentlist])
        testvals = np.array([ rmsval_dict[amoment][i] for amoment in momentlist])
        mean_pull = stacked.mean(1)
        rms_pull = stacked.std(1)
        failmask = np.fabs(rms_pull-testvals) > 0.1
        np.testing.assert_allclose(mean_pull, 0., atol=0.1)
        np.testing.assert_allclose(rms_pull, testvals, rtol=0.2)
