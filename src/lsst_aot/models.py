"""
Model comet generation: PSF nucleus + coma image models.
"""
import numpy as np
from astropy.io import fits

from .utils import LSST_R_SKY_MAG, ab_mag_to_njy, rho_grid, seeing_to_sigma_pixels, sky_flux_per_pixel


def make_nucleus_psf(shape, center, flux, sigma):
    """Generate a Gaussian PSF representing the bare nucleus, normalized to `flux`."""
    rho = rho_grid(shape, center)
    psf = np.exp(-rho ** 2 / (2 * sigma ** 2))
    return flux * psf / psf.sum()


def make_coma_profile(shape, center, flux, coma_type="symmetric", rho_min=1.0, **kwargs):
    """
    Generate a coma surface-brightness profile.

    coma_type options:
        "symmetric"   - classic 1/rho falloff
        "asymmetric"  - tail-dominated / sunward-antisunward asymmetry
        "outburst"    - localized flux excess (e.g. recent fragmentation event)
    """
    if coma_type != "symmetric":
        raise NotImplementedError(f"coma_type={coma_type!r} not implemented yet")

    rho = rho_grid(shape, center)
    rho = np.clip(rho, rho_min, None)  # avoid 1/rho singularity at the nucleus
    coma = 1.0 / rho
    return flux * coma / coma.sum()


def make_model_comet(
    shape=(101, 101),
    mag=20.0,
    nucleus_fraction=0.05,
    sigma=seeing_to_sigma_pixels(0.75),
    coma_type="symmetric",
    sky_mag=LSST_R_SKY_MAG,
    seed=None,
    **coma_kwargs,
):
    """
    Build a full model comet image: nucleus PSF + coma, optional sky noise.

    mag is the apparent AB magnitude of the comet, converted to a total
    flux in nJy. nucleus_fraction sets what fraction of that flux goes
    into the unresolved nucleus PSF; the rest goes into the coma.

    sky_mag is the sky surface brightness (mag/arcsec^2, AB) used to set
    the per-pixel background noise; pass None to skip noise entirely.

    seed controls the sky noise draw, for reproducible trials.

    Returns
    -------
    numpy.ndarray
    """
    total_flux = ab_mag_to_njy(mag)
    nucleus_flux = nucleus_fraction * total_flux
    coma_flux = total_flux - nucleus_flux

    center = (shape[0] // 2, shape[1] // 2)
    nucleus = make_nucleus_psf(shape, center, nucleus_flux, sigma)
    coma = make_coma_profile(shape, center, coma_flux, coma_type=coma_type, **coma_kwargs)
    image = nucleus + coma

    if sky_mag is not None:
        noise_sigma = np.sqrt(sky_flux_per_pixel(sky_mag))
        rng = np.random.default_rng(seed)
        image = image + rng.normal(0, noise_sigma, shape)

    return image


def save_model_comet(image, filepath, header_extra=None):
    """Save a model comet image as FITS with metadata about generation params."""
    hdu = fits.PrimaryHDU(image)
    if header_extra:
        for k, v in header_extra.items():
            hdu.header[k] = v
    hdu.writeto(filepath, overwrite=True)
