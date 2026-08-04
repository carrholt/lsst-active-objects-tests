"""
Model comet generation: PSF nucleus + coma image models.
"""
import numpy as np
from astropy.io import fits

from .utils import seeing_to_sigma_pixels


def _rho_grid(shape, center):
    """Per-pixel distance from center."""
    y, x = np.indices(shape)
    return np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)


def make_nucleus_psf(shape, center, flux, sigma):
    """Generate a Gaussian PSF representing the bare nucleus, normalized to `flux`."""
    rho = _rho_grid(shape, center)
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

    rho = _rho_grid(shape, center)
    rho = np.clip(rho, rho_min, None)  # avoid 1/rho singularity at the nucleus
    coma = 1.0 / rho
    return flux * coma / coma.sum()


def make_model_comet(
    shape=(101, 101),
    nucleus_flux=1000.0,
    coma_flux=500.0,
    sigma=seeing_to_sigma_pixels(0.75),
    coma_type="symmetric",
    noise_level=None,
    **coma_kwargs,
):
    """
    Build a full model comet image: nucleus PSF + coma, optional noise.

    Returns
    -------
    numpy.ndarray
    """
    center = (shape[0] // 2, shape[1] // 2)
    nucleus = make_nucleus_psf(shape, center, nucleus_flux, sigma)
    coma = make_coma_profile(shape, center, coma_flux, coma_type=coma_type, **coma_kwargs)
    image = nucleus + coma

    if noise_level is not None:
        image += np.random.normal(0, noise_level, shape)

    return image


def save_model_comet(image, filepath, header_extra=None):
    """Save a model comet image as FITS with metadata about generation params."""
    hdu = fits.PrimaryHDU(image)
    if header_extra:
        for k, v in header_extra.items():
            hdu.header[k] = v
    hdu.writeto(filepath, overwrite=True)
