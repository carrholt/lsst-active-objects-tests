"""
Model comet generation: PSF nucleus + coma image models.
"""
import numpy as np
from astropy.io import fits


def make_nucleus_psf(shape, center, flux, fwhm):
    """Generate a Gaussian (or Moffat) PSF representing the bare nucleus."""
    # TODO: implement PSF profile
    pass


def make_coma_profile(shape, center, flux, coma_type="symmetric", **kwargs):
    """
    Generate a coma surface-brightness profile.

    coma_type options:
        "symmetric"   - classic 1/rho falloff
        "asymmetric"  - tail-dominated / sunward-antisunward asymmetry
        "outburst"    - localized flux excess (e.g. recent fragmentation event)
    """
    # TODO: implement per coma_type
    pass


def make_model_comet(
    shape=(101, 101),
    nucleus_flux=1000.0,
    coma_flux=500.0,
    fwhm=3.0,
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
    nucleus = make_nucleus_psf(shape, center, nucleus_flux, fwhm)
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