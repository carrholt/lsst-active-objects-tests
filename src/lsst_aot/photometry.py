"""
Photometry functions for measuring flux from model comet images.
"""
import numpy as np

from .models import make_nucleus_psf
from .utils import LSST_PIXEL_SCALE, rho_grid, seeing_to_moffat_alpha_pixels


def psf_photometry(image, center, alpha=seeing_to_moffat_alpha_pixels(0.75)):
    """
    Estimate flux by fitting a point source with the given Moffat PSF alpha
    (the same PSF as field stars) via a weighted least-squares fit.
    """
    template = make_nucleus_psf(image.shape, center, flux=1.0, alpha=alpha)
    return np.sum(image * template) / np.sum(template ** 2)


def psf_flux_fraction(image, center, total_flux, alpha=seeing_to_moffat_alpha_pixels(0.75)):
    """Fraction of total_flux recovered by PSF photometry."""
    return psf_photometry(image, center, alpha) / total_flux


def aperture_photometry(image, center, radius_arcsec=2.4, pixel_scale=LSST_PIXEL_SCALE):
    """Sum flux within a circular aperture of the given radius (arcsec)."""
    radius_px = radius_arcsec / pixel_scale
    rho = rho_grid(image.shape, center)
    return image[rho <= radius_px].sum()


def aperture_flux_fraction(image, center, total_flux, radius_arcsec=2.4, pixel_scale=LSST_PIXEL_SCALE):
    """Fraction of total_flux captured within the aperture."""
    return aperture_photometry(image, center, radius_arcsec, pixel_scale) / total_flux
