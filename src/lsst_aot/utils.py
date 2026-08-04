"""
Shared conversion helpers.
"""
import numpy as np

LSST_PIXEL_SCALE = 0.2  # arcsec / pixel


def fwhm_to_sigma(fwhm):
    """Convert a Gaussian FWHM to sigma (same units in and out)."""
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def seeing_to_sigma_pixels(fwhm_arcsec, pixel_scale=LSST_PIXEL_SCALE):
    """Convert seeing (FWHM, arcsec) to a Gaussian sigma in pixels."""
    return fwhm_to_sigma(fwhm_arcsec / pixel_scale)
