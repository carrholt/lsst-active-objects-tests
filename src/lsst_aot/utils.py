"""
Shared conversion helpers.
"""
import numpy as np

LSST_PIXEL_SCALE = 0.2  # arcsec / pixel


def rho_grid(shape, center):
    """Per-pixel distance from center."""
    y, x = np.indices(shape)
    return np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)


def fwhm_to_sigma(fwhm):
    """Convert a Gaussian FWHM to sigma (same units in and out)."""
    return fwhm / (2 * np.sqrt(2 * np.log(2)))


def seeing_to_sigma_pixels(fwhm_arcsec, pixel_scale=LSST_PIXEL_SCALE):
    """Convert seeing (FWHM, arcsec) to a Gaussian sigma in pixels."""
    return fwhm_to_sigma(fwhm_arcsec / pixel_scale)


def ab_mag_to_njy(mag):
    """Convert an AB apparent magnitude to flux density in nanojansky."""
    return 3631e9 * 10 ** (-mag / 2.5)


LSST_R_SKY_MAG = 21.2  # dark-sky surface brightness, r-band, mag/arcsec^2

# TODO: sky_mag above is treated as fixed, but real sky brightness fluctuates
# with airmass, lunar phase, twilight proximity, and passing clouds -- typically
# up to ~0.2-0.3 mag of variation. Could sample sky_mag from a distribution
# instead of a constant if we want to model that.


def sky_flux_per_pixel(sky_mag, pixel_scale=LSST_PIXEL_SCALE):
    """Convert sky surface brightness (mag/arcsec^2) to flux per pixel, in nJy."""
    return ab_mag_to_njy(sky_mag) * pixel_scale ** 2
