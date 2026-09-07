"""
Shared conversion helpers.
"""
import numpy as np

LSST_PIXEL_SCALE = 0.2  # arcsec / pixel


def rho_grid(shape, center):
    """Per-pixel distance from center."""
    y, x = np.indices(shape)
    return np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)


# Moffat beta, fit to the real PSFIMAGE stamps in data/example_alerts/
# (see notebooks/real_comet_alert_investigation.ipynb) -- real LSST PSFs
# have heavier wings than a Gaussian, so this is the shape used for all
# fake PSFs (nucleus injection and psfFlux measurement template alike).
MOFFAT_BETA = 3.3


def fwhm_to_moffat_alpha(fwhm, beta=MOFFAT_BETA):
    """Convert a Moffat FWHM to alpha (same units in and out)."""
    return fwhm / (2 * np.sqrt(2 ** (1 / beta) - 1))


def seeing_to_moffat_alpha_pixels(fwhm_arcsec, beta=MOFFAT_BETA, pixel_scale=LSST_PIXEL_SCALE):
    """Convert seeing (FWHM, arcsec) to a Moffat alpha in pixels."""
    return fwhm_to_moffat_alpha(fwhm_arcsec / pixel_scale, beta)


def ab_mag_to_njy(mag):
    """Convert an AB apparent magnitude to flux density in nanojansky."""
    return 3631e9 * 10 ** (-mag / 2.5)


def njy_to_ab_mag(flux_njy):
    """Convert a flux density in nanojansky to an AB apparent magnitude."""
    return -2.5 * np.log10(flux_njy / 3631e9)


LSST_R_SKY_MAG = 21.2  # dark-sky surface brightness, r-band, mag/arcsec^2

# TODO: sky_mag above is treated as fixed, but real sky brightness fluctuates
# with airmass, lunar phase, twilight proximity, and passing clouds -- typically
# up to ~0.2-0.3 mag of variation. Could sample sky_mag from a distribution
# instead of a constant if we want to model that.


def sky_flux_per_pixel(sky_mag, pixel_scale=LSST_PIXEL_SCALE):
    """Convert sky surface brightness (mag/arcsec^2) to flux per pixel, in nJy."""
    return ab_mag_to_njy(sky_mag) * pixel_scale ** 2
