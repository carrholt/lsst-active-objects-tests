"""
Single-trial experiment: build a model comet, measure it with both
photometry methods, and package the results as one result row.
"""
from .models import make_model_comet
from .photometry import (
    aperture_flux_fraction,
    aperture_photometry,
    psf_flux_fraction,
    psf_photometry,
)
from .utils import ab_mag_to_njy, njy_to_ab_mag

FIELDNAMES = [
    "mag",
    "nucleus_fraction",
    "total_flux_njy",
    "total_mag",
    "psf_flux_njy",
    "psf_mag",
    "psf_fraction",
    "aperture_radius_arcsec",
    "aperture_flux_njy",
    "aperture_mag",
    "aperture_fraction",
    "seed",
]


def run_trial(mag, nucleus_fraction=0.05, aperture_radius_arcsec=2.4, seed=None):
    """Build one model comet and measure it; returns a result row dict."""
    image = make_model_comet(mag=mag, nucleus_fraction=nucleus_fraction, seed=seed)
    center = (image.shape[0] // 2, image.shape[1] // 2)
    total_flux = ab_mag_to_njy(mag)

    psf_flux = psf_photometry(image, center)
    aperture_flux = aperture_photometry(image, center, radius_arcsec=aperture_radius_arcsec)

    return {
        "mag": mag,
        "nucleus_fraction": nucleus_fraction,
        "total_flux_njy": total_flux,
        "total_mag": mag,
        "psf_flux_njy": psf_flux,
        "psf_mag": njy_to_ab_mag(psf_flux),
        "psf_fraction": psf_flux_fraction(image, center, total_flux),
        "aperture_radius_arcsec": aperture_radius_arcsec,
        "aperture_flux_njy": aperture_flux,
        "aperture_mag": njy_to_ab_mag(aperture_flux),
        "aperture_fraction": aperture_flux_fraction(
            image, center, total_flux, radius_arcsec=aperture_radius_arcsec
        ),
        "seed": seed,
    }
