"""
Single-trial experiment: build a model comet, measure it with both
photometry methods, and package the results as one result row.
"""
import numpy as np

from .detection import compute_bbox_size
from .models import make_model_comet
from .photometry import (
    aperture_flux_fraction,
    aperture_photometry,
    psf_flux_fraction,
    psf_photometry,
)
from .utils import LSST_R_SKY_MAG, ab_mag_to_njy, njy_to_ab_mag, seeing_to_sigma_pixels, sky_flux_per_pixel

BASE_FIELDNAMES = [
    "mag",
    "nucleus_fraction",
    "shape_y",
    "shape_x",
    "sigma_px",
    "sky_mag",
    "total_flux_njy",
    "total_mag",
    "psf_flux_njy",
    "psf_mag",
    "psf_fraction",
    "bboxSize",
    "seed",
]

DILATION_FACTOR = 2.4  # dilation radius = DILATION_FACTOR * psf sigma


def _radius_tag(radius_arcsec):
    """Turn an arcsec radius into a column-name-safe tag, e.g. 2.4 -> '2p4'."""
    return f"{radius_arcsec:g}".replace(".", "p")


def aperture_fieldnames(aperture_radii_arcsec):
    """Column names for the per-radius aperture measurements."""
    names = []
    for radius_arcsec in aperture_radii_arcsec:
        tag = _radius_tag(radius_arcsec)
        names += [f"aperture_r{tag}_flux_njy", f"aperture_r{tag}_mag", f"aperture_r{tag}_fraction"]
    return names


def fieldnames_for(aperture_radii_arcsec):
    """Full CSV column list for a given set of aperture radii."""
    return BASE_FIELDNAMES + aperture_fieldnames(aperture_radii_arcsec)


def run_trial(
    mag,
    nucleus_fraction=0.05,
    aperture_radii_arcsec=(2.4,),
    shape=(101, 101),
    sigma=seeing_to_sigma_pixels(0.75),
    sky_mag=LSST_R_SKY_MAG,
    seed=None,
):
    """
    Build one model comet, measure it with PSF photometry and aperture
    photometry at every radius in aperture_radii_arcsec, and return one
    result row dict.
    """
    image = make_model_comet(
        shape=shape,
        mag=mag,
        nucleus_fraction=nucleus_fraction,
        sigma=sigma,
        sky_mag=sky_mag,
        seed=seed,
    )
    center = (image.shape[0] // 2, image.shape[1] // 2)
    total_flux = ab_mag_to_njy(mag)

    psf_flux = psf_photometry(image, center)

    noise_sigma = np.sqrt(sky_flux_per_pixel(sky_mag))
    dilation_radius = DILATION_FACTOR * sigma
    bbox_size = compute_bbox_size(image, noise_sigma, dilation_radius)

    row = {
        "mag": mag,
        "nucleus_fraction": nucleus_fraction,
        "shape_y": shape[0],
        "shape_x": shape[1],
        "sigma_px": sigma,
        "sky_mag": sky_mag,
        "total_flux_njy": total_flux,
        "total_mag": mag,
        "psf_flux_njy": psf_flux,
        "psf_mag": njy_to_ab_mag(psf_flux),
        "psf_fraction": psf_flux_fraction(image, center, total_flux),
        "bboxSize": bbox_size,
        "seed": seed,
    }

    for radius_arcsec in aperture_radii_arcsec:
        tag = _radius_tag(radius_arcsec)
        aperture_flux = aperture_photometry(image, center, radius_arcsec=radius_arcsec)
        row[f"aperture_r{tag}_flux_njy"] = aperture_flux
        row[f"aperture_r{tag}_mag"] = njy_to_ab_mag(aperture_flux)
        row[f"aperture_r{tag}_fraction"] = aperture_flux_fraction(
            image, center, total_flux, radius_arcsec=radius_arcsec
        )

    return row
