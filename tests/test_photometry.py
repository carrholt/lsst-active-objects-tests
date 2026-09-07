import numpy as np
import pytest

from lsst_aot.models import make_model_comet, make_nucleus_psf
from lsst_aot.photometry import (
    aperture_flux_fraction,
    aperture_photometry,
    psf_flux_fraction,
    psf_photometry,
)
from lsst_aot.utils import ab_mag_to_njy, seeing_to_moffat_alpha_pixels


def test_psf_photometry_recovers_point_source_flux():
    shape = (51, 51)
    center = (25, 25)
    alpha = seeing_to_moffat_alpha_pixels(0.75)
    image = make_nucleus_psf(shape, center, flux=500.0, alpha=alpha)
    assert psf_photometry(image, center, alpha=alpha) == pytest.approx(500.0)


def test_psf_photometry_underestimates_extended_source():
    image = make_model_comet(mag=20.0, sky_mag=None)
    total_flux = ab_mag_to_njy(20.0)
    flux = psf_photometry(image, (50, 50))
    assert 0 < flux < total_flux


def test_psf_flux_fraction_matches_ratio():
    image = make_model_comet(mag=20.0, sky_mag=None)
    center = (50, 50)
    total_flux = ab_mag_to_njy(20.0)
    fraction = psf_flux_fraction(image, center, total_flux)
    expected = psf_photometry(image, center) / total_flux
    assert fraction == pytest.approx(expected)


def test_psf_flux_fraction_between_zero_and_one():
    image = make_model_comet(mag=20.0, sky_mag=None)
    total_flux = ab_mag_to_njy(20.0)
    fraction = psf_flux_fraction(image, (50, 50), total_flux)
    assert 0 < fraction < 1


def test_aperture_photometry_counts_pixels_in_radius():
    shape = (11, 11)
    center = (5, 5)
    image = np.ones(shape)
    radius = 2.0

    expected = sum(
        1
        for dy in range(-5, 6)
        for dx in range(-5, 6)
        if np.sqrt(dx ** 2 + dy ** 2) <= radius
    )
    result = aperture_photometry(image, center, radius_arcsec=radius, pixel_scale=1.0)
    assert result == pytest.approx(expected)


def test_aperture_photometry_increases_with_radius():
    image = make_model_comet(mag=20.0, sky_mag=None)
    center = (50, 50)
    small = aperture_photometry(image, center, radius_arcsec=1.0)
    large = aperture_photometry(image, center, radius_arcsec=5.0)
    assert large > small


def test_aperture_photometry_bounded_by_total_flux():
    image = make_model_comet(mag=20.0, sky_mag=None)
    total_flux = ab_mag_to_njy(20.0)
    flux = aperture_photometry(image, (50, 50), radius_arcsec=2.4)
    assert 0 < flux < total_flux


def test_aperture_flux_fraction_matches_ratio():
    image = make_model_comet(mag=20.0, sky_mag=None)
    center = (50, 50)
    total_flux = ab_mag_to_njy(20.0)
    fraction = aperture_flux_fraction(image, center, total_flux, radius_arcsec=2.4)
    expected = aperture_photometry(image, center, radius_arcsec=2.4) / total_flux
    assert fraction == pytest.approx(expected)


def test_aperture_flux_fraction_between_zero_and_one():
    image = make_model_comet(mag=20.0, sky_mag=None)
    total_flux = ab_mag_to_njy(20.0)
    fraction = aperture_flux_fraction(image, (50, 50), total_flux, radius_arcsec=2.4)
    assert 0 < fraction < 1
