import pytest

from lsst_aot.utils import (
    ab_mag_to_njy,
    fwhm_to_sigma,
    seeing_to_sigma_pixels,
    sky_flux_per_pixel,
)


def test_fwhm_to_sigma():
    # FWHM = 2*sqrt(2*ln2) * sigma, so this FWHM should map back to sigma=1
    assert fwhm_to_sigma(2.3548200450309493) == pytest.approx(1.0)


def test_seeing_to_sigma_pixels():
    # 0.75" seeing at 0.2"/pixel -> FWHM = 3.75 px -> sigma ~ 1.59 px
    assert seeing_to_sigma_pixels(0.75, pixel_scale=0.2) == pytest.approx(1.5924783755400358)


def test_ab_mag_to_njy_zeropoint():
    # AB mag 0 is defined as 3631 Jy = 3631e9 nJy
    assert ab_mag_to_njy(0.0) == pytest.approx(3631e9)


def test_ab_mag_to_njy():
    assert ab_mag_to_njy(20.0) == pytest.approx(36310.0)


def test_sky_flux_per_pixel():
    assert sky_flux_per_pixel(21.2, pixel_scale=0.2) == pytest.approx(480.93484084131495)
