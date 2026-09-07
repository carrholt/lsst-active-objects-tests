import pytest

from lsst_aot.utils import (
    ab_mag_to_njy,
    fwhm_to_moffat_alpha,
    seeing_to_moffat_alpha_pixels,
    sky_flux_per_pixel,
)


def test_fwhm_to_moffat_alpha():
    # FWHM = 2*alpha*sqrt(2**(1/beta) - 1), so this FWHM should map back to alpha=1 at beta=3.3
    assert fwhm_to_moffat_alpha(0.9669189902666457, beta=3.3) == pytest.approx(1.0)


def test_seeing_to_moffat_alpha_pixels():
    # 0.75" seeing at 0.2"/pixel -> FWHM = 3.75 px -> alpha ~ 3.878 px at beta=3.3
    assert seeing_to_moffat_alpha_pixels(0.75, pixel_scale=0.2) == pytest.approx(3.8782980143619565)


def test_ab_mag_to_njy_zeropoint():
    # AB mag 0 is defined as 3631 Jy = 3631e9 nJy
    assert ab_mag_to_njy(0.0) == pytest.approx(3631e9)


def test_ab_mag_to_njy():
    assert ab_mag_to_njy(20.0) == pytest.approx(36310.0)


def test_sky_flux_per_pixel():
    assert sky_flux_per_pixel(21.2, pixel_scale=0.2) == pytest.approx(480.93484084131495)
