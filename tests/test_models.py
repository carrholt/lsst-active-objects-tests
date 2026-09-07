import numpy as np
import pytest
from astropy.io import fits

from lsst_aot.models import (
    make_coma_profile,
    make_model_comet,
    make_nucleus_psf,
    save_model_comet,
)
from lsst_aot.utils import ab_mag_to_njy


def test_make_nucleus_psf_conserves_flux():
    psf = make_nucleus_psf((51, 51), (25, 25), flux=100.0, alpha=2.0)
    assert psf.sum() == pytest.approx(100.0)


def test_make_nucleus_psf_peaks_at_center():
    psf = make_nucleus_psf((51, 51), (25, 25), flux=100.0, alpha=2.0)
    assert np.unravel_index(np.argmax(psf), psf.shape) == (25, 25)
    assert np.all(psf >= 0)


def test_make_coma_profile_conserves_flux():
    coma = make_coma_profile((51, 51), (25, 25), flux=200.0)
    assert coma.sum() == pytest.approx(200.0)


def test_make_coma_profile_falls_off_with_radius():
    coma = make_coma_profile((51, 51), (25, 25), flux=200.0)
    assert coma[25, 25] > coma[25, 30] > coma[25, 40]


def test_make_coma_profile_unknown_type_raises():
    with pytest.raises(NotImplementedError):
        make_coma_profile((51, 51), (25, 25), flux=200.0, coma_type="asymmetric")


def test_make_model_comet_shape():
    image = make_model_comet(shape=(41, 41))
    assert image.shape == (41, 41)


def test_make_model_comet_conserves_flux_without_noise():
    image = make_model_comet(mag=20.0, sky_mag=None)
    assert image.sum() == pytest.approx(ab_mag_to_njy(20.0))


def test_make_model_comet_nucleus_fraction_changes_peak():
    low_nucleus = make_model_comet(mag=20.0, nucleus_fraction=0.01, sky_mag=None)
    high_nucleus = make_model_comet(mag=20.0, nucleus_fraction=0.5, sky_mag=None)
    center = (50, 50)
    assert high_nucleus[center] > low_nucleus[center]


def test_make_model_comet_no_noise_is_deterministic():
    image1 = make_model_comet(sky_mag=None)
    image2 = make_model_comet(sky_mag=None)
    np.testing.assert_array_equal(image1, image2)


def test_make_model_comet_sky_noise_adds_variation():
    noiseless = make_model_comet(sky_mag=None)
    noisy = make_model_comet(sky_mag=21.2)
    assert not np.allclose(noiseless, noisy)


def test_save_model_comet(tmp_path):
    image = make_model_comet(shape=(21, 21), sky_mag=None)
    filepath = tmp_path / "comet.fits"
    save_model_comet(image, filepath, header_extra={"MAG": 20.0})

    with fits.open(filepath) as hdul:
        np.testing.assert_array_equal(hdul[0].data, image)
        assert hdul[0].header["MAG"] == 20.0
