"""
Run PSF and aperture photometry on a model comet and append a row of
results to a CSV log.
"""
import csv
from pathlib import Path

from lsst_aot.models import make_model_comet
from lsst_aot.photometry import (
    aperture_flux_fraction,
    aperture_photometry,
    psf_flux_fraction,
    psf_photometry,
)
from lsst_aot.utils import ab_mag_to_njy, njy_to_ab_mag

OUTPUT_FILE = Path(__file__).resolve().parent.parent / "output" / "photometry_results.csv"

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
]


def main():
    mag = 20.0
    nucleus_fraction = 0.05
    aperture_radius_arcsec = 2.4

    image = make_model_comet(mag=mag, nucleus_fraction=nucleus_fraction)
    center = (image.shape[0] // 2, image.shape[1] // 2)
    total_flux = ab_mag_to_njy(mag)

    psf_flux = psf_photometry(image, center)
    aperture_flux = aperture_photometry(image, center, radius_arcsec=aperture_radius_arcsec)

    row = {
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
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_header = not OUTPUT_FILE.exists()
    with open(OUTPUT_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"Appended row to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
