"""
Run PSF, 2.4", and 5.0" aperture photometry over a grid of comet
magnitudes at fixed (0.75") seeing, once without noise and once with,
and write each to its own CSV log (one row per model).

nucleus_fraction is held fixed at the run_trial default for now.
"""
import csv
from pathlib import Path

from lsst_aot.experiments import fieldnames_for, run_trial

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

MAGS = [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0]
APERTURE_RADII_ARCSEC = [2.4, 5.0]
N_REPEATS = 10  # noise draws per mag, noisy run only


def write_rows(rows, filename):
    filepath = OUTPUT_DIR / filename
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_for(APERTURE_RADII_ARCSEC))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {filepath}")


def main():
    no_noise_rows = [
        run_trial(mag, aperture_radii_arcsec=APERTURE_RADII_ARCSEC, sky_mag=None)
        for mag in MAGS
    ]
    write_rows(no_noise_rows, "photometry_grid_no_noise.csv")

    noisy_rows = [
        run_trial(mag, aperture_radii_arcsec=APERTURE_RADII_ARCSEC, seed=repeat)
        for mag in MAGS
        for repeat in range(N_REPEATS)
    ]
    write_rows(noisy_rows, "photometry_grid_noisy.csv")


if __name__ == "__main__":
    main()
