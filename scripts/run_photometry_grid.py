"""
Run photometry over a grid of comet magnitudes and aperture radii, and
write all results to a CSV log.

nucleus_fraction is held fixed at the run_trial default for now.
"""
import csv
import itertools
from pathlib import Path

from lsst_aot.experiments import FIELDNAMES, run_trial

OUTPUT_FILE = Path(__file__).resolve().parent.parent / "output" / "photometry_grid.csv"

MAGS = [18.0, 19.0, 20.0, 21.0, 22.0]
APERTURE_RADII_ARCSEC = [1.2, 2.4, 3.6, 4.8]
N_REPEATS = 10  # noise draws per (mag, aperture_radius) combination


def main():
    rows = []
    for mag, aperture_radius_arcsec in itertools.product(MAGS, APERTURE_RADII_ARCSEC):
        for repeat in range(N_REPEATS):
            rows.append(
                run_trial(mag, aperture_radius_arcsec=aperture_radius_arcsec, seed=repeat)
            )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
