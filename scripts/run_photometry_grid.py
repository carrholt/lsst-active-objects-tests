"""
Run photometry over a grid of comet magnitudes, measuring every aperture
radius on each model, and write all results to a CSV log (one row per
model).

nucleus_fraction is held fixed at the run_trial default for now.
"""
import csv
from pathlib import Path

from lsst_aot.experiments import fieldnames_for, run_trial

OUTPUT_FILE = Path(__file__).resolve().parent.parent / "output" / "photometry_grid.csv"

MAGS = [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0]
APERTURE_RADII_ARCSEC = [0.6, 1.2, 1.8, 2.4, 3.4, 5.0, 7.0, 10.0, 14.0]  # 3,6,9,12,17,25,35,50,70 px
N_REPEATS = 10  # noise draws per mag

# Largest aperture radius is 70 px; shape needs a half-width comfortably
# bigger than that so the aperture isn't clipped at the image edge.
SHAPE = (181, 181)


def main():
    rows = []
    for mag in MAGS:
        for repeat in range(N_REPEATS):
            rows.append(
                run_trial(
                    mag,
                    aperture_radii_arcsec=APERTURE_RADII_ARCSEC,
                    shape=SHAPE,
                    seed=repeat,
                )
            )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_for(APERTURE_RADII_ARCSEC))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
