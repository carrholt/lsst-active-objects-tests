"""
Run PSF and aperture photometry on a single model comet and append a row
of results to a CSV log.
"""
import csv
from pathlib import Path

from lsst_aot.experiments import FIELDNAMES, run_trial

OUTPUT_FILE = Path(__file__).resolve().parent.parent / "output" / "photometry_results.csv"


def main():
    row = run_trial(mag=20.0, nucleus_fraction=0.05, aperture_radius_arcsec=2.4)

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
