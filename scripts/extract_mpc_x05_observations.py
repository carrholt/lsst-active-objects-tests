"""
Pull one object's Rubin (station X05) observations out of the bulk MPC/SBN
observation archive (`data/obs_sbn_X05_full.csv`, ~7.8M rows) and write them
to a small per-object CSV for use in the DP2 precovery notebook section.

Usage:
    python scripts/extract_mpc_x05_observations.py "2001 HL69"
"""
import argparse
import csv
from pathlib import Path

from astropy.time import Time

SOURCE_FILE = Path(__file__).resolve().parent.parent / "data" / "obs_sbn_X05_full.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "precovery"
OUTPUT_FIELDS = ["mjd", "ra", "dec", "mag", "band", "status", "permid", "provid", "obstime_text"]


def extract(designation):
    rows = []
    with open(SOURCE_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["permid"] == designation or row["provid"] == designation:
                rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("designation", help="MPC permid or provid, e.g. '2001 HL69'")
    args = parser.parse_args()

    rows = extract(args.designation)
    print(f"Found {len(rows)} X05 observations of {args.designation!r}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{args.designation.replace(' ', '_')}_x05_obs.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            mjd = Time(row["obstime_text"]).mjd
            writer.writerow({
                "mjd": mjd,
                "ra": row["ra"],
                "dec": row["dec"],
                "mag": row["mag"],
                "band": row["band"],
                "status": row["status"],
                "permid": row["permid"],
                "provid": row["provid"],
                "obstime_text": row["obstime_text"],
            })

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
