"""
Plot recovered PSF/aperture magnitude vs. true magnitude from the
no-noise and noisy photometry grids (output/photometry_grid_*.csv).
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
PLOT_FILE = OUTPUT_DIR / "plots" / "photometry_grid.png"

MEASURES = [
    ("psf_mag", "PSF"),
    ("aperture_r2p4_mag", '2.4" aperture'),
    ("aperture_r5_mag", '5.0" aperture'),
]
COLORS = {"PSF": "tab:blue", '2.4" aperture': "tab:orange", '5.0" aperture': "tab:green"}


def load_rows(filename):
    with open(OUTPUT_DIR / filename) as f:
        return list(csv.DictReader(f))


def group_by_mag(rows, field):
    grouped = defaultdict(list)
    for row in rows:
        grouped[float(row["mag"])].append(float(row[field]))
    mags = sorted(grouped)
    return mags, [grouped[m] for m in mags]


def main():
    no_noise_rows = load_rows("photometry_grid_no_noise.csv")
    noisy_rows = load_rows("photometry_grid_noisy.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)

    for field, label in MEASURES:
        mags, values = group_by_mag(no_noise_rows, field)
        ax1.plot(mags, [v[0] for v in values], "o-", color=COLORS[label], label=label)

        mags, values = group_by_mag(noisy_rows, field)
        means = [np.mean(v) for v in values]
        stds = [np.std(v) for v in values]
        ax2.errorbar(mags, means, yerr=stds, fmt="o-", color=COLORS[label], capsize=3, label=label)

    all_mags, _ = group_by_mag(no_noise_rows, "mag")
    for ax, title in [(ax1, "No noise"), (ax2, "With sky noise (mean ± std, 10 draws)")]:
        ax.plot(all_mags, all_mags, "k--", linewidth=1, label="true mag")
        ax.set_xlabel("True total magnitude")
        ax.set_title(title)
        ax.invert_yaxis()
        ax.legend()

    ax1.set_ylabel("Recovered magnitude")
    fig.tight_layout()

    PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_FILE, dpi=150)
    print(f"Wrote {PLOT_FILE}")


if __name__ == "__main__":
    main()
