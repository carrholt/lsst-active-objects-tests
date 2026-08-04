"""
Generate a model comet FITS image and save it to data/model_comets/.
"""
from pathlib import Path

from lsst_aot.models import make_model_comet, save_model_comet

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "model_comets"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mag = 20.0
    nucleus_fraction = 0.05
    image = make_model_comet(mag=mag, nucleus_fraction=nucleus_fraction)

    filepath = OUTPUT_DIR / f"model_comet_mag{mag:.1f}.fits"
    save_model_comet(
        image,
        filepath,
        header_extra={"MAG": mag, "NUCFRAC": nucleus_fraction},
    )
    print(f"Saved {filepath}")


if __name__ == "__main__":
    main()
