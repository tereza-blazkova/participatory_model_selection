"""
Where the pipeline reads its data and writes its results.

As shipped, everything points inside the repository: the synthetic data that
data/generate_fake_df.py and data/generate_pilot_snapshot.py write into data/, and
the results under results/. To run on other data, change DATA_ROOT (and the two
file names if yours differ); no other file needs editing.
"""
from pathlib import Path

## The repository root, derived from this file's location.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "data"

## The enrollment-period panel, and the October 2023 pilot snapshot.
DATA_PATH_PARQUET = DATA_ROOT / "fake_df.parquet"
PILOT_SNAPSHOT_PARQUET  = DATA_ROOT / "pilot_snapshot.parquet"

## Everything the pipeline writes lands under here.
RESULTS_PATH = PROJECT_ROOT / "results"

## Font used by the figure style.
HELVETICA = PROJECT_ROOT / "src" / "fonts" / "Helvetica.ttf"
