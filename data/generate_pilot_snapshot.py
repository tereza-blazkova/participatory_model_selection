"""Generate data/pilot_snapshot.parquet — synthetic Oct 2023 EWS snapshot."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from paths.paths import DATA_PATH_PARQUET
from src.config import PILOT_CONTACT_THRESHOLD

rng = np.random.default_rng(42)

df = pd.read_parquet(DATA_PATH_PARQUET)

# Students active at Oct 2023: started 2019–2023, October EWS rows.
# Take the latest (highest period) October row per enrollment → one row per enrollment.
pilot_snapshot = (
    df[(df["ews_month"] == "oct") & (df["start_year"].between(2019, 2023))]
    .sort_values("period")
    .groupby("enrollment", as_index=False)
    .last()
    .copy()
)

# Stamp the Oct 2023 EWS date
pilot_snapshot["ews_date"] = pd.Timestamp("2023-10-15")

n = len(pilot_snapshot)
print(f"Rows: {n}  |  unique enrollments: {pilot_snapshot['enrollment'].nunique()}")

# points_total: 0–15, uniform random
pilot_snapshot["points_total"] = rng.integers(0, 16, size=n)

# contact_fac_b / contact_fac_a: pilot outreach programs run by FAC_B and FAC_A
# faculties respectively in Oct 2023. Only students enrolled at that faculty
# are eligible, and — as in the pilot design — contact follows the points rule:
# points_total above PILOT_CONTACT_THRESHOLD (src/config.py). The pipeline
# recovers the threshold from who was contacted, so a rule keeps that recovery
# meaningful where a random contact list would not.
eligible_score = pilot_snapshot["points_total"] > PILOT_CONTACT_THRESHOLD
pilot_snapshot["contact_fac_b"]  = ((pilot_snapshot["faculty_code"] == "FAC_B")  & eligible_score).astype(int)
pilot_snapshot["contact_fac_a"] = ((pilot_snapshot["faculty_code"] == "FAC_A") & eligible_score).astype(int)

print("contact_fac_b  :", pilot_snapshot["contact_fac_b"].value_counts().to_dict())
print("contact_fac_a :", pilot_snapshot["contact_fac_a"].value_counts().to_dict())
print("points_total :", pilot_snapshot["points_total"].min(), "–", pilot_snapshot["points_total"].max())
print(f"contact rule : points_total > {PILOT_CONTACT_THRESHOLD}")
print("columns      :", pilot_snapshot.columns.tolist())

out = Path(__file__).parent / "pilot_snapshot.parquet"
pilot_snapshot.to_parquet(out, index=False)
print(f"Saved → {out}")