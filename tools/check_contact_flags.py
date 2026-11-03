"""
Where is each pilot contact flag set, and what recall and precision does it imply?

The Faculty B flag is set at other faculties too, so it is not a record of who
that faculty contacted, and the paper evaluates Faculty B's operational model by ranking only. See
config.CONTACTS_RECORDED_FACULTIES.

Read-only. The by-faculty counts can be small, so do not publish this output.

Usage:  python -m tools.check_contact_flags
"""
from src.data import load_pilot_snapshot, load_pilot_snapshot_unfiltered
from src.pilot_snapshot_comparison import _FACULTY_CONTACT


def main() -> None:
    t = load_pilot_snapshot()
    u = load_pilot_snapshot_unfiltered()
    print("Rows:", len(t), "| unfiltered:", len(u),
          "| duplicate enrollments:", t["enrollment"].duplicated().sum())

    for faculty, col in _FACULTY_CONTACT.items():
        print(f"\n=== {col} ===")
        print("Sum, whole snapshot:", t[col].sum(), "| unfiltered:", u[col].sum(),
              "| missing:", t[col].isna().sum())
        print("Sum by faculty:", t.groupby("faculty_code")[col].sum().to_dict())
        print("Sum by faculty, unfiltered:", u.groupby("faculty_code")[col].sum().to_dict())
        e = t[t["faculty_code"] == faculty].groupby("enrollment").agg(
            flag=(col, "max"), dropout=("dropout", "max"))
        tp = int((e["flag"] * e["dropout"]).sum())
        print("Enrollments:", len(e), "| dropouts:", int(e["dropout"].sum()),
              "| flagged:", int(e["flag"].sum()), "| true positives:", tp)
        print("Recall:", round(tp / e["dropout"].sum(), 3),
              "| precision:", round(tp / e["flag"].sum(), 3))


if __name__ == "__main__":
    main()
