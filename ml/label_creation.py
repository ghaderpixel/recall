import numpy as np
import pandas as pd
from .vehicle_matching import MATCH_COLUMNS

def create_recall_lookup(rcldf):
    valid = rcldf.dropna(subset=MATCH_COLUMNS + ["RECALL_DATE"])
    lookup = {}
    for key, dates in valid.groupby(MATCH_COLUMNS, sort=False)["RECALL_DATE"]:
        lookup[key] = np.sort(dates.to_numpy(dtype="datetime64[ns]"))
    return lookup

def create_labels(compdf, rcldf):
    """
    Create an exact-key baseline target:
    1 if a recall for the same make/model/year/component occurs strictly
    after the complaint and no later than one calendar year afterward;
    0 if no such recall is found within the observed date window;
    NA if the record is not eligible for labeling.

    IMPORTANT: This is a baseline labeling rule, not proof that the complaint
    caused the recall. Global min/max dates do not prove uninterrupted source
    coverage. Validate date definitions and coverage before final research use.
    """
    complaints = compdf.copy()
    lookup = create_recall_lookup(rcldf)

    recall_dates = rcldf["RECALL_DATE"].dropna()
    if recall_dates.empty:
        raise ValueError("No valid recall dates found; cannot create labels.")

    coverage_start = recall_dates.min()
    coverage_end = recall_dates.max()

    complaints["RECALL_WITHIN_12_MONTHS"] = pd.Series(
        pd.NA, index=complaints.index, dtype="Int64"
    )
    complaints["MATCHED_RECALL_DATE"] = pd.NaT

    valid_key = complaints[MATCH_COLUMNS].notna().all(axis=1)
    window_end = complaints["COMPLAINT_DATE"] + pd.DateOffset(years=1)

    eligible = (
        complaints["COMPLAINT_DATE"].notna()
        & valid_key
        & (complaints["COMPLAINT_DATE"] >= coverage_start)
        & (window_end <= coverage_end)
    )

    complaints.loc[eligible, "RECALL_WITHIN_12_MONTHS"] = 0

    eligible_groups = complaints.loc[eligible].groupby(
        MATCH_COLUMNS, sort=False
    ).groups

    for key, indices in eligible_groups.items():
        dates = lookup.get(key)
        if dates is None or len(dates) == 0:
            continue

        indices = list(indices)
        complaint_dates = complaints.loc[indices, "COMPLAINT_DATE"].to_numpy(
            dtype="datetime64[ns]"
        )
        ends = (
            complaints.loc[indices, "COMPLAINT_DATE"] + pd.DateOffset(years=1)
        ).to_numpy(dtype="datetime64[ns]")

        positions = np.searchsorted(dates, complaint_dates, side="right")
        has_candidate = positions < len(dates)

        candidates = np.full(len(indices), np.datetime64("NaT"), dtype="datetime64[ns]")
        candidates[has_candidate] = dates[positions[has_candidate]]

        positive = has_candidate & (candidates <= ends)
        if positive.any():
            positive_indices = np.asarray(indices)[positive]
            complaints.loc[positive_indices, "RECALL_WITHIN_12_MONTHS"] = 1
            complaints.loc[positive_indices, "MATCHED_RECALL_DATE"] = pd.to_datetime(
                candidates[positive]
            )

    return complaints

def create_label_distribution_report(compdf, output_dir):
    counts = compdf["RECALL_WITHIN_12_MONTHS"].value_counts(dropna=False)
    total = len(compdf)
    report = pd.DataFrame({
        "label": ["positive_1", "negative_0", "unknown_NA"],
        "count": [
            int((compdf["RECALL_WITHIN_12_MONTHS"] == 1).sum()),
            int((compdf["RECALL_WITHIN_12_MONTHS"] == 0).sum()),
            int(compdf["RECALL_WITHIN_12_MONTHS"].isna().sum())
        ]
    })
    report["percentage_of_all_complaints"] = (
        report["count"] / total * 100 if total else 0
    )
    report.to_csv(output_dir / "label_distribution_report.csv", index=False)
