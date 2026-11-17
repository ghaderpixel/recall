import pandas as pd

MATCH_COLUMNS = ["MAKETXT", "MODELTXT", "VEHICLE_YEAR", "COMPONENT_KEY"]

def prepare_matching_keys(compdf, rcldf):
    complaints = compdf.copy()
    recalls = rcldf.copy()

    for column in ["MAKETXT", "MODELTXT", "COMPONENT_KEY"]:
        complaints[column] = complaints[column].replace("", pd.NA)
        recalls[column] = recalls[column].replace("", pd.NA)

    complaints["MATCH_KEY_COMPLETE"] = complaints[MATCH_COLUMNS].notna().all(axis=1)
    recalls["MATCH_KEY_COMPLETE"] = recalls[MATCH_COLUMNS].notna().all(axis=1)
    return complaints, recalls

def create_matching_quality_report(compdf, rcldf, output_dir):
    complaint_complete = compdf[MATCH_COLUMNS].notna().all(axis=1)
    recall_complete = rcldf[MATCH_COLUMNS].notna().all(axis=1)

    recall_keys = set(
        map(tuple, rcldf.loc[recall_complete, MATCH_COLUMNS].itertuples(index=False, name=None))
    )
    complaint_keys = list(
        map(tuple, compdf.loc[complaint_complete, MATCH_COLUMNS].itertuples(index=False, name=None))
    )
    exact_key_exists = pd.Series(
        [key in recall_keys for key in complaint_keys],
        index=compdf.index[complaint_complete]
    )

    rows = [
        {"metric": "complaints_total", "value": len(compdf)},
        {"metric": "complaints_with_complete_match_key", "value": int(complaint_complete.sum())},
        {"metric": "complaints_missing_one_or_more_match_fields", "value": int((~complaint_complete).sum())},
        {"metric": "recalls_total", "value": len(rcldf)},
        {"metric": "recalls_with_complete_match_key", "value": int(recall_complete.sum())},
        {"metric": "complaints_with_exact_key_present_in_recall_data", "value": int(exact_key_exists.sum())},
        {"metric": "complaints_without_exact_key_present_in_recall_data", "value": int((~exact_key_exists).sum())}
    ]
    pd.DataFrame(rows).to_csv(output_dir / "matching_quality_report.csv", index=False)
