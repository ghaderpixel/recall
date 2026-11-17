import re
import pandas as pd

def normalize_text(value):
    
    if pd.isna(value):
        return pd.NA
    value = str(value).upper().strip()
    value = re.sub(r"[^A-Z0-9]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value if value else pd.NA

def clean_datasets(compdf, rcldf):
    complaints = compdf.copy()
    recalls = rcldf.copy()

    for column in ["MAKETXT", "MODELTXT"]:
        complaints[column] = complaints[column].map(normalize_text).astype("string")
        recalls[column] = recalls[column].map(normalize_text).astype("string")

    complaints["COMPONENT_KEY"] = (
        complaints["COMPDESC"].map(normalize_text).astype("string")
    )
    recalls["COMPONENT_KEY"] = (
        recalls["COMPNAME"].map(normalize_text).astype("string")
    )

    complaints["VEHICLE_YEAR"] = pd.to_numeric(
        complaints["YEARTXT"], errors="coerce"
    ).astype("Int64")
    recalls["VEHICLE_YEAR"] = pd.to_numeric(
        recalls["YEARTXT"], errors="coerce"
    ).astype("Int64")

    for column in ["CMPLID", "ODINO"]:
        complaints[column] = complaints[column].astype("string").str.strip()
    for column in ["RECORD_ID", "CAMPNO", "RCL_CMPT_ID"]:
        recalls[column] = recalls[column].astype("string").str.strip()

    return complaints, recalls

def create_missingness_report(complaints, recalls, output_dir):
    rows = []
    for dataset_name, df in [("complaints", complaints), ("recalls", recalls)]:
        for column in df.columns:
            series = df[column]
            missing = int(series.isna().sum())
            if pd.api.types.is_string_dtype(series.dtype) or series.dtype == object:
                missing += int(series.astype("string").str.strip().eq("").fillna(False).sum())
            rows.append({
                "dataset": dataset_name,
                "column": column,
                "missing_count": missing,
                "missing_percentage": round(100 * missing / len(df), 2) if len(df) else 0.0
            })
    pd.DataFrame(rows).to_csv(output_dir / "missing_values_report.csv", index=False)
