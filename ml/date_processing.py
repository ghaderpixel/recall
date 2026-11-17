import pandas as pd

def parse_nhtsa_date_column(series):
    values = series.astype("string").str.strip()
    values = values.replace("", pd.NA)
    parsed = pd.to_datetime(values, format="%Y%m%d", errors="coerce")
    remaining = parsed.isna() & values.notna()
    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(values.loc[remaining], errors="coerce")
    return parsed

def process_dates(compdf, rcldf):
    complaints = compdf.copy()
    recalls = rcldf.copy()

    complaints["COMPLAINT_DATE"] = parse_nhtsa_date_column(complaints["DATEA"])
    recalls["RECALL_DATE"] = parse_nhtsa_date_column(recalls["RCDATE"])

    return complaints, recalls

def create_date_quality_report(compdf, rcldf, output_dir):
    rows = []
    for name, df, col in [
        ("complaints", compdf, "COMPLAINT_DATE"),
        ("recalls", rcldf, "RECALL_DATE")
    ]:
        dates = df[col].dropna()
        rows.append({
            "dataset": name,
            "date_column": col,
            "total_rows": len(df),
            "valid_dates": int(df[col].notna().sum()),
            "invalid_or_missing_dates": int(df[col].isna().sum()),
            "earliest_date": dates.min() if len(dates) else pd.NaT,
            "latest_date": dates.max() if len(dates) else pd.NaT
        })
    pd.DataFrame(rows).to_csv(output_dir / "date_quality_report.csv", index=False)
