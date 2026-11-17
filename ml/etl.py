from pathlib import Path
import pandas as pd

from .load_data import load_datasets
from .data_cleaning import clean_datasets, create_missingness_report
from .date_processing import process_dates, create_date_quality_report
from .vehicle_matching import prepare_matching_keys, create_matching_quality_report
from .label_creation import create_labels, create_label_distribution_report

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

def export_positive_samples(compdf, rcldf, output_dir, limit=100):
    positives = compdf.loc[
        compdf["RECALL_WITHIN_12_MONTHS"] == 1
    ].head(limit).copy()

    recall_details = rcldf[
        ["MAKETXT", "MODELTXT", "VEHICLE_YEAR", "COMPONENT_KEY",
         "RECALL_DATE", "RECORD_ID", "CAMPNO", "COMPNAME"]
    ].rename(columns={"COMPNAME": "RECALL_COMPONENT"})

    samples = positives.merge(
        recall_details,
        left_on=["MAKETXT", "MODELTXT", "VEHICLE_YEAR", "COMPONENT_KEY",
                 "MATCHED_RECALL_DATE"],
        right_on=["MAKETXT", "MODELTXT", "VEHICLE_YEAR", "COMPONENT_KEY",
                  "RECALL_DATE"],
        how="left"
    )
    samples.to_csv(output_dir / "positive_match_samples.csv", index=False)

def create_etl_summary(compdf, rcldf, output_dir):
    labels = compdf["RECALL_WITHIN_12_MONTHS"]
    valid_recall_dates = rcldf["RECALL_DATE"].dropna()
    summary = [
        ("total_complaints", len(compdf)),
        ("total_recall_records", len(rcldf)),
        ("positive_labels", int((labels == 1).sum())),
        ("negative_labels", int((labels == 0).sum())),
        ("unknown_labels", int(labels.isna().sum())),
        ("recall_date_coverage_start", str(valid_recall_dates.min()) if len(valid_recall_dates) else ""),
        ("recall_date_coverage_end", str(valid_recall_dates.max()) if len(valid_recall_dates) else "")
    ]
    pd.DataFrame(summary, columns=["metric", "value"]).to_csv(
        output_dir / "etl_summary.csv", index=False
    )

def run_pipeline():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("RECALLRADAR ETL")
    print("=" * 60)

    complaints, recalls = load_datasets()
    complaints, recalls = clean_datasets(complaints, recalls)
    complaints, recalls = process_dates(complaints, recalls)
    complaints, recalls = prepare_matching_keys(complaints, recalls)

    # Quality reports are generated before label creation as well.
    create_missingness_report(complaints, recalls, PROCESSED_DIR)
    create_date_quality_report(complaints, recalls, PROCESSED_DIR)
    create_matching_quality_report(complaints, recalls, PROCESSED_DIR)

    complaints = create_labels(complaints, recalls)
    create_label_distribution_report(complaints, PROCESSED_DIR)
    create_etl_summary(complaints, recalls, PROCESSED_DIR)
    export_positive_samples(complaints, recalls, PROCESSED_DIR)

    # Preserve cleaned complaints separately from target/label columns.
    complaints_cleaned = complaints.drop(
        columns=["RECALL_WITHIN_12_MONTHS", "MATCHED_RECALL_DATE"],
        errors="ignore"
    )
    prediction_dataset = complaints.loc[
        complaints["RECALL_WITHIN_12_MONTHS"].notna()
    ].copy()

    complaints_cleaned.to_parquet(
        PROCESSED_DIR / "complaints_cleaned.parquet", index=False
    )
    recalls.to_parquet(
        PROCESSED_DIR / "recalls_cleaned.parquet", index=False
    )
    complaints.to_parquet(
        PROCESSED_DIR / "complaints_with_labels.parquet", index=False
    )
    prediction_dataset.to_parquet(
        PROCESSED_DIR / "recall_prediction_dataset.parquet", index=False
    )

    print("\nExported datasets:")
    for name in [
        "complaints_cleaned.parquet",
        "recalls_cleaned.parquet",
        "complaints_with_labels.parquet",
        "recall_prediction_dataset.parquet"
    ]:
        print(f" - {name}")

    print("\nETL completed. Review the CSV quality reports before modeling.")
    return complaints, recalls

if __name__ == "__main__":
    run_pipeline()
