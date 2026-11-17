"""
Reserved for model training.
Suggested input:
    data/processed/recall_prediction_dataset.parquet
Define an explicit feature list and exclude target leakage fields such as
MATCHED_RECALL_DATE, RECALL_DATE, DESC_DEFECT, CORRECTIVE_ACTION,
and other information unavailable at complaint prediction time.
"""
