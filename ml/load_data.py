import csv
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
COMPLAINTS_FILE = RAW_DIR / "FLAT_CMPL.txt"
RECALLS_FILE = RAW_DIR / "FLAT_RCL_POST_2010.txt"

COMPLAINT_COLUMNS = [
    "CMPLID", "ODINO", "MFR_NAME", "MAKETXT", "MODELTXT",
    "YEARTXT", "CRASH", "FAILDATE", "FIRE", "INJURED",
    "DEATHS", "COMPDESC", "CITY", "STATE", "VIN",
    "DATEA", "LDATE", "MILES", "OCCURENCES", "CDESCR",
    "CMPL_TYPE", "POLICE_RPT_YN", "PURCH_DT", "ORIG_OWNER_YN",
    "ANTI_BRAKES_YN", "CRUISE_CONT_YN", "NUM_CYLS",
    "DRIVE_TRAIN", "FUEL_SYS", "FUEL_TYPE", "TRANS_TYPE",
    "VEH_SPEED", "DOT", "TIRE_SIZE", "LOC_OF_TIRE",
    "TIRE_FAIL_TYPE", "ORIG_EQUIP_YN", "MANUF_DT",
    "SEAT_TYPE", "RESTRAINT_TYPE", "DEALER_NAME",
    "DEALER_TEL", "DEALER_CITY", "DEALER_STATE", "DEALER_ZIP",
    "PROD_TYPE", "REPAIRED_YN", "MEDICAL_ATTN",
    "VEHICLES_TOWED_YN", "STATE_OF_INCIDENT", "VEHICLE_OPERATOR"
]

RECALL_COLUMNS = [
    "RECORD_ID", "CAMPNO", "MAKETXT", "MODELTXT",
    "YEARTXT", "MFGCAMPNO", "COMPNAME", "MFGNAME",
    "BGMAN", "ENDMAN", "RCLTYPECD", "POTAFF", "ODATE",
    "INFLUENCED_BY", "MFGTXT", "RCDATE", "DATEA",
    "RPNO", "FMVSS", "DESC_DEFECT", "CONEQUENCE_DEFECT",
    "CORRECTIVE_ACTION", "NOTES", "RCL_CMPT_ID",
    "MFR_COMP_NAME", "MFR_COMP_DESC", "MFR_COMP_PTNO",
    "DO_NOT_DRIVE", "PARK_OUTSIDE"
]

def _read_flat_file(path: Path, expected_columns: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find {path}. Put the original NHTSA file in data/raw/."
        )
    df = pd.read_csv(
        path, sep="\t", header=None, dtype="string",
        encoding="utf-8", quoting=csv.QUOTE_NONE,
        low_memory=False
    )
    if df.shape[1] != expected_columns:
        raise ValueError(
            f"{path.name}: expected {expected_columns} columns, "
            f"found {df.shape[1]}. Check the source file/layout."
        )
    return df

def load_datasets():
    print("Loading NHTSA flat files...")
    complaints = _read_flat_file(COMPLAINTS_FILE, len(COMPLAINT_COLUMNS))
    recalls = _read_flat_file(RECALLS_FILE, len(RECALL_COLUMNS))
    complaints.columns = COMPLAINT_COLUMNS
    recalls.columns = RECALL_COLUMNS
    print(f"Complaints loaded: {len(complaints):,}")
    print(f"Recall records loaded: {len(recalls):,}")
    return complaints, recalls