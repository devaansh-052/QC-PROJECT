"""Read the published PaySim CSV without cleaning or altering source records."""

import csv
import hashlib
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantum_fraud.data.policy import TARGET_COLUMN, primary_feature_policy

LOGGER = logging.getLogger(__name__)
NUMERIC_COLUMNS = (
    "step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
)
REQUIRED_COLUMNS = (
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
)
TRANSACTION_TYPES = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}


class DatasetError(ValueError):
    """Actionable data error, optionally carrying a complete quality report."""

    def __init__(self, message: str, report: dict[str, Any] | None = None):
        super().__init__(message)
        self.report = report


def validate_paysim(frame: pd.DataFrame) -> dict[str, Any]:
    """Inspect a frame without mutation; reject invalid labels and numeric values.

    Missing feature values and duplicate rows are reported, not silently removed.
    A single-class file is accepted with a warning, not fit for comparative evaluation.
    """
    if not frame.columns.is_unique:
        raise DatasetError("Duplicate column names are not supported")
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise DatasetError(f"Missing required PaySim columns: {', '.join(missing)}")
    if frame.empty:
        raise DatasetError("Dataset contains no transaction rows")
    report: dict[str, Any] = {
        "shape": {"rows": len(frame), "columns": len(frame.columns)},
        "dtypes": {str(key): str(value) for key, value in frame.dtypes.items()},
        "missing_values": {str(key): int(value) for key, value in frame.isna().sum().items()},
        "duplicate_rows": int(frame.duplicated().sum()),
        "extra_columns": sorted(set(frame.columns) - set(REQUIRED_COLUMNS)),
        "target_distribution": {}, "fraud_percentage": None,
        "target_column": TARGET_COLUMN,
        "feature_policy": primary_feature_policy(),
        "errors": [], "warnings": [],
    }
    errors = report["errors"]
    for column in NUMERIC_COLUMNS:
        series = frame[column].dropna()
        if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            errors.append(f"{column} must be numeric")
        elif not np.isfinite(series.to_numpy(dtype=float)).all() or (series < 0).any():
            errors.append(f"{column} must contain finite, nonnegative values")
        elif column == "step" and ((series < 1) | (series % 1 != 0)).any():
            errors.append("step must contain positive integers")
    for column in ("isFraud", "isFlaggedFraud"):
        series = frame[column]
        if series.isna().any() or not pd.api.types.is_numeric_dtype(series) or not series.isin([0, 1]).all():
            errors.append(f"{column} must contain only nonmissing binary labels 0 and 1")
    target = frame[TARGET_COLUMN]
    if target.notna().all() and pd.api.types.is_numeric_dtype(target) and target.isin([0, 1]).all():
        report["target_distribution"] = {str(label): int((target == label).sum()) for label in (0, 1)}
        report["fraud_percentage"] = float(target.mean() * 100)
        if target.nunique() < 2:
            report["warnings"].append("Only one target class; classification evaluation requires both classes")
    transaction_types = frame["type"].dropna()
    if not transaction_types.isin(TRANSACTION_TYPES).all():
        errors.append("type contains unsupported transaction categories")
    for column in ("nameOrig", "nameDest"):
        if not frame[column].dropna().map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            errors.append(f"{column} must contain nonempty string account identifiers")
    if any(report["missing_values"].values()):
        report["warnings"].append("Missing values found; handle features in training-only preprocessing")
    if report["duplicate_rows"]:
        report["warnings"].append("Duplicate rows found; investigate before defining data partitions")
    if report["extra_columns"]:
        report["warnings"].append("Extra columns retained for inspection; not automatically approved as model inputs")
    if errors:
        raise DatasetError("Invalid PaySim dataset: " + "; ".join(errors), report)
    steps = frame["step"].dropna()
    report["step_range"] = {"min": int(steps.min()), "max": int(steps.max())} if not steps.empty else None
    report["warnings"].append("Balance columns are retained for validation only; exclude all four and derived balance features from primary model inputs (see docs/DATASET.md)")
    return report


def load_paysim(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the entire CSV and return validation plus SHA-256 file provenance.

    Full-file quality/duplicate inspection needs RAM for the entire frame. This
    function never silently samples or substitutes another file.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise DatasetError(f"PaySim CSV not found: {source}. Place paysim.csv in data/raw/ or pass --data PATH; see data/README.md.")
    if source.suffix.lower() != ".csv":
        raise DatasetError("Expected an extracted .csv file; extract the downloaded archive first")
    try:
        before = source.stat()
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            header = next(csv.reader(handle), [])
        if not header:
            raise DatasetError("Dataset file is empty")
        if len(header) != len(set(header)):
            raise DatasetError("Duplicate CSV column names; repair the source schema")
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ParserWarning)
            frame = pd.read_csv(source, encoding="utf-8-sig", low_memory=False, index_col=False)
        report = validate_paysim(frame)
        with source.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise DatasetError("Dataset changed during validation; retry with an immutable file")
    except DatasetError:
        raise
    except (OSError, UnicodeError, csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError, pd.errors.ParserWarning) as exc:
        raise DatasetError(f"Cannot read PaySim CSV {source}: {exc}") from exc
    except MemoryError as exc:
        raise DatasetError("Insufficient RAM for full-file validation; use a machine with more RAM. No rows were silently sampled.") from exc
    report["source"] = {"path": str(source), "sha256": digest, "bytes": before.st_size, "dataset": "PaySim", "kind": "published_synthetic"}
    LOGGER.info("Validated PaySim: rows=%d columns=%d duplicates=%d target=%s", len(frame), len(frame.columns), report["duplicate_rows"], report["target_distribution"])
    return frame, report
