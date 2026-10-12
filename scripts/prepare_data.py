"""Phase 1: validate raw data and save a uniquely named quality report."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4

from quantum_fraud.data import DatasetError, load_paysim
from quantum_fraud.utils.config import load_config
from quantum_fraud.utils.logging import configure_logging


def main() -> int:
    """Validate the supplied dataset; exit 2 on a configuration/data/output error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/base.yaml")
    parser.add_argument("--data", type=Path, help="Override CSV path; relative to current working directory")
    parser.add_argument("--output-dir", type=Path, help="Override report directory")
    args = parser.parse_args()
    logger = configure_logging()
    try:
        config = load_config(args.config)
        logger = configure_logging(config.log_level)
        source = (args.data if args.data is not None else config.data_path).expanduser().resolve()
        output_dir = (args.output_dir if args.output_dir is not None else config.results_dir / "metrics").expanduser().resolve()
        raw_dir = Path(__file__).resolve().parents[1] / "data/raw"
        if output_dir.is_relative_to(raw_dir) or (source.parent.name.lower() == "raw" and output_dir.is_relative_to(source.parent)):
            raise ValueError("Report output must be outside data/raw/; use results/metrics/")
        frame, report = load_paysim(source)
        del frame  # Only the small report is needed beyond validation.
        now = datetime.now(timezone.utc)
        report["validated_at_utc"] = now.isoformat()
        report["scope"] = "full_file_validation_only_no_cleaning"
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / f"dataset_validation_{now:%Y%m%dT%H%M%S}_{uuid4().hex}.json"
        with output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        for warning in report["warnings"]:
            logger.warning(warning)
        print(output.resolve())
        return 0
    except (ValueError, OSError) as exc:
        logger.error("%s", exc)
        if isinstance(exc, DatasetError) and exc.report:
            print(json.dumps(exc.report, indent=2, allow_nan=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
