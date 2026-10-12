from pathlib import Path
import argparse
import json
from quantum_fraud.quantum.hardware import prepare_hardware, submit_hardware, collect_hardware

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare, explicitly submit, or collect a tiny IBM hardware job")
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--data", type=Path, required=True)
    prepare.add_argument("--vqc-run", type=Path, required=True)
    submit = sub.add_parser("submit")
    submit.add_argument("--bundle", type=Path, required=True)
    submit.add_argument("--backend", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        print(prepare_hardware(args.data, args.vqc_run, Path("results/hardware")))
    else:
        result = submit_hardware(args.bundle, args.backend) if args.action == "submit" else collect_hardware(args.bundle)
        print(json.dumps(result, indent=2, default=str))
