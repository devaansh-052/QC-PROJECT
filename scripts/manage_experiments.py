"""Create a new immutable research snapshot, or verify an existing snapshot."""
import argparse
from pathlib import Path
from quantum_fraud.evaluation.registry import build_registry, validate_registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    print(validate_registry(args.verify) if args.verify else build_registry())


if __name__ == "__main__":
    main()
