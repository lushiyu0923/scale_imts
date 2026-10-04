from __future__ import annotations

"""Command-line dataset validation; no dataset files are written to this repo."""

import argparse
from pathlib import Path

import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scaleimts.preprocessing.validate import validate_classification_root, validate_forecasting_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate external ScaleIMTS dataset files.")
    parser.add_argument("--task", choices=["forecasting", "classification"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", type=int, default=1)
    args = parser.parse_args()
    if args.task == "forecasting":
        files = validate_forecasting_root(args.dataset, Path(args.data_root))
    else:
        files = validate_classification_root(args.dataset, Path(args.data_root), args.split)
    print(f"{args.dataset}: {len(files)} required file(s) found")
    for file in files:
        print(f"  {file}")


if __name__ == "__main__":
    main()
