"""Unified entrypoint for demo1 variants."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parent
VARIANT_TO_SCRIPT = {
    "numpy": DEMO_DIR / "numpy_fit.py",
    "pytorch": DEMO_DIR / "pytorch_fit.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run demo1 implementations from a single entrypoint."
    )
    parser.add_argument(
        "variant",
        nargs="?",
        choices=[*VARIANT_TO_SCRIPT.keys(), "all"],
        help="Which implementation to run. Use 'all' to run every variant in sequence.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available demo1 variants and exit.",
    )
    return parser.parse_args()


def list_variants() -> None:
    print("Available demo1 variants:")
    for variant, script_path in VARIANT_TO_SCRIPT.items():
        print(f"  - {variant:7s} -> {script_path.name}")


def run_variant(variant: str) -> int:
    script_path = VARIANT_TO_SCRIPT[variant]
    print(f"\n{'=' * 60}", flush=True)
    print(f"Running demo1 variant: {variant}", flush=True)
    print(f"Script: {script_path}", flush=True)
    print(f"{'=' * 60}\n", flush=True)
    completed = subprocess.run([sys.executable, str(script_path)], cwd=DEMO_DIR)
    return completed.returncode


def main() -> int:
    args = parse_args()

    if args.list:
        list_variants()
        return 0

    if args.variant is None:
        print("Please choose a variant or use --list.", file=sys.stderr)
        return 2

    variants = list(VARIANT_TO_SCRIPT) if args.variant == "all" else [args.variant]
    for variant in variants:
        exit_code = run_variant(variant)
        if exit_code != 0:
            return exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
