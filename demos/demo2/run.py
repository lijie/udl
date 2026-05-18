"""Unified entrypoint for demo2 variants."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parent
VARIANT_TO_SCRIPT = {
    "numpy": DEMO_DIR / "numpy_classification.py",
    "pytorch": DEMO_DIR / "pytorch_classification.py",
}
DATASETS = ["moons", "circles", "spiral"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run demo2 implementations from a single entrypoint.")
    parser.add_argument(
        "variant",
        nargs="?",
        choices=[*VARIANT_TO_SCRIPT.keys(), "all"],
        help="Which implementation to run. Use 'all' to run every variant in sequence.",
    )
    parser.add_argument(
        "--dataset",
        choices=["all", *DATASETS],
        default="all",
        help="Restrict execution to a specific dataset.",
    )
    parser.add_argument("--epochs", type=int, help="Override the epoch count passed to the variant script.")
    parser.add_argument("--no-show", action="store_true", help="Save figures without calling plt.show().")
    parser.add_argument("--list", action="store_true", help="List available variants and datasets, then exit.")
    return parser.parse_args()


def list_options() -> None:
    print("Available demo2 variants:")
    for variant, script_path in VARIANT_TO_SCRIPT.items():
        print(f"  - {variant:7s} -> {script_path.name}")
    print("Available datasets:")
    for dataset in DATASETS:
        print(f"  - {dataset}")


def build_command(script_path: Path, args: argparse.Namespace) -> list[str]:
    command = [sys.executable, str(script_path), "--dataset", args.dataset]
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    if args.no_show:
        command.append("--no-show")
    return command


def run_variant(variant: str, args: argparse.Namespace) -> int:
    script_path = VARIANT_TO_SCRIPT[variant]
    command = build_command(script_path, args)
    print(f"\n{'=' * 60}", flush=True)
    print(f"Running demo2 variant: {variant}", flush=True)
    print(f"Command: {' '.join(command)}", flush=True)
    print(f"{'=' * 60}\n", flush=True)
    completed = subprocess.run(command, cwd=DEMO_DIR)
    return completed.returncode


def main() -> int:
    args = parse_args()

    if args.list:
        list_options()
        return 0

    if args.variant is None:
        print("Please choose a variant or use --list.", file=sys.stderr)
        return 2

    variants = list(VARIANT_TO_SCRIPT) if args.variant == "all" else [args.variant]
    for variant in variants:
        exit_code = run_variant(variant, args)
        if exit_code != 0:
            return exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
