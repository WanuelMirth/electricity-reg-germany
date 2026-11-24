"""Generate a Markdown leaderboard from stored validation or test JSON summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.tracking.mlflow_utils import render_leaderboard  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for building the Markdown leaderboard."""
    parser = argparse.ArgumentParser(
        description="Build a Markdown leaderboard from stored JSON model summaries."
    )
    parser.add_argument(
        "--split",
        choices=["val_2023", "test_2024"],
        default="val_2023",
        help="Result split to summarize (default: val_2023).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Directory containing <family>_<task>.json files (default: results/<split>).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output Markdown path (default: results/validation_leaderboard.md "
            "or results/test_leaderboard.md)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Render leaderboard Markdown for the requested split."""
    args = parse_args()
    results_dir = args.results_dir or (REPO_ROOT / "results" / args.split)
    default_filename = (
        "validation_leaderboard.md" if args.split == "val_2023" else "test_leaderboard.md"
    )
    output_path = args.output or (REPO_ROOT / "results" / default_filename)

    written = render_leaderboard(results_dir, output_path, split=args.split)
    print(f"Leaderboard written: {written}")


if __name__ == "__main__":
    main()
