"""Manually promote `@challenger` to `@champion` in the MLflow Model Registry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.splits import TaskType  # noqa: E402
from src.tracking.mlflow_utils import (  # noqa: E402
    MLflowConfig,
    configure_mlflow,
    promote_challenger_to_champion,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for model alias promotion."""
    parser = argparse.ArgumentParser(
        description="Promote @challenger to @champion for a registered task model."
    )
    parser.add_argument(
        "--task",
        choices=["hourly", "daily", "all"],
        required=True,
        help="Task model to promote.",
    )
    parser.add_argument(
        "--keep-runner-up",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the previous @champion as @challenger (default: True).",
    )
    return parser.parse_args()


def main() -> None:
    """Promote @challenger -> @champion for the selected task(s)."""
    args = parse_args()
    config = MLflowConfig.from_env(REPO_ROOT)
    client = configure_mlflow(config)

    tasks: list[TaskType] = ["hourly", "daily"] if args.task == "all" else [args.task]
    for task in tasks:
        model_name = config.registered_model_name(task)
        aliases = promote_challenger_to_champion(
            client,
            model_name,
            keep_previous_as_challenger=args.keep_runner_up,
        )
        chall_label = f"v{aliases['challenger']}" if aliases["challenger"] else "None"
        print(f"{model_name}: @champion=v{aliases['champion']}, @challenger={chall_label}")


if __name__ == "__main__":
    main()
