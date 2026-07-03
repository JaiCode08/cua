"""CLI entry point for cua-driver Python wrapper.

This module is invoked when running:
    python -m cua_driver [args...]
or via the installed script:
    cua-driver [args...]
"""

import sys
from .wrapper import run_cua_driver


def main() -> None:
    """Main entry point for the cua-driver CLI."""

    if len(sys.argv) > 1 and sys.argv[1] == "flakiness":
        try:
            from flakiness.runner import main as flakiness_main
        except ImportError as e:
            print(
                f"Error: Missing required packages for flakiness runner.\nPlease run: uv sync\nDetails: {e}",
                file=sys.stderr,
            )
            sys.exit(1)
        sys.exit(flakiness_main(sys.argv[2:]))

    if len(sys.argv) > 1 and sys.argv[1] == "continue_replay":
        try:
            from replay.tool import cli as continue_replay_cli
        except ImportError as e:
            print(
                f"Error: Missing required packages for continue_replay.\nPlease run: uv sync\nDetails: {e}",
                file=sys.stderr,
            )
            sys.exit(1)
        # Click handles sys.exit natively
        continue_replay_cli(sys.argv[2:])

    exit_code = run_cua_driver()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
