"""Root CLI adapter for historical model evaluation."""

from __future__ import annotations

import subprocess
import sys
from typing import Sequence


_TARGET_MODULE = (
    "tools.validation.run_model_evaluation"
)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Delegate model evaluation to the validated Project M CLI."""

    arguments = (
        list(argv)
        if argv is not None
        else list(sys.argv[1:])
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            _TARGET_MODULE,
            *arguments,
        ],
        check=False,
    )

    return int(
        completed.returncode
    )


if __name__ == "__main__":
    raise SystemExit(
        main()
    )