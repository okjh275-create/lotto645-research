"""Repository backup command."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from lrp.operations import create_backup


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lrp backup")
    parser.add_argument("--source", default=".")
    parser.add_argument("--output", default="backups")
    parser.add_argument("--label", default="lotto645-research")
    arguments = parser.parse_args(argv)
    result = create_backup(source_root=arguments.source, destination_root=arguments.output, label=arguments.label)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
