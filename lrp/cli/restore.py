"""Restore a repository backup."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from lrp.operations import restore_backup


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lrp restore")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = restore_backup(
            archive_path=args.archive,
            destination_root=args.destination,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "error_type": type(exc).__name__, "message": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
