"""Manifest verification command."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from lrp.operations import verify_manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lrp verify")
    parser.add_argument("--manifest", required=True)
    arguments = parser.parse_args(argv)
    result = verify_manifest(arguments.manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1
