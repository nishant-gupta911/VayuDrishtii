#!/usr/bin/env python3
"""VayuDrishti canonical training entrypoint."""

from __future__ import annotations

from typing import Optional, List

from src.train import main as train_main


def main(argv: Optional[List[str]] = None) -> None:
    train_main(argv)


if __name__ == "__main__":
    main()
