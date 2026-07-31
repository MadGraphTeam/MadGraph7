# SPDX-License-Identifier: 0BSD
"""Installed worker entry point for packaged profiling campaigns."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
