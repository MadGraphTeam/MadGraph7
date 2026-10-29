# SPDX-License-Identifier: 0BSD
"""Select one complete rational-completion branch at evaluation-call entry."""

from __future__ import annotations

from dataclasses import dataclass

from pyamplicol.config import R2Strategy


@dataclass(frozen=True, slots=True)
class R2Selection:
    requested: R2Strategy
    effective: R2Strategy
    reason: str
