# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import io

import pytest

from pyamplicol.cli import LicenseRequestInvocation, parse_cli


def test_trial_request_parses_without_entering_run_configuration() -> None:
    invocation = parse_cli(
        [
            "request-symbolica-trial-license",
            "--name",
            "Ada",
            "--email",
            "ada@example.org",
            "--organization",
            "Institute",
            "--yes",
        ]
    )
    assert isinstance(invocation, LicenseRequestInvocation)
    assert invocation.kind == "trial"
    assert invocation.assume_yes


def test_noninteractive_request_requires_all_fields_and_yes() -> None:
    invocation = LicenseRequestInvocation(
        kind="hobbyist",
        name="Ada",
        email=None,
        assume_yes=True,
    )
    with pytest.raises(ValueError, match="missing email"):
        invocation.run(stdin=io.StringIO(), stdout=io.StringIO())
