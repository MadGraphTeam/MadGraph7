# SPDX-License-Identifier: 0BSD
"""The NGluon leading-colour driver's pure parts (M-FINAL MF-8G-3 (vi),
``tools/loop_reference/ngluon/run_ngluon.py``): the V/B conversion, the
exact rotation every point gets before NGluon sees it, and the comparison
with a ``stored_checks.py`` record. No NJet build, no pyAmpliCol generation.
"""

from __future__ import annotations

import importlib.util
import math
import os
from fractions import Fraction
from pathlib import Path

import pytest

pytestmark = pytest.mark.loop_fast

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "tools" / "loop_reference" / "ngluon" / "run_ngluon.py"


@pytest.fixture(scope="module")
def ngluon():
    spec = importlib.util.spec_from_file_location("run_ngluon", DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ratio_is_the_leading_colour_virtual_over_born(ngluon):
    # V/B = (alpha_s/4pi)[N_c S1/S0 + c0 + c1/eps]: c0 = -2 N_c/3 (FDH -> CDR
    # at n gluons), c1 = -(n-2) 11 N_c/3 (MS-bar coupling renormalisation).
    alpha_s, legs = 0.118, 7
    k = alpha_s / (4 * math.pi)
    assert ngluon.ratio(2.0, [0.0, 0.0, 0.0], legs, alpha_s) == pytest.approx(
        [k * -2.0, k * -55.0, 0.0], rel=1e-15
    )
    s1 = [0.5, -1.25, 3.0]
    finite, single, double = ngluon.ratio(2.0, s1, legs, alpha_s)
    assert finite == pytest.approx(k * (3 * 0.25 - 2.0), rel=1e-15)
    assert single == pytest.approx(k * (3 * -0.625 - 55.0), rel=1e-15)
    assert double == pytest.approx(k * 3 * 1.5, rel=1e-15)


def test_rotation_is_exact_and_generic(ngluon):
    matrix = ngluon.rotation_matrix(ngluon.ROTATION)
    for i in range(3):
        for j in range(3):
            dot = sum(matrix[k][i] * matrix[k][j] for k in range(3))
            assert dot == (1 if i == j else 0)
    a, b, c = matrix
    det = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    assert det == 1
    # No rotated coordinate axis keeps a zero component (a beam on the z
    # axis leaves every axis and coordinate plane).
    assert all(entry != 0 for row in matrix for entry in row)
    # A massless 2 -> 2 point stays exactly on shell and momentum conserving.
    e = Fraction(1, 2)
    point = [
        [e, Fraction(0), Fraction(0), e],
        [e, Fraction(0), Fraction(0), -e],
        [e, Fraction(3, 10), Fraction(0), Fraction(2, 5)],
        [e, Fraction(-3, 10), Fraction(0), Fraction(-2, 5)],
    ]
    rotated = ngluon.rotate(point, matrix)
    for leg in rotated:
        assert leg[0] ** 2 == sum(x * x for x in leg[1:])
    for k in range(4):
        assert rotated[0][k] + rotated[1][k] == rotated[2][k] + rotated[3][k]
    assert rotated[0][3] != e
    with pytest.raises(ValueError):
        ngluon.rotation_matrix((0, 0, 0, 0))


def _checks(born, virtual, estimate, dd=None):
    record = {
        "checked": [
            {
                "index": 0,
                "born": born,
                "sectors": {"QCD^8": {"virtual": virtual, "estimate": estimate}},
            }
        ]
    }
    if dd is not None:
        record["dd"] = [{"index": 0, "sectors": {"QCD^8": {"double_double": dd}}}]
    return record


def test_compare_bounds_by_tolerance_and_estimate(ngluon):
    born = 2.0
    virtual = [4.0, -2.0, 1.0]
    estimate = [2e-8, 2e-8, 2e-12]
    theirs = _checks(born, virtual, estimate, dd=[4.0 + 1e-9, -2.0, 1.0])
    exact = [v / born for v in virtual]

    def ours(values, precision="double"):
        return {
            "precision": precision,
            "rotation": [5, 3, -2, 7],
            "rows": [{"index": 0, "virtual_over_born": values}],
        }

    # Within 10 E (over the Born) of pyAmpliCol's binary64 value: passes.
    near = [exact[0] + 5e-8, exact[1], exact[2]]
    result = ngluon.compare(ours(near), theirs, 1e-9, 10)
    assert result["passed"] and result["rows"][0]["carrier"] == "double"
    assert result["worst"] == pytest.approx(2.5e-8)
    # Beyond it (and without the estimate term): fails.
    assert not ngluon.compare(ours([exact[0] + 2e-7, *exact[1:]]), theirs, 1e-9, 10)[
        "passed"
    ]
    assert not ngluon.compare(ours(near), theirs, 1e-9, 0)["passed"]
    # A double-double record is compared with the dd row (no estimate term).
    dd = [(4.0 + 1e-9) / born, exact[1], exact[2]]
    result = ngluon.compare(ours(dd, "dd"), theirs, 1e-12, 10)
    assert result["passed"] and result["rows"][0]["carrier"] == "double-double"
    assert not ngluon.compare(ours(exact, "dd"), theirs, 1e-12, 10)["passed"]
    # No common point: nothing compared, not passed.
    assert not ngluon.compare({"rows": [{"index": 3}]}, theirs, 1e-9, 10)["passed"]
    # An NGluon point without its pyAmpliCol row fails the comparison.
    extra = ours(near)
    extra["rows"].append({"index": 9})
    result = ngluon.compare(extra, theirs, 1e-9, 10)
    assert not result["passed"] and result["unmatched"] == [9]
    # A double-double point is compared only with a dd row.
    assert ngluon.compare(ours(dd, "dd"), _checks(born, virtual, estimate), 1e-12, 10)[
        "unmatched"
    ] == [0]


def test_a_failed_job_leaves_no_child_running(ngluon, monkeypatch, tmp_path):
    # ``run --jobs 2`` with a stand-in for the NJet binary: job 0 fails as
    # soon as job 1 runs, job 1 would run for minutes; the driver fails and
    # job 1 is gone (killed and reaped), the points file removed.
    fake = tmp_path / "ngluon_lc"
    fake.write_text(
        "#!/bin/sh\n"
        'for a in "$@"; do case "$a" in first=*) first=${a#first=};; esac; done\n'
        f"cd {tmp_path}\n"
        # job 0 fails once job 1 runs (at most ~5 s)
        '[ "$first" = 0 ] && { for i in $(seq 100); do [ -e pid.1 ] && break;'
        " sleep 0.05; done; exit 3; }\n"
        "echo $$ > pid.$first.tmp && mv pid.$first.tmp pid.$first\n"
        "exec sleep 300\n"
    )
    fake.chmod(0o755)
    monkeypatch.setattr(ngluon, "build", lambda njet, scratch: fake)
    scratch = tmp_path / "scratch"
    argv = ["run", "gg_gg", "--jobs", "2", "--njet-build", str(tmp_path)]
    argv += ["--scratch", str(scratch), "--json", str(tmp_path / "out.json")]
    with pytest.raises(SystemExit, match="ngluon_lc failed"):
        ngluon.main(argv)
    with pytest.raises(ProcessLookupError):
        os.kill(int((tmp_path / "pid.1").read_text()), 0)
    assert not list(scratch.glob("*.points"))
    assert not (tmp_path / "out.json").exists()
