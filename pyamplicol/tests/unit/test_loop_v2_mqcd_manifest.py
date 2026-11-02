# SPDX-License-Identifier: 0BSD
"""The M-QCD campaign manifest and the §1.4 comparison helper
(``tools/loop_benchmark/mqcd.py``; m-qcd/PLAN.md §1.4, §3.2 MQ-VALIDATE,
§4.2).

The manifest holds one record per PLAN §4.2 row; these checks need no
generation: the row set and its tiers equal the plan's, the structural
features of §1.1 are covered, every active row's reference resolves and
agrees with the row's request, every pending row or extra names the package
it waits for, and the rule helpers decide synthetic cases as §1.4 says
(bands, listings, outliers, sector sets, the field-factor poles).
"""

from __future__ import annotations

import json
import math
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from tools.loop_benchmark import mqcd

pytestmark = pytest.mark.loop_fast

# PLAN §4.2, verbatim row ids per block (§4.3: 76 core rows), plus the VV +
# jet rows D28 and D29 (mq-ewjoin, slice-status DECISIONS 2026-10-29 ~11:00):
# 78 rows.
PLAN_ROWS = {
    "A": "01 02 03 04 05 06 07 08 09 12 13 14 15 18 19 20 21 22 23 24 25 26",
    "B": "01 02 03 05 06 07 08",
    "C": "01 02 03 04 05 06 07 08 09",
    "D": "01 02 05 06 07 08 09 10 11 12 13 15 16 17 18 19 20 21 23 24 25 26 27 28 29",
    "E": "01 02 05 06",
    "F": "01 02 03 04",
    "G": "01 02",
    "H": "01 02 03",
    "M": "01",
    "R": "01",
}
# PLAN §4.3 "Tiers": the default tier's ten rows.
DEFAULT_TIER = {"A01", "A04", "A09", "A12", "B01", "C01", "D01", "D05", "F03", "G01"}
# PLAN §1.1: the classes and the structural features the rows must cover
# (initial states, multiplicities 2->1 ... 2->5, flavour structures, masses,
# Yukawas, the Q-EWBorn list, tt+X, the single-top channels, mixed orders,
# CKM, multiparticle labels, the 4FS twins, the special points of §4.1).
CLASSES = (
    "Q-massless",
    "Q-top",
    "Q-heavy",
    "Q-EWBorn",
    "Q-ttX",
    "Q-singletop",
    "Q-mixed",
    "Q-CKM",
    "Q-multi",
)
FEATURES = {
    "initial:qqbar", "initial:qg", "initial:gg", "initial:qq", "initial:qbarqbar",
    "initial:photon-hadron", "initial:lepton-hadron", "initial:leptons",
    "initial:multiparticle",
    "2->1", "2->2", "2->3", "2->4", "2->5",
    "gluons-only", "quark-line:1", "quarks:identical-flavour",
    "quarks:distinct-flavour", "quarks:three-lines",
    "massive:top", "four-tops", "massive:b", "massive:c", "massive:leptons",
    "yukawa:b", "yukawa:top", "scheme:4FS", "veto:top",
    "ew:dilepton", "ew:dilepton+jets", "ew:W-leptons", "ew:onshell-V+jet", "ew:VV",
    "ew:VV+jet",
    "ew:radiation-zero", "ew:VH", "ew:VBF-H", "ew:lepton-initiated-jets",
    "ew:W-exchange", "widths:colourless",
    "ttX", "singletop:s", "singletop:t", "singletop:tW", "singletop:4FS-t",
    "mixed-born-orders", "ckm:non-diagonal", "multiparticle",
    "special:threshold", "special:soft-collinear", "special:radiation-zero",
    "special:parameters",
}  # fmt: skip
# PLAN §1.1: every restriction loop_sm ships except test and parallel_test.
RESTRICTIONS = {
    "default",
    "no_b_mass",
    "no_widths",
    "no_masses",
    "no_tau_mass",
    "lepton_masses",
    "c_mass",
    "ckm",
    "zeromass_ckm",
}


@pytest.fixture(scope="module")
def manifest():
    return mqcd.load_manifest()


@pytest.fixture(scope="module")
def rows(manifest):
    return mqcd.rows(manifest)


def test_rows_are_the_plan_rows(rows):
    expected = {
        f"{block}{number}" for block, ids in PLAN_ROWS.items() for number in ids.split()
    }
    assert len(expected) == 78
    core = {name: row for name, row in rows.items() if row.base is None}
    assert set(core) == expected
    assert {name for name, row in core.items() if row.tier == "default"} == DEFAULT_TIER
    # Variants (a second reference of a row) carry their base row's id.
    assert all(name.split("@")[0] == row.base for name, row in rows.items() if row.base)


def test_rows_are_well_formed(manifest, rows):
    codes = manifest["restriction_codes"]
    manifest_rows = {data["id"]: data for data in manifest["rows"]}
    for name, row in rows.items():
        assert row.cls in CLASSES, name
        assert row.tier in mqcd.TIERS and row.status in mqcd.STATUSES, name
        assert row.corrections == ("QCD",), name
        assert set(row.extras) <= set(mqcd.EXTRAS), name
        # An extra the harness does not check yet names its package, or
        # (rule 7's nc) lists the other package's test ids, never both.
        for extra in set(row.extras) - mqcd.IMPLEMENTED_EXTRAS:
            delegated = extra in mqcd.DELEGATED_EXTRAS and bool(row.nc_column)
            assert (extra in row.extras_pending) != delegated, (name, extra)
        assert not row.nc_column or "nc" in row.extras, name
        assert set(row.extras_pending) <= set(row.extras), name
        # The special points (sp, PLAN §4.1 S1-S4) run as the row's gating
        # variants.
        if "sp" in row.extras and "sp" not in row.extras_pending:
            assert any(
                other.base == name and other.gating for other in rows.values()
            ), name
        if row.restriction_code in codes:
            code = codes[row.restriction_code]
            assert row.restriction == code["restriction"], name
            assert row.scheme == code["scheme"], name
        else:
            assert row.restriction_code == "G-full" and name == "G01", name
        assert row.points, name
        assert set(row.points) <= set(mqcd.TIERS), name
        if row.tier == "default":
            assert row.points["default"] == (0, 1), name
        else:
            assert "default" not in row.points, name
        if row.status == "pending":
            assert row.pending["waits_for"] in mqcd.UNBLOCKERS, name
            assert row.pending["reason"], name
        elif row.status == "composite":
            # Made of its gating variants, checks and delegated columns.
            assert row.pending is None and row.reference is None, name
            assert row.delegated, name
            assert any(o.base == name and o.gating for o in rows.values()), name
        else:
            assert row.pending is None and row.reference is not None, name
            assert row.reference["format"] in mqcd.REFERENCE_FORMATS, name
        assert row.inputs in ("python", "native"), name
        assert row.route in mqcd.ROUTES, name
        assert row.route == "tools" or row.subprocess, name
        # A checked per-helicity extra has MadLoop's per-helicity rows (the
        # row's reference or MQ-REF's per-helicity run of the row).
        if row.status == "active" and "hel" in set(row.extras) - set(
            row.extras_pending
        ):
            found = mqcd.helicity_reference(row)
            assert found["rows"] and found["rows"][0], name
        # Rule 4's prediction: the Python-input executor's; on the native-
        # input route (no executor prediction, WP3.1 I5c) the tree engine's
        # committed correlations, or none without them (status "partial");
        # none on the public route (its tools-route twin holds rule 4).
        if row.route == "public":
            assert row.catani == "none", name
        elif row.inputs == "native":
            assert row.catani in ("tree", "none"), name
            assert (row.catani == "tree") == (row.tree_correlations is not None)
        else:
            assert row.catani == "python", name
        if row.tree_correlations is not None:
            data = json.loads((mqcd.ROOT / row.tree_correlations).read_text())
            assert data["row"] == (row.base or name) and data["process"] == row.process
            indices = {e["index"] for e in data["points"]}
            assert {i for p in row.points.values() for i in p} <= indices, name
        if row.base is not None:
            base = rows[row.base]
            # A variant is another reference on the base request, or the
            # base request and reference on another restriction (its own
            # generation: the restricted-UFO variants of the 5F* rows).
            assert row.tier == "nightly", name
            assert (
                row.reference != base.reference or row.restriction != base.restriction
            ), name
            # A subprocess variant of a multiparticle row is one of the
            # processes MadLoop lists for it; any other variant keeps the
            # base request.
            if row.subprocess:
                listed = {
                    process
                    for processes in manifest_rows[row.base][
                        "madloop_subprocesses"
                    ].values()
                    for process in processes
                }
                assert row.process in listed, name
            else:
                assert row.process == base.process, name
        if isinstance(row.loop_orders, dict):
            # One loop order: the Born's plus the correction's two powers.
            assert row.loop_orders == {
                **row.born_orders,
                "QCD": row.born_orders["QCD"] + 2,
            }, name


def test_features_cover_the_plan(manifest, rows):
    rows = {name: row for name, row in rows.items() if row.base is None}
    assert set(manifest["features"]) == FEATURES
    covered = {feature for row in rows.values() for feature in row.features}
    assert covered == FEATURES
    assert {row.cls for row in rows.values()} == set(CLASSES)
    restrictions = {row.restriction for row in rows.values()}
    smokes = set(manifest["smokes"]["same_request"])
    assert restrictions | smokes >= RESTRICTIONS
    for restriction, spec in manifest["smokes"]["same_request"].items():
        assert spec["row"] in rows, restriction
        source = rows[spec.get("points_from", spec["row"])]
        assert source.active and source.process == rows[spec["row"]].process


def test_active_references_agree_with_their_rows(manifest, rows):
    # MadLoop's process syntax accepts MG5 aliases loop_sm does not name
    # (``mu-`` for loop_sm's ``m-``); the rows use the model's own names.
    aliases = manifest["process_aliases"]
    for name, row in rows.items():
        if not row.active:
            continue
        reference = mqcd.load_reference(row)
        indices = [i for points in row.points.values() for i in points]
        assert max(indices) < len(reference.points), name
        assert len(reference.scales) == len(reference.points), name
        assert reference.providers == tuple(row.reference["providers"]), name
        if reference.fixed_parameters:
            ignored = set(row.ignored_fixed_parameters)
            assert ignored <= set(reference.fixed_parameters), name
            fixed = {
                k: v for k, v in reference.fixed_parameters.items() if k not in ignored
            }
            assert fixed == dict(row.fixed_parameters), name
        if row.reference["format"] == "required":
            data = json.loads((mqcd.ROOT / row.reference["path"]).read_text())
            process = data["process"].split(" [")[0]
            tokens = [aliases.get(t, t) for t in process.split() if "=" not in t]
            expected = rows[row.base].process if row.subprocess else row.process
            assert " ".join(tokens) == expected, name
            # A mixed-order request lists every Born order (``born_orders_all``,
            # MQ-REF); the row holds their bounds and every loop order.
            borns = data.get("born_orders_all", [data["born_orders"]])
            bounds = {k: max(o.get(k, 0) for o in borns) for k in row.born_orders}
            assert bounds == dict(row.born_orders), name
            if len(borns) == 1:
                assert data["loop_orders"] == dict(row.loop_orders), name
        for point in reference.points:
            assert len(point) == len(row.process.replace(">", "").split()), name


def test_listings_name_real_points_and_coefficients(rows):
    for name, row in rows.items():
        for entry in row.reference_outliers:
            assert entry["point"] in {i for p in row.points.values() for i in p}, name
            assert entry["coefficient"] in mqcd.COEFFICIENTS, name
        for entry in row.vanishing:
            # A Laurent coefficient, or a colour-forbidden Born sector.
            assert entry["coefficient"] in mqcd.COEFFICIENTS, name


def test_pending_rows_and_extras_name_their_package(rows):
    pending = {
        name: row.pending["waits_for"]
        for name, row in rows.items()
        if row.status == "pending"
    }
    waiting = {
        (name, extra): reason
        for name, row in rows.items()
        for extra, reason in row.extras_pending.items()
    }
    assert all(package in mqcd.UNBLOCKERS for package in pending.values())
    assert all(reason.split()[0] in mqcd.UNBLOCKERS for reason in waiting.values())


def test_rule7_is_mq_nc_s_column_or_waits_for_it(manifest, rows):
    """Rule 7 (the nc extra): every nc row waits for MQ-NC, or every nc row
    lists MQ-NC's test ids, equal to its live ``NC_COLUMN`` (the projection
    id first), in test files that exist; the LC/NLC smoke on every shipped
    restriction likewise waits or names MQ-NC's test."""

    nc = {name: row for name, row in rows.items() if "nc" in row.extras}
    smoke = manifest["smokes"]["lc_every_restriction"]
    assert len(nc) == 13 and all(row.base is None for row in nc.values())
    others = [
        node
        for row in rows.values()
        for name, ids in row.delegated.items()
        if name != "nc"
        for node in ids
    ]
    if not any(row.nc_column for row in nc.values()):
        assert all(r.extras_pending["nc"].startswith("MQ-NC") for r in nc.values())
        assert smoke == "MQ-NC"
        assert mqcd.delegated_ids(manifest) == others
        return
    import importlib

    module = importlib.import_module("tests.unit.test_loop_v2_nc_projection")
    live = {name: tuple(ids) for name, ids in module.NC_COLUMN.items()}
    assert live == {name: row.nc_column for name, row in nc.items()}
    for name, row in nc.items():
        assert "test_loop_v2_nc_projection.py::" in row.nc_column[0], name
        assert row.nc_column[0].endswith(f"[{name}]"), name
    for node in {node for row in nc.values() for node in row.nc_column}:
        assert (mqcd.ROOT / node.split("::")[0]).is_file(), node
    assert (mqcd.ROOT / smoke["test"].split("::")[0]).is_file()
    # The sweep's list: each id once, the smoke last.
    ids = mqcd.delegated_ids(manifest)
    assert len(ids) == len(set(ids)) and ids[-1] == smoke["test"]
    assert set(ids[:-1]) == {
        node for row in nc.values() for node in row.nc_column
    } | set(others)


# ------------------------------------------------------------ the helpers

SECTOR = (("QCD", 8),)
BORN_SECTOR = (("QCD", 6),)


def _row(**changes):
    data = {
        "id": "X01",
        "class": "Q-massless",
        "process": "u u~ > d d~ g",
        "born_orders": {"QCD": 3, "QED": 0},
        "loop_orders": {"QCD": 5, "QED": 0},
        "restriction": "no_b_mass",
        "restriction_code": "5F",
        "tier": "default",
        "status": "active",
        "reference": None,
        "points": {"default": [0, 1]},
        "predicted_sectors": [{"QCD": 8}],
    }
    data.update(changes)
    return mqcd.Row.from_dict(data)


def _point(carrier, born, laurent, estimate=None, catani=None):
    return mqcd.Evaluated(
        carrier,
        {BORN_SECTOR: Decimal(str(born))},
        {SECTOR: tuple(Decimal(str(x)) for x in laurent)},
        catani,
        None if estimate is None else {SECTOR: tuple(estimate)},
    )


def _reference(providers):
    """A one-point reference: provider -> (born, finite, single, double),
    accuracy -> per provider."""

    names = tuple(providers)
    return mqcd.Reference(
        "synthetic",
        names,
        [[["1", "0", "0", "1"]]],
        ["1"],
        {},
        None,
        [{p: {mqcd.TOTAL: v[0]} for p, v in providers.items()}],
        [{p: {mqcd.TOTAL: tuple(v[1:4])} for p, v in providers.items()}],
        [{p: v[4] for p, v in providers.items() if len(v) > 4}],
        0.118,
    )


def test_relative_is_strict():
    assert mqcd.relative(1, 1) == 0.0
    assert mqcd.relative(0, 0) == 0.0
    assert mqcd.relative("1", "1.000000000001") == pytest.approx(1e-12, rel=1e-6)
    assert mqcd.relative(0, 1e-300) == 1.0  # no floor
    assert mqcd.o1_bound(1e-13) == 1e-12
    assert mqcd.o1_bound(1e-12) == 1e-12
    assert mqcd.o1_bound(3e-11) == 3e-11
    assert mqcd.o1_bound(1e-9) == 1e-9
    assert mqcd.o1_bound(1.0001e-9) is None


def test_rule1_holds_double_double_to_the_primary_provider():
    dd = _point("double-double", 2.0, (1.0, -3.0, 0.5))
    exact = {"madloop": (2.0, 1.0, -3.0, 0.5, 1e-13), "recola": (2.0, 1.0, -3.0, 0.5)}
    checks = mqcd.rule1(_row(), _reference(exact), 0, dd)
    assert len(checks) == 4 and all(c.passed for c in checks)
    off = {"madloop": (2.0, 1.0 + 3e-12, -3.0, 0.5, 1e-13), "recola": exact["recola"]}
    (failed,) = [c for c in mqcd.rule1(_row(), _reference(off), 0, dd) if not c.passed]
    assert failed.coefficient == "finite" and "not a listed outlier" in failed.note


def test_rule1_outliers_need_their_accuracy_and_a_second_provider():
    dd = _point("double-double", 2.0, (1.0, -3.0, 0.5))
    listed = _row(
        reference_outliers=[
            {"point": 0, "sector": "total", "coefficient": "finite", "provider": "x"}
        ]
    )

    def finite(providers):
        (check,) = [
            c
            for c in mqcd.rule1(listed, _reference(providers), 0, dd)
            if c.coefficient == "finite"
        ]
        return check

    covered = (2.0, 1.0 + 3e-12, -3.0, 0.5, 1e-11)
    assert finite({"madloop": covered, "recola": (2.0, 1.0, -3.0, 0.5)}).passed
    # The provider's accuracy does not cover the difference.
    short = (2.0, 1.0 + 3e-12, -3.0, 0.5, 1e-12)
    assert not finite({"madloop": short, "recola": (2.0, 1.0, -3.0, 0.5)}).passed
    # No second provider agrees within 1e-12.
    other = (2.0, 1.0 + 2e-12, -3.0, 0.5)
    assert not finite({"madloop": covered, "recola": other}).passed
    # A listed outlier that agrees is a stale listing.
    agreeing = {"madloop": (2.0, 1.0, -3.0, 0.5, 1e-11), "recola": other}
    stale = finite(agreeing)
    assert not stale.passed and stale.note == "stale outlier listing"


def test_rule1_holds_each_row_to_its_stated_tolerance():
    """Rule 1 at the row's ``reference_tolerance`` (user ruling 2026-10-29
    ~01:12: up to 1e-10 from 2->4 up where needed); the absolute checks stay
    at 1e-12 F_s."""

    dd = _point("double-double", 2.0, (1.0, -3.0, 0.5))
    high = _row(process="u u~ > d d~ s s~", reference_tolerance=1e-10)
    off = {"madloop": (2.0, 1.0 + 5e-11, -3.0, 0.5, 0.0)}
    checks = mqcd.rule1(high, _reference(off), 0, dd)
    assert all(c.passed for c in checks)
    (finite,) = [c for c in checks if c.coefficient == "finite"]
    assert finite.bound == 1e-10 and finite.error == pytest.approx(5e-11, rel=1e-6)
    (failed,) = [
        c
        for c in mqcd.rule1(
            high, _reference({"madloop": (2.0, 1.0 + 2e-10, -3.0, 0.5)}), 0, dd
        )
        if not c.passed
    ]
    assert failed.coefficient == "finite" and failed.bound == 1e-10
    assert not all(c.passed for c in mqcd.rule1(_row(), _reference(off), 0, dd))
    # The absolute check of a vanishing coefficient is not widened.
    flux = 2.0 * 0.118 / (2 * math.pi)
    vanishing = _row(
        process="u u~ > d d~ s s~",
        reference_tolerance=1e-10,
        vanishing=[{"sector": "total", "coefficient": "double_pole"}],
    )
    near = _point("double-double", 2.0, (1.0, -3.0, 5e-12 * flux))
    (double,) = [
        c
        for c in mqcd.rule1(
            vanishing, _reference({"madloop": (2.0, 1.0, -3.0, 0.0)}), 0, near
        )
        if c.coefficient == "double_pole"
    ]
    assert not double.passed and double.bound == pytest.approx(1e-12 * flux)


def test_rule1_tolerance_is_relaxed_only_where_the_providers_need_it(rows):
    """1e-12, relaxed to 1e-10 only on the 2->4+ rows whose providers' own
    precision needs it (A18, A23; the user's ruling ~01:12 "if need be"),
    or, under the user's ruling ~10:20, to at most ATTENTION on a row whose
    provider misses by less (D28: MadLoop's finite part at point 1); never
    above the ceiling; a variant keeps its base row's."""

    relaxed, attention = set(), set()
    for name, row in rows.items():
        final = len(row.process.split(">")[1].split())
        assert row.reference_tolerance <= mqcd.REFERENCE_TOLERANCE_CEILING
        if (row.base or name) in {"D28"}:
            assert row.reference_tolerance <= mqcd.ATTENTION, name
            attention.add(row.base or name)
        elif row.reference_tolerance != mqcd.REFERENCE_TOLERANCE:
            assert final >= 4, name
            assert row.reference_tolerance == mqcd.REFERENCE_TOLERANCE_HIGH, name
            relaxed.add(row.base or name)
        if row.base and not row.subprocess:
            assert row.reference_tolerance == rows[row.base].reference_tolerance
    assert relaxed == {"A18", "A23"}
    assert attention == {"D28"}


def test_rule1_checks_a_vanishing_coefficient_absolutely():
    vanishing = _row(
        vanishing=[{"sector": "total", "coefficient": "double_pole"}],
        unpredicted_sectors=[],
    )
    flux = 2.0 * 0.118 / (2 * math.pi)
    dd = _point("double-double", 2.0, (1.0, -3.0, 0.5e-12 * flux))
    reference = _reference({"madloop": (2.0, 1.0, -3.0, 0.0)})
    (double,) = [
        c
        for c in mqcd.rule1(vanishing, reference, 0, dd)
        if c.coefficient == "double_pole"
    ]
    assert double.passed and double.bound == pytest.approx(1e-12 * flux)
    (strict,) = [
        c
        for c in mqcd.rule1(_row(), reference, 0, dd)
        if c.coefficient == "double_pole"
    ]
    assert not strict.passed


def test_rule2_bands_and_rule3_bracketing():
    dd = _point("double-double", 2.0, (1.0, -3.0, 0.5))

    def checks(row, error, estimate):
        b64 = _point("double", 2.0, (1.0 + error, -3.0, 0.5), (estimate, 0.0, 0.0))
        found = mqcd.rule2_and_3(row, _reference({"m": (2, 1, -3, 0.5)}), 0, b64, dd)
        return {(c.rule, c.coefficient): c for c in found if c.coefficient == "finite"}

    # E <= 1e-12: held to 1e-12 (and 10 E brackets the error).
    found = checks(_row(), 5e-13, 1e-13)
    assert found["2", "finite"].passed and found["3", "finite"].passed
    found = checks(_row(), 2e-12, 1e-12)
    assert not found["2", "finite"].passed and found["3", "finite"].passed
    # 1e-12 < E <= 1e-9: held to E.
    found = checks(_row(), 5e-11, 6e-11)
    assert found["2", "finite"].passed
    assert found["2", "finite"].bound == pytest.approx(6e-11, rel=1e-9)
    # The estimate under-reports tenfold: rule 3 fails.
    found = checks(_row(), 5e-11, 4e-12)
    assert not found["2", "finite"].passed and not found["3", "finite"].passed
    # E > 1e-9: outside the scope, reported (E, true error), not asserted;
    # rule 3 still holds it.
    found = checks(_row(), 1e-8, 2e-9)
    assert found["2", "finite"].passed and not found["2", "finite"].asserted
    assert found["2", "finite"].bound == pytest.approx(2e-9, rel=1e-6)
    assert found["3", "finite"].passed
    found = checks(_row(), 1e-7, 2e-9)
    assert not found["3", "finite"].passed


def test_relaxed_tolerances_apply_to_their_rule_only():
    """The user's ruling of 2026-10-29 ~10:20: a check failing below ~1e-5
    gets its row's tolerance relaxed (``relaxed``: rule -> tolerance); the
    bands stay where they are tighter than the relaxed value only above it,
    and nothing else changes."""

    dd = _point("double-double", 2.0, (1.0, -3.0, 0.5))
    b64 = _point("double", 2.0, (1.0 + 2e-10, -3.0, 0.5), (3e-11, 0.0, 0.0))
    reference = _reference({"m": (2, 1, -3, 0.5)})

    def rule2(row):
        (check,) = [
            c
            for c in mqcd.rule2_and_3(row, reference, 0, b64, dd)
            if c.rule == "2" and c.coefficient == "finite"
        ]
        return check

    assert not rule2(_row()).passed  # 2e-10 > E = 3e-11
    relaxed = rule2(_row(relaxed={"2": 1e-8}))
    assert relaxed.passed and relaxed.bound == 1e-8
    assert rule2(_row(relaxed={"6": 1e-8})).bound == pytest.approx(3e-11)


def test_relaxed_tolerances_stay_below_the_attention_threshold(rows):
    """Every relaxed tolerance of the manifest is at most ~1e-5 (the user's
    ruling) and names a rule the helper relaxes."""

    for name, row in rows.items():
        for rule, tolerance in row.relaxed.items():
            assert rule in ("2", "6"), name
            assert 0 < tolerance <= mqcd.ATTENTION, name


def test_rule4_holds_the_poles_and_the_sector_sets():
    catani = {SECTOR: (Decimal("-3.0"), Decimal("0.5"))}
    dd = _point("double-double", 2.0, (1.0, -3.0, 0.5), (1e-28, 1e-28, 1e-28), catani)
    assert all(c.passed for c in mqcd.rule4(_row(), 0, dd))
    off = {SECTOR: (Decimal("-3.0000000000000000000001"), Decimal("0.5"))}
    dd_off = _point("double-double", 2.0, (1.0, -3.0, 0.5), (1e-28,) * 3, off)
    (failed,) = [c for c in mqcd.rule4(_row(), 0, dd_off) if not c.passed]
    assert failed.coefficient == "single_pole" and failed.bound == 1e-24
    # A predicted sector the manifest does not expect fails, and so does an
    # unexpected sector without a prediction.
    other = _row(predicted_sectors=[{"QCD": 6, "QED": 2}])
    names = {c.coefficient for c in mqcd.rule4(other, 0, dd) if not c.passed}
    assert names == {"predicted sectors"}
    extra = _row(unpredicted_sectors=[{"QCD": 6, "QED": 2}])
    names = {c.coefficient for c in mqcd.rule4(extra, 0, dd) if not c.passed}
    assert names == {"unpredicted sectors"}
    none = _point("double-double", 2.0, (1.0, -3.0, 0.5), (1e-28,) * 3, None)
    unpredicted = _row(predicted_sectors=[], unpredicted_sectors=[{"QCD": 8}])
    assert all(c.passed for c in mqcd.rule4(unpredicted, 0, none))


def test_rule4_relaxes_only_its_carrier():
    """A row's relaxed rule 4 is per carrier (the user's ruling of ~10:20:
    passing checks keep their tolerances): ``"4"`` relaxes the binary64
    poles only, ``"4dd"`` the double-double ones only."""

    off = {SECTOR: (Decimal("-3.000000000000003"), Decimal("0.5"))}  # 1e-15
    dd = _point("double-double", 2.0, (1.0, -3.0, 0.5), (1e-28,) * 3, off)
    wide = {SECTOR: (Decimal("-3.0000000003"), Decimal("0.5"))}  # 1e-10
    b64 = _point("double", 2.0, (1.0, -3.0, 0.5), (1e-14,) * 3, wide)

    def single(row, point):
        (check,) = [
            c for c in mqcd.rule4(row, 0, point) if c.coefficient == "single_pole"
        ]
        return check

    assert not single(_row(), dd).passed
    assert not single(_row(relaxed={"4": 1e-14}), dd).passed
    assert single(_row(relaxed={"4dd": 1e-14}), dd).passed
    assert not single(_row(), b64).passed
    assert not single(_row(relaxed={"4dd": 1e-8}), b64).passed
    assert single(_row(relaxed={"4": 1e-8}), b64).passed


def test_rule4_reports_binary64_poles_outside_the_scope():
    """Rule 4 in binary64: the bands of rule 2; a pole outside the O1 scope
    is reported, not asserted."""

    catani = {SECTOR: (Decimal("-3.0"), Decimal("0.5"))}
    b64 = _point("double", 2.0, (1.0, -3.0, 0.5), (1e-13,) * 3, catani)
    assert all(c.passed and c.asserted for c in mqcd.rule4(_row(), 0, b64))
    off = {SECTOR: (Decimal("-3.0000001"), Decimal("0.5"))}
    wide = _point("double", 2.0, (1.0, -3.0, 0.5), (1e-13, 3e-8, 1e-13), off)
    (single,) = [
        c for c in mqcd.rule4(_row(), 0, wide) if c.coefficient == "single_pole"
    ]
    assert single.passed and not single.asserted


def test_massless_catani_operator():
    """Rule 4's "tree" operator: q q~ -> colour singlet, <T_1.T_2> = -C_F B:
    poles (alpha_s / 2 pi) C_F B (-2 / eps^2 - (3 + 2 log(mu^2 / s)) / eps)."""

    from fractions import Fraction

    born, s, mu2 = Decimal("1.5"), Fraction(4), Fraction(1)
    point = [[Fraction(1), 0, 0, Fraction(1)], [Fraction(1), 0, 0, Fraction(-1)]]
    legs = {1: (3, Fraction(2)), 2: (3, Fraction(2))}  # gamma_q = 3/2 C_F
    correlated = {(1, 2): -Decimal(4) / 3 * born, (2, 1): -Decimal(4) / 3 * born}
    single, double = mqcd.massless_catani(
        legs, point, correlated, mu2, Decimal("0.118")
    )
    with localcontext() as context:
        context.prec = 50
        factor = Decimal("0.118") / (2 * mqcd.PI) * Decimal(4) / 3 * born
        log = (Decimal(mu2.numerator) / Decimal(s.numerator)).ln()
        expected = (-(3 + 2 * log) * factor, -2 * factor)
    assert mqcd.relative(double, expected[1]) < 1e-38
    assert mqcd.relative(single, expected[0]) < 1e-38


def test_rule6_checks_a_vanishing_coefficient_absolutely():
    """Rule 6 holds an identically vanishing coefficient absolutely (against
    double-double and 60 digits, |a - r| against F_s), not skipped."""

    row = _row(vanishing=[{"sector": "QCD^8", "coefficient": "double_pole"}])
    dd = _point("double-double", 2.0, (1.0, -3.0, 0.0), (1e-28,) * 3)
    ap = _point("arbitrary-40", 2.0, (1.0, -3.0, 0.0))
    near = _point("arbitrary-60", 2.0, (1.0, -3.0, 1e-30))
    checks = mqcd.rule6(row, 0, dd, ap, ap)
    doubles = [c for c in checks if c.coefficient == "double_pole"]
    assert len(doubles) == 2 and all(c.passed for c in doubles)
    checks = mqcd.rule6(row, 0, dd, ap, near)
    (failed,) = [c for c in checks if not c.passed]
    assert failed.coefficient == "double_pole" and "vanishes" in failed.note


def test_field_factor_poles():
    from fractions import Fraction

    # Units of alpha_s / 4 pi: Z_3 = 5/3 C_A - 4/3 T_F n_f, Z_2 = -C_F.
    gluon = Fraction(5, 3) * 3 - Fraction(4, 3) * Fraction(1, 2) * 6
    four = mqcd.field_poles([(8, False)] * 4, 6)
    assert isinstance(four, Fraction) and four == 4 * gluon == 4
    assert mqcd.field_poles([(8, False)], 5) == Fraction(5) - Fraction(10, 3)
    assert mqcd.field_poles([(3, False), (3, True), (1, False)], 5) == Fraction(-8, 3)
    with pytest.raises(ValueError):
        mqcd.field_poles([(6, False)], 5)


def test_sector_keys_drop_zero_powers():
    assert mqcd.sector_key({"QCD": 8, "QED": 0}) == (("QCD", 8),)
    assert mqcd.sector_key([["QED", 2], ["QCD", 4]]) == (("QCD", 4), ("QED", 2))
    assert mqcd.sector_text((("QCD", 4), ("QED", 2))) == "QCD^4 QED^2"
    assert mqcd.born_sector((("QCD", 6), ("QED", 2))) == (("QCD", 4), ("QED", 2))


def test_per_helicity_rows_take_the_helicity_average():
    """MadLoop's SLOOPMATRIXHEL rows have no helicity average: ours (with
    the inclusive initial average) times the incoming legs' helicity count."""

    def ours(born, laurent):
        return _point("double-double", born, laurent)

    rows = {
        (-1, -1, 1): ours(0.25, (0.5, -0.25, 0.125)),
        (1, -1, 1): ours(0.0, (0.0, 0.0, 0.0)),
        (-1, 1, 1): ours(0.5, (1.0, -0.5, 0.25)),
        (1, 1, 1): ours(0.0, (0.0, 0.0, 0.0)),
    }
    madloop = [
        {"helicity": [-1, -1, 1], "born": 1.0, "finite": 2.0, "single_pole": -1.0,
         "double_pole": 0.5},
        {"helicity": [1, -1, 1], "born": 0.0, "finite": 0.0, "single_pole": 0.0,
         "double_pole": 0.0},
        {"helicity": [-1, 1, 1], "born": 2.0, "finite": 4.0, "single_pole": -2.0,
         "double_pole": 1.0},
        {"helicity": [1, 1, 1], "born": 1e-33, "finite": 0.0, "single_pole": 0.0,
         "double_pole": 0.0},
    ]  # fmt: skip
    checks = mqcd.rule_hel(_row(), 0, rows, madloop, 0.118)
    assert all(c.passed for c in checks), [c for c in checks if not c.passed]
    madloop[2]["finite"] = 4.0 * (1 + 2e-12)
    failed = [c for c in mqcd.rule_hel(_row(), 0, rows, madloop, 0.118) if not c.passed]
    assert [(c.sector, c.coefficient) for c in failed] == [("h-++", "finite")]
    madloop[2]["finite"] = 4.0
    # A vanishing helicity (our double-double Born an exact zero, MadLoop's
    # its rounding) is compared absolutely against the summed Born (its F
    # for the Laurent coefficients), also when MQ-REF's record says so.
    madloop[3]["finite"] = 1e-14  # 1.8e-13 of F
    madloop[3]["reference"] = {"comparison": "absolute", "reason": "vanishing"}
    checks = mqcd.rule_hel(_row(), 0, rows, madloop, 0.118)
    finite = next(c for c in checks if (c.sector, c.coefficient) == ("h+++", "finite"))
    assert finite.passed and finite.bound == pytest.approx(
        1e-12 * 3.0 * 0.118 / (2 * math.pi)
    )
    assert "vanishing helicity" in finite.note
    madloop[3]["finite"] = 1e-12
    checks = mqcd.rule_hel(_row(), 0, rows, madloop, 0.118)
    assert not next(
        c for c in checks if c.coefficient == "finite" and c.sector == "h+++"
    ).passed
    madloop[3]["finite"] = 0.0
    # A suppressed helicity stays strictly relative, also when MQ-REF's
    # record compares it absolutely (that would loosen rule 1): 1e-6
    # relative fails although it is 3.5e-15 of F.
    rows[(1, 1, 1)] = ours(0.25e-10, (0.5e-10, -0.25e-10, 0.125e-10))
    madloop[3].update(born=1e-10, finite=2e-10 * (1 + 1e-6), single_pole=-1e-10,
                      double_pole=0.5e-10)  # fmt: skip
    madloop[3]["reference"] = {
        "comparison": "absolute",
        "absolute_scale": 3.0,
        "reason": "suppressed helicity",
    }
    failed = [c for c in mqcd.rule_hel(_row(), 0, rows, madloop, 0.118) if not c.passed]
    assert [(c.sector, c.coefficient) for c in failed] == [("h+++", "finite")]
    assert failed[0].bound == 1e-12 and "not honoured" in failed[0].note
    assert "suppressed helicity" in failed[0].note
    madloop[3]["finite"] = 2e-10
    assert all(c.passed for c in mqcd.rule_hel(_row(), 0, rows, madloop, 0.118))
    # An entry MQ-REF records as no reference is reported, not compared.
    madloop[3]["finite"] = 1.0
    madloop[3]["reference"] = {"status": "none", "reason": "unstable helicity"}
    checks = mqcd.rule_hel(_row(), 0, rows, madloop, 0.118)
    (none,) = [c for c in checks if c.sector == "h+++"]
    assert none.passed and not none.asserted and "unstable" in none.note
    del madloop[3]["reference"]
    # A configuration MadLoop has and we do not (or the reverse) fails.
    failed = mqcd.rule_hel(_row(), 0, rows, madloop[:3], 0.118)
    assert not failed[0].passed and failed[0].coefficient == "helicity rows"


def test_nightly_jobs_cover_the_tier_within_the_budget(manifest):
    """PLAN §3.2 acceptance: each nightly file job <= 30 min. The jobs
    partition the nightly row ids (a row's variants in its job, which reuse
    its generation), each job has its module, and its measured cost is
    within the budget."""

    jobs = manifest["nightly_jobs"]["jobs"]
    assigned = [name for job in jobs for name in job["rows"]]
    assert len(assigned) == len(set(assigned))
    assert sorted(assigned) == mqcd.nightly_ids(manifest)
    owner = {name: job["job"] for job in jobs for name in job["rows"]}
    for name in assigned:
        base = name.split("@")[0]
        if "@" in name and base in owner:
            assert owner[name] == owner[base], name
    root = mqcd.ROOT
    for job in jobs:
        assert job["seconds"] <= 30 * 60, job
        text = (root / job["file"]).read_text()
        assert f"nightly_module({job['job']})" in text, job["file"]
    files = sorted(p.name for p in (root / "tests" / "unit").glob(
        "test_loop_v2_mqcd_nightly*.py"))  # fmt: skip
    assert files == sorted(Path(job["file"]).name for job in jobs)


def test_ckm_relations_name_a_twin_and_model_entries(rows):
    """PLAN §4.1: the H rows replace a second provider with the exact
    CKM-factor relation to a diagonal twin."""

    related = {name for name, row in rows.items() if row.ckm_relation}
    assert {"H01", "H02"} <= related
    for name in related:
        relation = rows[name].ckm_relation
        assert relation["twin_process"] != rows[name].process, name
        assert relation["numerator"] and relation["denominator"], name


def test_ckm_relation_rule():
    relation = {"twin_process": "u d~ > w+ g", "numerator": ["CKM1x2"]}
    row = _row(ckm_relation={**relation, "denominator": ["CKM1x1"]})
    factor = Decimal("0.0535")
    twin = _point("double-double", "2.5", ("1.5", "-0.75", "-0.25"))
    exact = _point(
        "double-double",
        str(factor * Decimal("2.5")),
        tuple(str(factor * Decimal(x)) for x in ("1.5", "-0.75", "-0.25")),
    )
    checks = mqcd.rule_ckm(row, 0, exact, twin, factor)
    assert len(checks) == 4 and all(c.passed for c in checks)
    off = _point(
        "double-double",
        str(factor * Decimal("2.5")),
        (str(factor * Decimal("1.5") * (1 + Decimal("1e-23"))), *(
            str(factor * Decimal(x)) for x in ("-0.75", "-0.25"))),
    )  # fmt: skip
    failed = [c for c in mqcd.rule_ckm(row, 0, off, twin, factor) if not c.passed]
    assert [(c.coefficient, c.rule) for c in failed] == [("finite", "ckm")]


def _nc_manifest(manifest, columns, path, waiting=()):
    """A copy of the manifest at ``path`` whose nc rows in ``columns`` list
    those rule 7 ids, whose other nc rows wait for MQ-NC, and whose rows or
    variants in ``waiting`` wait for MQ-5F (so the report tests do not
    depend on whether MQ-NC or MQ-5F has landed)."""

    data = json.loads(json.dumps(manifest))
    for entry in data["rows"]:
        units = [(entry["id"], entry)] + [
            (f"{entry['id']}@{v['name']}", v) for v in entry.get("variants", ())
        ]
        for name, unit in units:
            if name in waiting:
                unit["status"] = "pending"
                unit["pending"] = {"waits_for": "MQ-5F", "reason": "test"}
        if "nc" not in entry["extras"]:
            continue
        (entry.get("delegated") or {}).pop("nc", None)
        pending = entry.setdefault("extras_pending", {})
        pending.pop("nc", None)
        if entry["id"] in columns:
            entry.setdefault("delegated", {})["nc"] = list(columns[entry["id"]])
        else:
            pending["nc"] = "MQ-NC (rule 7)"
    path.write_text(json.dumps(data))
    return path


def _points(name: str, tier: str) -> list[int]:
    """A row's (or variant's) manifest points of ``tier``, as its record
    states them."""

    row = mqcd.rows().get(name)
    return [] if row is None else list(row.tier_points(tier))


def _report_builder():
    import importlib.util

    path = mqcd.ROOT / "docs/performance_reports/loop/scripts/build_mqcd.py"
    spec = importlib.util.spec_from_file_location("loop_report_build_mqcd", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_data_counts_variants_smokes_and_findings(
    tmp_path, monkeypatch, manifest
):
    """build_mqcd.py: a failing gating variant fails its row, a non-gating
    one does not, smoke and expansion records are kept apart, a row without
    a record is not-run and a pending row stays pending; the findings name
    their rows in the per-row table."""

    builder = _report_builder()
    waiting = ("C09", "D17@5F")
    path = _nc_manifest(manifest, {}, tmp_path / "manifest.json", waiting)
    monkeypatch.setattr(builder, "MANIFEST", path)
    tmp_path = tmp_path / "results"
    tmp_path.mkdir()

    def record(name, tier, passed):
        check = {"rule": "1", "point": 0, "carrier": "double-double",
                 "sector": "total", "coefficient": "finite", "error": 2e-12,
                 "bound": 1e-12, "note": "", "passed": passed}  # fmt: skip
        data = {"row": name, "tier": tier, "points": _points(name, tier),
                "passed": passed, "checks": 1, "failures": int(not passed),
                "check_records": [check]}  # fmt: skip
        (tmp_path / f"{name}.{tier}.json").write_text(json.dumps(data))

    record("A01", "default", True)
    record("A01@sectors", "nightly", True)
    # Without its nightly record (point 2, extras) A01 is not run.
    data = builder.build([tmp_path], {}, None)
    a01 = next(row for row in data["rows"] if row["id"] == "A01")
    assert (a01["status"], a01["missing_records"]) == ("not-run", ["nightly"])
    record("A01", "nightly", True)
    record("A06", "nightly", True)  # its gating variant A06@sectors not run
    record("A09", "default", True)
    record("A09@soft", "nightly", False)
    record("C04", "nightly", True)
    record("C04@z", "nightly", False)
    record("smoke:no_masses", "nightly", True)
    findings = {"findings": [{"id": "F9", "rows": ["A09@soft"]}]}
    data = builder.build([tmp_path], {}, findings)
    rows = {row["id"]: row for row in data["rows"]}
    assert rows["A01"]["status"] == "pass"
    # A partial rerun (other points) in a later directory is no tier record.
    rerun = tmp_path.parent / "rerun"
    rerun.mkdir()
    partial = {"row": "A01", "tier": "nightly", "points": [0], "passed": True,
               "checks": 1, "failures": 0, "check_records": []}  # fmt: skip
    (rerun / "A01.nightly.json").write_text(json.dumps(partial))
    (tmp_path / "A01.nightly.json").unlink()
    later = {r["id"]: r for r in builder.build([tmp_path, rerun], {}, None)["rows"]}
    assert later["A01"]["status"] == "not-run"
    record("A01", "nightly", True)
    assert rows["A09"]["status"] == "fail"
    assert rows["C04"]["status"] == "pass"
    assert rows["A02"]["status"] == "not-run"
    assert rows["A06"]["status"] == "not-run"
    assert rows["C09"]["status"] == "pending"
    assert data["smokes_and_expansions"]["smoke:no_masses"]["passed"]
    assert len(data["rows"]) == 78
    assert sum(data["counts"].values()) == len(data["rows"])
    table = builder.markdown(data)
    line = next(x for x in table.splitlines() if x.startswith("| A09"))
    assert "fail" in line and "F9" in line
    assert "soft: r1 p0 DD total finite 2e-12>1e-12" in line
    # Rule 1's tolerance is stated per row (1e-10 only where needed).
    assert rows["A01"]["reference_tolerance"] == 1e-12
    assert rows["A19"]["reference_tolerance"] == 1e-12
    assert rows["A18"]["reference_tolerance"] == 1e-10
    line = next(x for x in table.splitlines() if x.startswith("| A18"))
    assert "| 1e-10 |" in line
    # A pending variant is reported and leaves its row's status alone.
    record("D17", "nightly", True)
    data = builder.build([tmp_path], {}, findings)
    d17 = next(row for row in data["rows"] if row["id"] == "D17")
    assert d17["status"] == "pass"
    assert {v["id"]: v["status"] for v in d17["variants"]}["D17@5F"] == "pending"
    line = next(x for x in builder.markdown(data).splitlines() if x.startswith("| D17"))
    assert "+5F (waits for MQ-5F)" in line


def test_report_data_reads_the_nc_column_from_junit(tmp_path, monkeypatch, manifest):
    """build_mqcd.py --junit: a row's nc column (rule 7, MQ-NC's test ids)
    passes when every id passed (an unparametrised id stands for all its
    parametrisations), fails its row when one failed, and leaves its row
    not-run when absent; the LC smoke on every restriction likewise."""

    builder = _report_builder()
    projection = (
        "tests/unit/test_loop_v2_nc_projection.py::"
        "test_lc_and_nlc_are_the_kept_powers_of_the_nc_scan"
    )
    oracle = "tests/unit/test_loop_v2_nc_oracles.py::test_oracle"
    smoke = (
        "tests/unit/test_loop_v2_nc_declaration.py::"
        "test_lc_and_nlc_smoke_on_every_shipped_restriction"
    )
    columns = {name: (f"{projection}[{name}]", oracle) for name in ("A12", "D05")}
    path = _nc_manifest(manifest, columns, tmp_path / "manifest.json")
    data = json.loads(path.read_text())
    data["smokes"]["lc_every_restriction"] = {"test": smoke}
    path.write_text(json.dumps(data))
    monkeypatch.setattr(builder, "MANIFEST", path)
    results = tmp_path / "results"
    results.mkdir()
    for name, tier in (("A12", "default"), ("A12", "nightly"),
                       ("A12@sectors", "nightly"), ("D05", "default")):  # fmt: skip
        record = {"row": name, "tier": tier, "points": _points(name, tier),
                  "passed": True, "checks": 1, "failures": 0,
                  "check_records": []}  # fmt: skip
        (results / f"{name}.{tier}.json").write_text(json.dumps(record))
    cases = [
        ("test_loop_v2_nc_projection", f"{projection.split('::')[1]}[A12]", ""),
        ("test_loop_v2_nc_projection", f"{projection.split('::')[1]}[D05]",
         "<failure message='x'/>"),
        ("test_loop_v2_nc_oracles", "test_oracle[0]", ""),
        ("test_loop_v2_nc_oracles", "test_oracle[1]", ""),
        ("test_loop_v2_nc_declaration", f"{smoke.split('::')[1]}[default]", ""),
        ("test_loop_v2_nc_declaration", f"{smoke.split('::')[1]}[ckm]", "<skipped/>"),
    ]  # fmt: skip
    junit = tmp_path / "nc.xml"
    junit.write_text(
        "<testsuites><testsuite>"
        + "".join(
            f'<testcase classname="tests.unit.{module}" name="{name}">{body}</testcase>'
            for module, name, body in cases
        )
        + "</testsuite></testsuites>"
    )
    built = builder.build([results], {}, None, [junit])
    by_id = {row["id"]: row for row in built["rows"]}
    assert by_id["A12"]["nc"]["status"] == "pass"
    assert by_id["A12"]["status"] == "pass"
    assert by_id["D05"]["nc"]["tests"][f"{projection}[D05]"] == "failed"
    assert by_id["D05"]["status"] == "fail"
    assert by_id["A01"]["nc"] is None  # pending MQ-NC in this manifest
    assert builder._nc_cell(by_id["A01"]) == "pending"
    lc = built["smokes_and_expansions"]["lc:every_restriction"]
    assert lc["status"] == "not-run" and lc["tests"][smoke] == "skipped"
    table = builder.markdown(built).splitlines()
    line = next(x for x in table if x.startswith("| D05"))
    assert "| fail |" in line and "r7 test_lc_and_nlc" in line
    # Without the JUnit XML the nc column is not run, and so is its row.
    built = builder.build([results], {}, None)
    assert {row["id"]: row["status"] for row in built["rows"]}["A12"] == "not-run"


def test_m01_is_composite_of_both_routes_and_mq_model_s_bitwise_test(rows):
    """PLAN §4.2 M01: no evaluation of its own; each MadLoop P reference
    against the subprocess's own generation (tools route, with Catani) and
    against the public route's multiparticle artifact (same points and
    reference), and MQ-MODEL's test that each subprocess of that artifact is
    bitwise its own generation, by an id that exists."""

    m01 = rows["M01"]
    assert m01.status == "composite" and not m01.active
    parts = {name: row for name, row in rows.items() if row.base == "M01"}
    assert set(parts) == {"M01@P0", "M01@P2", "M01@public-P0", "M01@public-P2"}
    for name in ("P0", "P2"):
        tools, public = parts[f"M01@{name}"], parts[f"M01@public-{name}"]
        assert tools.route == "tools" and public.route == "public"
        assert tools.catani == "python" and public.catani == "none"
        assert public.reference == tools.reference and public.points == tools.points
        assert public.process == tools.process and public.subprocess
        assert public.gating and public.active
    assert set(mqcd.nightly_ids()) >= set(parts) and "M01" not in mqcd.nightly_ids()
    (node,) = m01.delegated["bitwise"]
    path, function = node.split("::")
    assert f"def {function}(" in (mqcd.ROOT / path).read_text()


def test_public_route_values_become_absolute_estimates():
    """A public ``VirtualEvaluation`` point: Born and Laurent coefficients
    per sector key, the per-point relative estimate made absolute on the
    single sector (an infinite relative estimate of an exact zero takes the
    sector's relative bound times its largest coefficient)."""

    from types import SimpleNamespace

    laurent = SimpleNamespace(finite=2.0, single_pole=-0.5, double_pole=0.0)
    estimate = SimpleNamespace(
        relative=1e-12, finite=1e-13, single_pole=4e-13, double_pole=math.inf
    )
    result = SimpleNamespace(
        born_orders=((("QCD", 4), ("QED", 0)),),
        born_values=((1.5,),),
        sectors=(SimpleNamespace(order=(("QCD", 6),)),),
        values=((laurent,),),
        stability=(estimate,),
    )
    point = mqcd.public_evaluated(result, 0, "double", 0.0)
    assert point.born == {(("QCD", 4),): Decimal("1.5")}
    assert point.virtual == {(("QCD", 6),): (Decimal(2), Decimal("-0.5"), Decimal(0))}
    (budget,) = point.estimate.values()
    assert budget == pytest.approx((2e-13, 2e-13, 2e-12), rel=1e-12)
    assert point.catani is None
    result.stability = None
    assert mqcd.public_evaluated(result, 0, "double", 0.0).estimate is None


def test_report_data_builds_the_composite_row_from_its_parts(
    tmp_path, monkeypatch, manifest
):
    """build_mqcd.py: M01 passes when its four variants, its expansion check
    and MQ-MODEL's bitwise test passed; a failed bitwise test fails it, a
    missing expansion record leaves it not-run."""

    builder = _report_builder()
    path = _nc_manifest(manifest, {}, tmp_path / "manifest.json")
    monkeypatch.setattr(builder, "MANIFEST", path)
    results = tmp_path / "results"
    results.mkdir()
    for name in ("M01@P0", "M01@P2", "M01@public-P0", "M01@public-P2"):
        record = {"row": name, "tier": "nightly", "points": _points(name, "nightly"),
                  "passed": True, "checks": 1, "failures": 0,
                  "check_records": []}  # fmt: skip
        (results / f"{name}.nightly.json").write_text(json.dumps(record))
    (node,) = mqcd.rows()["M01"].delegated["bitwise"]
    path, function = node.split("::")
    module = path.removesuffix(".py").replace("/", ".")
    junit = tmp_path / "bitwise.xml"

    def write_junit(body):
        junit.write_text(
            f'<testsuites><testsuite><testcase classname="{module}" '
            f'name="{function}">{body}</testcase></testsuite></testsuites>'
        )

    write_junit("")

    def m01():
        built = builder.build([results], {}, None, [junit])
        return next(r for r in built["rows"] if r["id"] == "M01")

    assert m01()["status"] == "not-run"  # no expansion record
    expansion = {"row": "expansion:M01", "tier": "nightly", "passed": True,
                 "checks": 1, "failures": 0, "check_records": []}  # fmt: skip
    (results / "expansion:M01.nightly.json").write_text(json.dumps(expansion))
    assert m01()["status"] == "pass"
    assert m01()["delegated"]["bitwise"]["status"] == "pass"
    write_junit("<failure message='x'/>")
    assert m01()["status"] == "fail"


def test_restricted_ufo_variants_wait_for_mq5f(rows):
    """The 5F* rows' restricted-UFO variants (PLAN §3.0 MQ-5F): the base
    request and reference on no_b_mass, pending until MQ-5F lands, and out of
    the nightly tier while pending."""

    nightly = set(mqcd.nightly_ids())
    for name in ("D17", "F01", "F02", "F04"):
        base, variant = rows[name], rows[f"{name}@5F"]
        assert base.restriction_code == "5F*" and variant.restriction == "no_b_mass"
        assert variant.reference == base.reference and variant.process == base.process
        assert variant.gating and not variant.subprocess
        if not variant.active:
            assert variant.pending["waits_for"] == "MQ-5F"
            assert variant.id not in nightly


def test_reference_records_without_a_reference_skip_rule1_only():
    """MQ-REF's per-point record: status "none" is one not-asserted rule 1
    check with the reason; an absolute point must be a manifest zero point
    with the record's neighbour."""

    reference = _reference({"madloop": (2.0, 1.0, -3.0, 0.5)})
    reference.records = [
        {"status": "none", "reason": "unstable", "comparison": "relative"}
    ]
    (check,) = mqcd.reference_record_checks(_row(), reference, 0)
    assert check.passed and not check.asserted and "unstable" in check.note
    assert check.coefficient == "no reference"
    absolute = {"status": "ok", "comparison": "absolute", "absolute_scale_row": 2}
    reference.records = [absolute]
    (named,) = mqcd.reference_record_checks(_row(zero_points={"0": 2}), reference, 0)
    assert named.passed
    (stale,) = mqcd.reference_record_checks(_row(), reference, 0)
    assert not stale.passed
    reference.records = [{"status": "ok", "comparison": "relative"}]
    assert mqcd.reference_record_checks(_row(), reference, 0) == []
    reference.records = []
    assert mqcd.reference_record_checks(_row(), reference, 0) == []


def _required_fixture(path, providers, record=None, point=("500", "0", "0", "500")):
    """A one-point required-format fixture: provider -> (born, finite,
    single, double, accuracy)."""

    row = {
        "providers": {
            name: dict(zip(("born", *mqcd.LAURENT, "accuracy"), values, strict=True))
            for name, values in providers.items()
        }
    }
    if record is not None:
        row["reference"] = record
    data = {
        "points": [[list(point), [point[0], "0", "0", f"-{point[3]}"]]],
        "mu_r": ["1000"],
        "parameters": {"aS": "0.118"},
        "fixed_parameters": {"MB": "0"},
        "primary_provider": next(iter(providers)),
        "rows": [row],
    }
    path.write_text(json.dumps(data))
    return str(path)


def test_superseded_points_take_the_binding_rerun(tmp_path):
    """A point of an existing reference superseded by MQ-REF's quadruple-
    precision rerun (point_binding): rule 1 holds the rerun's primary
    values, the existing fixture's second providers stay; a rerun at
    another point, with other parameters or without an ok record is
    refused."""

    ok = {"status": "ok", "comparison": "relative"}
    existing = _required_fixture(
        tmp_path / "old.json",
        {"madloop": (2.0, 1.0, -3.0, 0.5, 0.0), "recola": (2.0, 1.0, -3.0, 0.5, None)},
    )
    rerun = _required_fixture(
        tmp_path / "new.json", {"madloop": (2.0, 1.5, -3.0, 0.5, 0.0)}, ok
    )
    spec = {
        "format": "required",
        "path": existing,
        "providers": ["madloop", "recola"],
        "superseded": {"0": {"path": rerun, "row": 0}},
    }
    reference = mqcd.load_reference(_row(reference=spec))
    assert reference.binding == {0: (rerun, 0)}
    assert reference.virtual[0]["madloop"][mqcd.TOTAL] == (1.5, -3.0, 0.5)
    assert reference.virtual[0]["recola"][mqcd.TOTAL] == (1.0, -3.0, 0.5)
    assert mqcd.reference_record_checks(_row(), reference, 0) == []
    dd = _point("double-double", 2.0, (1.5, -3.0, 0.5))
    assert all(c.passed for c in mqcd.rule1(_row(), reference, 0, dd))
    for name, kwargs in (
        ("moved.json", {"point": ("600", "0", "0", "600")}),
        ("noref.json", {}),
    ):
        bad = _required_fixture(
            tmp_path / name,
            {"madloop": (2.0, 1.5, -3.0, 0.5, 0.0)},
            ok if kwargs else {"status": "none", "reason": "unstable"},
            **kwargs,
        )
        with pytest.raises(ValueError):
            mqcd.load_reference(
                _row(reference={**spec, "superseded": {"0": {"path": bad, "row": 0}}})
            )
    other = json.loads(Path(rerun).read_text())
    other["fixed_parameters"] = {"MB": "4.7"}
    (tmp_path / "params.json").write_text(json.dumps(other))
    with pytest.raises(ValueError):
        mqcd.load_reference(
            _row(
                reference={
                    **spec,
                    "superseded": {
                        "0": {"path": str(tmp_path / "params.json"), "row": 0}
                    },
                }
            )
        )


def test_manifest_zero_points_are_the_references_absolute_points(rows):
    """Every absolute reference point (MQ-REF's S3 records) is a zero point
    of its row with the same regular neighbour, and vice versa."""

    for name, row in rows.items():
        if not row.active or row.reference is None:
            continue
        if row.reference["format"] != "required":
            continue
        data = json.loads((mqcd.ROOT / row.reference["path"]).read_text())
        absolute = {
            i: entry["reference"]["absolute_scale_row"]
            for i, entry in enumerate(data["rows"])
            if (entry.get("reference") or {}).get("comparison") == "absolute"
        }
        assert absolute == dict(row.zero_points), name


def test_ward_rule_reads_the_ward_criterion():
    from pyamplicol.loops.v2 import ward

    def entry(virtual, born=1e-17, passed=True, **extra):
        return {
            "leg": 1,
            "factor": (1, 0),
            "born": born,
            "born_passed": born <= 1e-12,
            "parts": {"virtual": virtual},
            "virtual_passed": passed,
            **extra,
        }

    dd = mqcd.rule_ward(
        _row(), 0, "double-double", {"rows": [entry([1e-33, 2e-32, 0.0])]}
    )
    assert [c.coefficient for c in dd] == ["born", "virtual"] and all(
        c.passed for c in dd
    )
    assert dd[1].error == 2e-32 and dd[1].bound == ward.VIRTUAL_BOUNDS["double-double"]
    b64 = entry(
        [1e-16, 1e-15, 0.0],
        o1_over=[1e-4, 2e-3, 0.0],
        estimate_over=[0.1, 0.3, 0.0],
        estimate=[3e-15, 3e-15, 0.0],
        in_scope=True,
    )
    _born, virtual = mqcd.rule_ward(_row(), 0, "double", {"rows": [b64]})
    assert virtual.error == 2e-3 and virtual.bound == 1.0 and virtual.passed
    failing = entry(
        [1e-10, 0.0, 0.0],
        passed=False,
        o1_over=[100.0, 0, 0],
        estimate_over=[1e3, 0, 0],
        estimate=[1e-13, 0, 0],
        in_scope=True,
    )
    assert not mqcd.rule_ward(_row(), 0, "double", {"rows": [failing]})[1].passed
    # Outside the binary64 scope (E_k > 1e-9 of the scale): not asserted,
    # still bracketed tenfold.
    outside = entry(
        [1e-9, 0.0, 0.0],
        passed=False,
        o1_over=[0.03, 0, 0],
        estimate_over=[0.03, 0, 0],
        estimate=[3e-8, 0, 0],
        in_scope=False,
    )
    reported = mqcd.rule_ward(_row(), 0, "double", {"rows": [outside]})[1]
    assert reported.passed and not reported.asserted
    unbracketed = {**outside, "estimate_over": [11.0, 0, 0]}
    assert not mqcd.rule_ward(_row(), 0, "double", {"rows": [unbracketed]})[1].passed
    # No reduction-direction estimate: no binary64 criterion, a failure.
    assert not mqcd.rule_ward(
        _row(), 0, "double", {"rows": [entry([0, 0, 0], passed=None)]}
    )[1].passed


def test_ward_extras_are_checked_on_their_rows(rows):
    """PLAN §4.2's ward column (A08, A14, B03, C01, D05, E01, F04) and the
    VV + jet rows D28, D29: checked by the harness, A14 on one leg."""

    ward_rows = {name for name, row in rows.items() if "ward" in row.extras}
    assert ward_rows == {"A08", "A14", "B03", "C01", "D05", "D28", "D29", "E01", "F04"}
    for name in ward_rows:
        assert "ward" not in rows[name].extras_pending, name
    assert rows["A14"].ward_legs == (1,)
    assert all(rows[n].ward_legs is None for n in ward_rows - {"A14"})
