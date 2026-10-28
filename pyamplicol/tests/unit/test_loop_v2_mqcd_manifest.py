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
from decimal import Decimal

import pytest

from tools.loop_benchmark import mqcd

pytestmark = pytest.mark.loop_fast

# PLAN §4.2, verbatim row ids per block (§4.3: 76 core rows).
PLAN_ROWS = {
    "A": "01 02 03 04 05 06 07 08 09 12 13 14 15 18 19 20 21 22 23 24 25 26",
    "B": "01 02 03 05 06 07 08",
    "C": "01 02 03 04 05 06 07 08 09",
    "D": "01 02 05 06 07 08 09 10 11 12 13 15 16 17 18 19 20 21 23 24 25 26 27",
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
    assert len(expected) == 76
    assert set(rows) == expected
    assert {name for name, row in rows.items() if row.tier == "default"} == DEFAULT_TIER


def test_rows_are_well_formed(manifest, rows):
    codes = manifest["restriction_codes"]
    for name, row in rows.items():
        assert row.cls in CLASSES, name
        assert row.tier in mqcd.TIERS and row.status in mqcd.STATUSES, name
        assert row.corrections == ("QCD",), name
        assert set(row.extras) <= set(mqcd.EXTRAS), name
        # An extra the harness does not check yet names its package.
        for extra in set(row.extras) - mqcd.IMPLEMENTED_EXTRAS:
            assert extra in row.extras_pending, (name, extra)
        assert set(row.extras_pending) <= set(row.extras), name
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
        else:
            assert row.pending is None and row.reference is not None, name
            assert row.reference["format"] in mqcd.REFERENCE_FORMATS, name
        if isinstance(row.loop_orders, dict):
            # One loop order: the Born's plus the correction's two powers.
            assert row.loop_orders == {
                **row.born_orders,
                "QCD": row.born_orders["QCD"] + 2,
            }, name


def test_features_cover_the_plan(manifest, rows):
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
            assert " ".join(tokens) == row.process, name
            assert data["born_orders"] == dict(row.born_orders), name
            assert data["loop_orders"] == dict(row.loop_orders), name
        for point in reference.points:
            assert len(point) == len(row.process.replace(">", "").split()), name


def test_listings_name_real_points_and_coefficients(rows):
    for name, row in rows.items():
        for entry in (
            *row.binary64_out_of_scope,
            *row.reference_outliers,
        ):
            assert entry["point"] in {i for p in row.points.values() for i in p}, name
            assert entry["coefficient"] in mqcd.COEFFICIENTS, name
        for entry in row.binary64_out_of_scope:
            # Listed with E and its true error (PLAN §1.4 rule 2).
            assert entry["E"] > mqcd.O1_CAP and entry["true_error"] >= 0, name
        for entry in row.vanishing:
            assert entry["coefficient"] in mqcd.LAURENT, name


def test_pending_rows_and_extras_name_their_package(rows):
    pending = {
        name: row.pending["waits_for"] for name, row in rows.items() if not row.active
    }
    waiting = {
        (name, extra): reason
        for name, row in rows.items()
        for extra, reason in row.extras_pending.items()
    }
    assert all(package in mqcd.UNBLOCKERS for package in pending.values())
    assert all(reason.split()[0] in mqcd.UNBLOCKERS for reason in waiting.values())


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
    # E > 1e-9: outside the scope, which the manifest must list.
    found = checks(_row(), 1e-8, 2e-9)
    assert not found["2", "finite"].asserted and not found["2", "finite"].passed
    listed = _row(
        binary64_out_of_scope=[
            {"point": 0, "sector": "QCD^8", "coefficient": "finite", "E": 2e-9,
             "true_error": 1e-8}
        ]
    )  # fmt: skip
    found = checks(listed, 1e-8, 2e-9)
    assert found["2", "finite"].passed and not found["2", "finite"].asserted
    assert found["3", "finite"].passed
    # A listing for an in-scope coefficient is stale.
    found = checks(listed, 1e-13, 1e-13)
    assert not found["2", "finite"].passed


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


def test_field_factor_poles():
    from fractions import Fraction

    # Units of alpha_s / 4 pi: Z_3 = 5/3 C_A - 4/3 T_F n_f, Z_2 = -C_F.
    gluon = Fraction(5, 3) * 3 - Fraction(4, 3) * Fraction(1, 2) * 6
    assert mqcd.field_poles([(8, False)] * 4, 6) == 4 * gluon == 4
    assert mqcd.field_poles([(3, False), (3, True), (1, False)], 5) == Fraction(-8, 3)
    with pytest.raises(ValueError):
        mqcd.field_poles([(6, False)], 5)


def test_sector_keys_drop_zero_powers():
    assert mqcd.sector_key({"QCD": 8, "QED": 0}) == (("QCD", 8),)
    assert mqcd.sector_key([["QED", 2], ["QCD", 4]]) == (("QCD", 4), ("QED", 2))
    assert mqcd.sector_text((("QCD", 4), ("QED", 2))) == "QCD^4 QED^2"
    assert mqcd.born_sector((("QCD", 6), ("QED", 2))) == (("QCD", 4), ("QED", 2))
