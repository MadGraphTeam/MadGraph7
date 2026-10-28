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
from pathlib import Path

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
        # An extra the harness does not check yet names its package.
        for extra in set(row.extras) - mqcd.IMPLEMENTED_EXTRAS:
            assert extra in row.extras_pending, (name, extra)
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
        else:
            assert row.pending is None and row.reference is not None, name
            assert row.reference["format"] in mqcd.REFERENCE_FORMATS, name
        assert row.inputs in ("python", "native"), name
        # The native-input route forms no Catani prediction: such a row says
        # so (rule 4 recorded as not formed).
        assert row.catani == ("none" if row.inputs == "native" else "python"), name
        if row.base is not None:
            base = rows[row.base]
            assert row.tier == "nightly" and row.reference != base.reference, name
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
        for entry in (
            *row.binary64_out_of_scope,
            *row.reference_outliers,
        ):
            assert entry["point"] in {i for p in row.points.values() for i in p}, name
            assert entry["coefficient"] in mqcd.COEFFICIENTS, name
        for entry in row.binary64_out_of_scope:
            # Listed with E and its true error (PLAN §1.4 rule 2); a limit
            # point's Born (no estimate) only on a special-point variant.
            if entry["coefficient"] == "born":
                assert row.base is not None and entry["E"] is None, name
            else:
                assert entry["E"] > mqcd.O1_CAP, name
            assert entry["true_error"] >= 0, name
        for entry in row.vanishing:
            # A Laurent coefficient, or a colour-forbidden Born sector.
            assert entry["coefficient"] in mqcd.COEFFICIENTS, name


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
    # MQ-REF's record compares a suppressed helicity absolutely against the
    # summed Born (its F for the Laurent coefficients).
    madloop[2]["reference"] = {
        "status": "ok",
        "comparison": "absolute",
        "absolute_scale": 3.0,
        "reason": "suppressed helicity",
    }
    madloop[2]["finite"] = 4.0 + 1e-14  # 2.5e-15 relative, 1.8e-13 of F
    checks = mqcd.rule_hel(_row(), 0, rows, madloop, 0.118)
    finite = next(c for c in checks if (c.sector, c.coefficient) == ("h-++", "finite"))
    assert finite.passed and finite.bound == pytest.approx(
        1e-12 * 3.0 * 0.118 / (2 * math.pi)
    )
    assert "suppressed helicity" in finite.note
    madloop[2]["finite"] = 4.0 + 1e-12
    checks = mqcd.rule_hel(_row(), 0, rows, madloop, 0.118)
    assert not next(
        c for c in checks if c.coefficient == "finite" and c.sector == "h-++"
    ).passed
    del madloop[2]["reference"]
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


def _report_builder():
    import importlib.util

    path = mqcd.ROOT / "docs/performance_reports/loop/scripts/build_mqcd.py"
    spec = importlib.util.spec_from_file_location("loop_report_build_mqcd", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_data_counts_variants_smokes_and_findings(tmp_path):
    """build_mqcd.py: a failing gating variant fails its row, a non-gating
    one does not, smoke and expansion records are kept apart, a row without
    a record is not-run and a pending row stays pending; the findings name
    their rows in the per-row table."""

    builder = _report_builder()

    def record(name, tier, passed):
        check = {"rule": "1", "point": 0, "carrier": "double-double",
                 "sector": "total", "coefficient": "finite", "error": 2e-12,
                 "bound": 1e-12, "note": "", "passed": passed}  # fmt: skip
        data = {"row": name, "tier": tier, "passed": passed, "checks": 1,
                "failures": int(not passed), "check_records": [check]}  # fmt: skip
        (tmp_path / f"{name}.{tier}.json").write_text(json.dumps(data))

    record("A01", "default", True)
    record("A09", "default", True)
    record("A09@soft", "nightly", False)
    record("C04", "nightly", True)
    record("C04@z", "nightly", False)
    record("smoke:no_masses", "nightly", True)
    findings = {"findings": [{"id": "F9", "rows": ["A09@soft"]}]}
    data = builder.build([tmp_path], {}, findings)
    rows = {row["id"]: row for row in data["rows"]}
    assert rows["A01"]["status"] == "pass"
    assert rows["A09"]["status"] == "fail"
    assert rows["C04"]["status"] == "pass"
    assert rows["A02"]["status"] == "not-run"
    assert rows["C09"]["status"] == "pending"
    assert data["smokes_and_expansions"]["smoke:no_masses"]["passed"]
    assert len(data["rows"]) == 76
    assert sum(data["counts"].values()) == len(data["rows"])
    table = builder.markdown(data)
    line = next(x for x in table.splitlines() if x.startswith("| A09"))
    assert "fail" in line and "F9" in line and "soft: r1 p0" in line


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
    # Outside the binary64 scope (E_k > 1e-9 of the scale): listed, not
    # asserted, still bracketed tenfold; unlisted it fails.
    outside = entry(
        [1e-9, 0.0, 0.0],
        passed=False,
        o1_over=[0.03, 0, 0],
        estimate_over=[0.03, 0, 0],
        estimate=[3e-8, 0, 0],
        in_scope=False,
    )
    listing = [{"point": 0, "leg": 1, "factor": [1, 0], "E": 3e-8}]
    listed = mqcd.rule_ward(
        _row(ward_out_of_scope=listing), 0, "double", {"rows": [outside]}
    )[1]
    assert listed.passed and not listed.asserted
    assert not mqcd.rule_ward(_row(), 0, "double", {"rows": [outside]})[1].passed
    unbracketed = {**outside, "estimate_over": [11.0, 0, 0]}
    assert not mqcd.rule_ward(
        _row(ward_out_of_scope=listing), 0, "double", {"rows": [unbracketed]}
    )[1].passed
    # A listing of an evaluation within the scope is stale.
    assert not mqcd.rule_ward(
        _row(ward_out_of_scope=listing), 0, "double", {"rows": [b64]}
    )[1].passed
    # No reduction-direction estimate: no binary64 criterion, a failure.
    assert not mqcd.rule_ward(
        _row(), 0, "double", {"rows": [entry([0, 0, 0], passed=None)]}
    )[1].passed


def test_ward_extras_are_checked_on_their_rows(rows):
    """PLAN §4.2's ward column (A08, A14, B03, C01, D05, E01, F04): checked
    by the harness, A14 on one leg."""

    ward_rows = {name for name, row in rows.items() if "ward" in row.extras}
    assert ward_rows == {"A08", "A14", "B03", "C01", "D05", "E01", "F04"}
    for name in ward_rows:
        assert "ward" not in rows[name].extras_pending, name
    assert rows["A14"].ward_legs == (1,)
    for name, row in rows.items():
        for listing in row.ward_out_of_scope:
            assert name in ward_rows and listing["E"] > 1e-9, name
            legs = row.ward_legs or ()
            assert not legs or listing["leg"] in legs, name
    assert all(rows[n].ward_legs is None for n in ward_rows - {"A14"})
