#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""NJet reference fixture for g g -> g g g g g and its pyAmpliCol comparison.

``build``    assemble ``tests/fixtures/loop/required/gg_ggggg_njet.json`` from
             ``run_njet.py run`` results (double-double values with NJet's
             scaling-test estimate, double-precision values as a second
             column) and the calibration summary.
``compare``  compare a pyAmpliCol record (``campaign.py`` JSON with the full
             ``virtual``, or ``loop_v2_generate.py poles`` JSON with the
             ``bare`` virtual) with the fixture, coefficient by coefficient,
             under the rules of ``comparison_rules`` below.

The fixture is massless five-flavour QCD (NJet has no massive quarks), while
pyAmpliCol's ``loop_sm`` virtual contains closed top-quark loops (MT=173) with
the top decoupled from the running of alpha_s. Consequences, all derived (no
fitted factor):

* Born and the double pole agree directly (the decoupled top changes neither).
* The renormalised single pole agrees directly: the top's counterterm cancels
  the top loops' UV pole, and the IR poles involve only the massless legs and
  beta_0 of the five light flavours.
* The bare single pole (no counterterm, no R2) exceeds the renormalised one by
  the counterterm's pole, (alpha_s/4pi) B [(n-2) beta_0(5) + (8/3) T_F]:
  (n-2) beta_0(5) from MS-bar coupling renormalisation with five flavours
  (beta_0 = 11/3 C_A - 4/3 T_F n_l) and (8/3) T_F from the top: in the
  decoupling scheme Z_g Z_A^(1/2) = 1 for the heavy-quark part, so the top
  counterterm of an amplitude with n external gluons and n-2 couplings is
  dZ_A^(t) A_0 with dZ_A^(t) = -(alpha_s/4pi)(4/3) T_F (1/eps + log(mu^2/MT^2)),
  i.e. -(alpha_s/4pi)(8/3) T_F B / eps in the interference, independent of n.
* The finite part differs by the top-loop contribution T (top loops plus their
  decoupling counterterm); it is compared only for a pyAmpliCol record that
  declares ``"top_quark_loops": false``, i.e. one generated with
  ``particle_veto = ["t"]`` (the public generation option of v2 virtuals,
  REQUIREMENTS_ADDENDUM J). Otherwise the difference is reported as
  ``finite_minus_reference`` and is NOT a test.
"""

import argparse
import json
import math
from fractions import Fraction
from pathlib import Path

COEFFICIENTS = ("born", "finite", "single_pole", "double_pole")
LAURENT = ("finite", "single_pole", "double_pole")
NC = 3
T_F = Fraction(1, 2)
N_LIGHT = 5
CA = Fraction(NC)
BETA0_LIGHT = Fraction(11, 3) * CA - Fraction(4, 3) * T_F * N_LIGHT  # 23/3
SCHEMA_VERSION = 1


def _f(x):
    return float(x)


def universal_double_pole_factor(legs, alphas):
    """-(alpha_s/2pi) sum_i C_A over the n massless gluons."""
    return -float(alphas) * legs * NC / (2 * math.pi)


def bare_single_pole_shift(legs, alphas, born):
    """Bare minus renormalised single pole of loop_sm (top decoupled), see module doc."""
    units = (legs - 2) * BETA0_LIGHT + Fraction(8, 3) * T_F
    return float(alphas) / (4 * math.pi) * float(units) * born


def merge_metric(a, b, floor=0.0):
    """|a-b| / max(|a|, |b|, floor), as tools/loop_reference/references.py."""
    scale = max(abs(a), abs(b), floor)
    return abs(a - b) / scale if scale else 0.0


# ------------------------------------------------------------------ build


def _rows_by_index(paths):
    rows, meta = {}, None
    for path in paths:
        data = json.loads(Path(path).read_text())
        meta = meta or data
        for row in data["rows"]:
            rows[row["index"]] = (row, data)
    return rows, meta


def _value_entry(row, alphas):
    values = {k: row[k] for k in COEFFICIENTS}
    errors = dict(
        zip(("double_pole", "single_pole", "finite"), row["errors"], strict=True)
    )
    floor = abs(_f(row["born"])) * float(alphas) / (2 * math.pi)
    estimate = {k: abs(_f(errors[k])) / max(abs(_f(values[k])), floor) for k in LAURENT}
    estimate["born"] = abs(_f(row["born_error"])) / abs(_f(row["born"]))
    return values, estimate


def _labelled(specs):
    out = {}
    for spec in specs or []:
        label, pattern = spec.split("=", 1)
        out[label] = _rows_by_index(sorted(Path("/").glob(pattern.lstrip("/"))))
    return out


def _run_summary(meta):
    return {
        "tool": meta["tool"],
        "build": meta["njet_build"],
        "options": meta["options"],
        "driver_source_sha256": meta["driver"]["source_sha256"],
        "host": meta["host"],
        "cpu_model": meta["cpu_model"],
    }


def build(args):
    points = json.loads(Path(args.points).read_text())
    calibration = json.loads(Path(args.calibration).read_text())
    catani = {r["index"]: r for r in json.loads(Path(args.catani).read_text())["rows"]}
    catani_meta = json.loads(Path(args.catani).read_text())
    alphas = args.alphas
    legs = points["legs"]
    ((primary_label, (primary, primary_meta)),) = _labelled([args.primary]).items()
    secondary = _labelled(args.secondary)
    factor = universal_double_pole_factor(legs, alphas)
    rows = []
    worst = {
        "universal_double_pole": 0.0,
        "catani_single_pole": 0.0,
        "estimate": 0.0,
        "secondary": 0.0,
    }
    for i in range(len(points["points"])):
        if i not in primary:
            raise SystemExit(f"no {primary_label} value for point {i}")
        values, estimate = _value_entry(primary[i][0], alphas)
        born = _f(values["born"])
        floor = abs(born) * float(alphas) / (2 * math.pi)
        providers = {
            primary_label: {
                **values,
                "estimated_relative_accuracy": estimate,
                "seconds": primary[i][0]["seconds"],
            }
        }
        for label, (srows, _) in secondary.items():
            if i not in srows:
                continue
            svals, sest = _value_entry(srows[i][0], alphas)
            dev = {
                k: merge_metric(
                    _f(svals[k]), _f(values[k]), 0.0 if k == "born" else floor
                )
                for k in COEFFICIENTS
            }
            providers[label] = {
                **svals,
                "estimated_relative_accuracy": sest,
                "seconds": srows[i][0]["seconds"],
                "relative_deviation_from_primary": dev,
            }
            worst["secondary"] = max(worst["secondary"], *dev.values())
        c = catani[i]
        catani_dev = {
            k: merge_metric(_f(values[k]), _f(c[k]), floor)
            for k in ("single_pole", "double_pole")
        }
        providers["catani"] = {
            "born": c["born"],
            "single_pole": c["single_pole"],
            "double_pole": c["double_pole"],
            "colour_conservation_residual": c["colour_conservation"],
            "relative_deviation_of_primary": catani_dev,
        }
        pole = merge_metric(_f(values["double_pole"]), factor * born)
        worst["universal_double_pole"] = max(worst["universal_double_pole"], pole)
        worst["catani_single_pole"] = max(
            worst["catani_single_pole"], catani_dev["single_pole"]
        )
        worst["estimate"] = max(worst["estimate"], *estimate.values())
        rows.append(
            {
                **values,
                "estimated_relative_accuracy": estimate,
                "universal_double_pole_relative_error": pole,
                "providers": providers,
            }
        )
    rotation = primary_meta["options"].get("rotation")
    fixture = {
        "schema_version": SCHEMA_VERSION,
        "process": "g g > g g g g g QED=0 [virt=QCD]",
        "name": "gg_ggggg_njet",
        "pdgs": [21] * legs,
        "correction": "QCD",
        "born_orders": {"QCD": legs - 2, "QED": 0},
        "loop_orders": {"QCD": legs, "QED": 0},
        "scheme": "massless-qcd-nf5",
        "scheme_description": (
            "massless QCD: five massless quark flavours (u, d, s, c, b) in the loops and in "
            "the MS-bar running of alpha_s(mu_R); NO top quark (NJet has no massive quarks). "
            "pyAmpliCol's loop_sm (MT=173, top loops, top decoupled from alpha_s) differs from "
            "this only in the finite part; see comparison_rules."
        ),
        "source": (
            f"{primary_meta['tool']} (https://bitbucket.org/njet/njet, release tarball), "
            "numerical colour-summed Amp0q7g through NJetAccuracy, driver "
            "tools/loop_reference/njet/njet_driver.cpp; poles also from NJet 2.1.1's "
            "colour-correlated Borns and the Catani I operator"
        ),
        "scope": (
            "Independent reference values at pyAmpliCol's exact g g > g g g g g points "
            "(points 0-2 are those of the work/v2-generation-scale campaign); no pyAmpliCol "
            "code involved."
        ),
        "evaluation_orientation": (
            None
            if not rotation
            else {
                **rotation,
                "reason": (
                    "NJet's numerics depend on the orientation of the event: at the points as "
                    "given (beams along +-z) NJet 3.1.2 and 2.1.1 lose 4-6 digits in double "
                    "precision (NaN at points 7-9) and the QCDLoop double-double mode is no "
                    "better; at generic orientations the poles agree with the Catani "
                    "prediction to <= 3e-9 in double. The virtual is a Lorentz scalar, so the "
                    "values are those of the points as given."
                ),
            }
        ),
        "convention": (
            "MadLoop/BLHA C_eps=(4pi)^eps/Gamma(1-eps); CDR/HV ('t Hooft-Veltman) with MS-bar "
            "alpha_s(mu_R); physical couplings g_s^2=4 pi alpha_s; initial spin and colour "
            "average and final identical-gluon factor 1/5!; the virtual is 2 Re<M0|M1>. "
            "This is the convention of every tests/fixtures/loop/required fixture (MadLoop, "
            "OpenLoops truepoles=1 polenorm=0, Recola +pi^2/6 D) and of pyAmpliCol "
            "(loops/v2/conventions.py); calibrated without any factor, see calibration."
        ),
        "normalization": {
            "initial_average": 256,
            "final_symmetry": math.factorial(legs - 2),
            "denominator": 256 * math.factorial(legs - 2),
        },
        "normalization_denominator": 256 * math.factorial(legs - 2),
        "points": points["points"],
        "mu_r": points["mu_r"],
        "point_construction": {
            "construction": points["source"],
            "cuts": {"min_2pipj_over_s": "1/1000"},
        },
        "parameters": {"aS": str(alphas)},
        "active_flavours": N_LIGHT,
        "loop_quark_flavours": N_LIGHT,
        "top_quark": "absent (massless five-flavour QCD)",
        "comparison_rules": {
            "born": "direct",
            "double_pole": "direct (also for a bare virtual: counterterms and R2 have no 1/eps^2)",
            "single_pole": (
                "direct for a renormalised virtual; a bare loop_sm virtual is "
                "compared after subtracting (alpha_s/4pi) B [(n-2) beta_0(5) + "
                "(8/3) T_F] = (alpha_s/4pi) B (119/3) at n=7 "
                "(njet_fixture.bare_single_pole_shift)"
            ),
            "finite": (
                "only for a top-free pyAmpliCol configuration (record "
                "'top_quark_loops': false); with loop_sm's top loops the difference "
                "is the top-loop contribution, reported, not tested"
            ),
        },
        "rows": rows,
        "primary_provider": primary_label,
        "universal_double_pole_factor": factor,
        "maximum_universal_double_pole_relative_error": worst["universal_double_pole"],
        "maximum_catani_single_pole_relative_error": worst["catani_single_pole"],
        "maximum_estimated_relative_accuracy": worst["estimate"],
        "maximum_secondary_relative_deviation": worst["secondary"],
        "error_metric": (
            "|a-b|/max(|a|,|b|,|Born| alpha_s/(2pi)) for Laurent coefficients, "
            "|a-b|/max(|a|,|b|) for the Born (references.py merge metric); "
            "estimated_relative_accuracy = |NJet scaling-test difference| / "
            "max(|value|, |Born| alpha_s/(2pi))"
        ),
        "providers": {
            primary_label: {
                **_run_summary(primary_meta),
                "build_info": primary_meta["build_info"],
            },
            **{label: _run_summary(meta) for label, (_, meta) in secondary.items()},
            "catani": {
                "tool": catani_meta["tool"],
                "nf": catani_meta["nf"],
                "formula": (
                    "single = (alpha_s/2pi)[-n gamma_g B + sum_{i!=j} <T_i.T_j> "
                    "log(mu^2/|2p_i.p_j|)], gamma_g = 11/6 C_A - 2/3 T_F n_f; "
                    "double = -(alpha_s/2pi) n C_A B (run_njet.py catani)"
                ),
            },
            "accuracy_estimate": (
                "NJet's scaling test (NJetAccuracy::virt): the amplitude is evaluated at the "
                "point and with all momenta (and mu_R) rescaled by 1.2857142857142855; the "
                "value is their average, the estimate their difference."
            ),
        },
        "calibration": calibration,
    }
    Path(args.output).write_text(json.dumps(fixture, indent=1) + "\n")
    print(args.output)


# ------------------------------------------------------------- validation


def validate_fixture(data):
    """Schema and internal consistency checks; raises ValueError."""

    def need(cond, message):
        if not cond:
            raise ValueError(message)

    for key in (
        "schema_version",
        "process",
        "pdgs",
        "born_orders",
        "loop_orders",
        "points",
        "mu_r",
        "parameters",
        "rows",
        "convention",
        "comparison_rules",
        "universal_double_pole_factor",
        "providers",
        "calibration",
        "active_flavours",
        "loop_quark_flavours",
        "normalization_denominator",
    ):
        need(key in data, f"missing {key}")
    need(data["schema_version"] == SCHEMA_VERSION, "schema_version")
    legs = len(data["pdgs"])
    need(
        data["normalization_denominator"] == 256 * math.factorial(legs - 2),
        "normalisation",
    )
    need(len(data["points"]) == len(data["rows"]) == len(data["mu_r"]), "row count")
    need("MT" not in data["parameters"], "a massless reference must not carry MT")
    need(data["loop_quark_flavours"] == data["active_flavours"] == N_LIGHT, "flavours")
    alphas = Fraction(data["parameters"]["aS"])
    factor = universal_double_pole_factor(legs, alphas)
    need(
        abs(data["universal_double_pole_factor"] - factor) <= 1e-15 * abs(factor),
        "factor",
    )
    for point, mu, row in zip(data["points"], data["mu_r"], data["rows"], strict=True):
        need(len(point) == legs and all(len(p) == 4 for p in point), "point shape")
        exact = [[Fraction(x) for x in p] for p in point]
        for p in exact:
            need(
                p[0] ** 2 == p[1] ** 2 + p[2] ** 2 + p[3] ** 2 and p[0] > 0,
                "mass shell",
            )
        for mu_index in range(4):
            need(
                exact[0][mu_index] + exact[1][mu_index]
                == sum(p[mu_index] for p in exact[2:]),
                "momentum conservation",
            )
        need(Fraction(mu) > 0, "mu_r")
        for key in COEFFICIENTS:
            need(isinstance(row[key], str) and math.isfinite(float(row[key])), key)
        born = float(row["born"])
        need(born > 0, "born")
        estimate = row["estimated_relative_accuracy"]
        need(set(estimate) == set(COEFFICIENTS), "estimates")
        pole = merge_metric(float(row["double_pole"]), factor * born)
        need(pole <= max(10 * estimate["double_pole"], 1e-13), "universal double pole")
        need(data["primary_provider"] in row["providers"], "primary provider row")
        catani = row["providers"]["catani"]
        need(catani["colour_conservation_residual"] < 1e-13, "colour conservation")
        floor = born * float(alphas) / (2 * math.pi)
        single = merge_metric(
            float(row["single_pole"]), float(catani["single_pole"]), floor
        )
        need(single <= max(10 * estimate["single_pole"], 1e-12), "Catani single pole")
    return True


# ------------------------------------------------------------- comparison


def compare_pyamplicol(fixture, record, kind=None):
    """Per-point comparison of a pyAmpliCol record with the NJet fixture.

    ``record`` is a ``campaign.py`` record (points with ``virtual``) or a
    ``loop_v2_generate.py poles`` record (points with ``bare``). Points are
    matched by index: both use ``exact_points.generate`` for massless legs in
    the same order; the Born doubles as a guard (it must agree to 1e-10).
    """
    legs = len(fixture["pdgs"])
    process = " ".join(
        t for t in fixture["process"].split(" [")[0].split() if "=" not in t
    )
    if record.get("process") != process:
        raise ValueError(f"record process {record.get('process')!r} is not {process!r}")
    alphas = Fraction(fixture["parameters"]["aS"])
    top_free = record.get("top_quark_loops") is False
    out = []
    for entry in record["points"]:
        index = entry["index"]
        row = fixture["rows"][index]
        kind_here = kind or ("virtual" if "virtual" in entry else "bare")
        finite, single, double = (float(x) for x in entry[kind_here])
        born = float(entry["born"])
        ref = {k: float(row[k]) for k in COEFFICIENTS}
        floor = abs(ref["born"]) * float(alphas) / (2 * math.pi)
        result = {
            "index": index,
            "kind": kind_here,
            "born": merge_metric(born, ref["born"]),
        }
        if result["born"] > 1e-10:
            raise ValueError(
                f"point {index}: Born differs by {result['born']:.2e}; not the "
                "fixture's point"
            )
        result["double_pole"] = merge_metric(double, ref["double_pole"], floor)
        reference_single = ref["single_pole"]
        if kind_here == "bare":
            reference_single += bare_single_pole_shift(legs, alphas, ref["born"])
        result["single_pole"] = merge_metric(single, reference_single, floor)
        if kind_here == "virtual" and top_free:
            result["finite"] = merge_metric(finite, ref["finite"], floor)
        elif kind_here == "virtual":
            result["finite_minus_reference"] = finite - ref["finite"]
            result["finite_minus_reference_over_floor"] = (
                finite - ref["finite"]
            ) / floor
        result["reference_estimate"] = row["estimated_relative_accuracy"]
        out.append(result)
    return out


def command_compare(args):
    fixture = json.loads(Path(args.fixture).read_text())
    record = json.loads(Path(args.record).read_text())
    for row in compare_pyamplicol(fixture, record, args.kind):
        print(json.dumps(row))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--points", required=True)
    b.add_argument("--primary", required=True, help="label=glob of run_njet.py results")
    b.add_argument("--secondary", nargs="*", help="label=glob of further results")
    b.add_argument("--catani", required=True, help="run_njet.py catani result")
    b.add_argument("--calibration", required=True)
    b.add_argument("--alphas", default="0.118")
    b.add_argument("--output", required=True)
    c = sub.add_parser("compare")
    c.add_argument("fixture")
    c.add_argument("record")
    c.add_argument("--kind", choices=("virtual", "bare"))
    args = parser.parse_args()
    {"build": build, "compare": command_compare}[args.command](args)


if __name__ == "__main__":
    main()
