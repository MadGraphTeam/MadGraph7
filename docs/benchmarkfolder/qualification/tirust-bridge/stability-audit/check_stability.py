#!/usr/bin/env python3
"""Independent reversal/rotation diagnostic for the retained N4R5 stress lane.

Run from the MadGraph root under .tirust-build/numerical-tests.lock. No production
source or card is changed. Output paths must be new, preserving prior attempts.
"""
import argparse
import ctypes as c
from decimal import Decimal, localcontext
import hashlib
import importlib.util
from itertools import product
import json
from math import comb
from pathlib import Path
import sys

LABEL = "li-rows.json LI-32 N4-massive R4 row 57 R=5 lane 44"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def estimator(values):
    """Literal COMPUTE_ACCURACY real-component formula, for diagnostic use."""
    averages, accuracies = [], []
    for column in zip(*values):
        delta = max(column) - min(column)
        avg = (max(column) + min(column)) / 2.0
        averages.append(avg)
        accuracies.append(delta / abs(avg) if avg else delta)
    answer = sum(a * abs(v) for a, v in zip(accuracies, averages))
    if sum(averages):
        answer /= abs(sum(averages)) / 3.0
    return {"accuracy": answer, "estimate": averages,
            "default_threshold": 1e-3,
            "default_requests_rescue": answer >= 1e-3,
            "diagnostic_threshold": 1e-6,
            "diagnostic_requests_rescue": answer >= 1e-6}


def comparisons(first, others, native=None):
    result = {}
    if first["status"] or any(row["status"] for row in others):
        return {"backend_failure_requests_rescue": True}
    rows = [first] + others
    for projection, component in [("real_interference", 0), ("imaginary_interference", 1)]:
        values = [[z[component] for z in row["coefficients"]] for row in rows]
        result[projection] = estimator(values)
        if native:
            actual = native.accuracy(values)
            assert abs(actual["accuracy"] - result[projection]["accuracy"]) <= 1e-15
            assert actual["estimate"] == result[projection]["estimate"]
            result[projection]["native_fortran"] = actual
    return result


def permute(lane, permutation):
    n = lane["n"]
    full = [[0.0] * n for _ in range(n)]
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            full[i][j] = full[j][i] = lane["invariants"][k][0]
            k += 1
    return dict(lane, shifts=[lane["shifts"][i] for i in permutation],
        masses=[lane["masses"][i] for i in permutation],
        invariants=[[full[permutation[i]][permutation[j]], 0.0]
                    for i in range(n) for j in range(i + 1, n)])


def rotate(lane, number):
    # No external-particle-at-rest exception is assumed for this tensor lane.
    # These are the ordinary exact axis permutations in improve_ps.inc.
    if number == 1:
        shifts = [[t, z, -x, -y] for t, x, y, z in lane["shifts"]]
        coefficients = [([a, d, b, cc], [v * (-1) ** (b + cc) for v in z])
                        for (a, b, cc, d), z in lane["coefficients"]]
    else:
        shifts = [[t, -z, y, x] for t, x, y, z in lane["shifts"]]
        coefficients = [([a, d, cc, b], [v * (-1) ** d for v in z])
                        for (a, b, cc, d), z in lane["coefficients"]]
    return dict(lane, shifts=shifts, coefficients=coefficients)


def translate_origin(lane):
    """q'=q+p0, hence N'(q')=N(q'-p0); coefficients rounded once to f64."""
    with localcontext() as context:
        context.prec = 80
        anchor = [Decimal.from_float(v) for v in lane["shifts"][0]]
        coefficients = {}
        for alpha, coefficient in lane["coefficients"]:
            for beta in product(*(range(a + 1) for a in alpha)):
                factor = Decimal(1)
                for a, b, p in zip(alpha, beta, anchor):
                    factor *= comb(a, b) * ((-p) ** (a - b) if a != b else 1)
                values = coefficients.setdefault(beta, [Decimal(0), Decimal(0)])
                for i in range(2):
                    values[i] += Decimal.from_float(coefficient[i]) * factor
        shifts = [[float(Decimal.from_float(v) - p) for v, p in zip(shift, anchor)]
                  for shift in lane["shifts"]]
    return dict(lane, shifts=shifts,
                coefficients=[(alpha, list(map(float, values))) for alpha, values in coefficients.items()])


class Adapter:
    def __init__(self, path, helpers):
        self.lib = c.CDLL(str(Path(path).resolve()))
        self.h = helpers
        self.lib.tirust_stability_contract.argtypes = [c.c_int, c.c_void_p, c.c_void_p,
            c.c_double, c.c_void_p, c.c_void_p]
        self.lib.tirust_stability_contract.restype = c.c_int
        self.lib.tirust_host_accuracy.argtypes = [c.c_void_p, c.c_int, c.c_void_p, c.c_void_p]
        self.lib.tirust_host_accuracy.restype = c.c_int
        self.lib.tirust_stability_clear.argtypes = []
        self.lib.tirust_stability_finalize.argtypes = []
        sys.path.insert(0, str(Path.cwd()))
        from madgraph.various.q_polynomial import FortranPolynomialRoutines
        self.positions = {alpha: i for i, alpha in
                          enumerate(FortranPolynomialRoutines(5).get_COLLIER_mapping())}

    def clear(self):
        self.lib.tirust_stability_clear()

    def accuracy(self, values):
        flat = [v for row in values for v in row]
        data = (c.c_double * 12)(*flat)
        accuracy = c.c_double()
        estimate = (c.c_double * 3)()
        assert self.lib.tirust_host_accuracy(data, len(values), c.byref(accuracy), estimate) == 0
        return {"accuracy": accuracy.value, "estimate": list(estimate)}

    def evaluate(self, lane, mode):
        shifts = lane["shifts"]
        pl = [[shifts[(i + 1) % 4][k] - shifts[i][k] for k in range(4)] for i in range(4)]
        numerator = [0j] * 126
        for alpha, coefficient in lane["coefficients"]:
            numerator[self.positions[tuple(alpha)]] += complex(*coefficient) * (-1) ** sum(alpha)
        output = (self.h.Complex * 3)()
        status = self.lib.tirust_stability_contract(mode,
            (c.c_double * 16)(*(v for p in pl for v in p)),
            (self.h.Complex * 4)(*(self.h.Complex(*z) for z in lane["masses"])), lane["mu2"],
            (self.h.Complex * 126)(*(self.h.Complex(z.real, z.imag) for z in numerator)), output)
        row = {"ct_mode": mode, "status": status, "pl": pl,
               "pl_sum": [sum(p[k] for p in pl) for k in range(4)]}
        if not status:
            values = [complex(z.re, z.im) for z in output]
            row["coefficients"] = [[z.real, z.imag] for z in values]
            truth = [complex(*z) for z in lane["truth"]]
            row["reference_relative_error"] = max(abs(a - b) for a, b in zip(values, truth)) / max(map(abs, truth))
        return row

    def close(self):
        self.lib.tirust_stability_finalize()


class Bridge:
    def __init__(self, path, helpers):
        self.lib = c.CDLL(str(Path(path).resolve()))
        self.h = helpers
        self.lib.ml_tirust_context_create.argtypes = [c.c_uint32, c.POINTER(c.c_void_p)]
        for name in ["ml_tirust_context_destroy", "ml_tirust_clear_cache"]:
            getattr(self.lib, name).argtypes = [c.c_void_p]
        self.lib.ml_tirust_cache_statistics.argtypes = [c.c_void_p, c.c_void_p, c.c_size_t]
        self.lib.ml_tirust_last_error.argtypes = [c.c_void_p, c.c_size_t]
        self.lib.ml_tirust_last_error.restype = c.c_size_t
        self.lib.ml_tirust_evaluate_dp.argtypes = [c.c_void_p, c.c_int32, c.c_int32,
            *([c.c_void_p, c.c_size_t] * 3), c.c_double, c.c_void_p, c.c_size_t,
            *([c.c_int32] * 4)]
        self.lib.ml_tirust_evaluate_mp.argtypes = [c.c_void_p, c.c_int32, c.c_int32,
            *([c.c_void_p, c.c_size_t] * 5), *([c.c_int32] * 4)]
        self.context = c.c_void_p()
        assert self.lib.ml_tirust_context_create(1, c.byref(self.context)) == 0

    def clear(self):
        assert self.lib.ml_tirust_clear_cache(self.context) == 0

    def counts(self):
        out = (c.c_uint64 * 9)()
        assert self.lib.ml_tirust_cache_statistics(self.context, out, 9) == 0
        return list(out)

    def evaluate(self, lane, precision, check=False, cache=True):
        n, rank = lane["n"], lane["rank"]
        basis = list(self.h.basis(rank))
        positions = {a: i for i, a in enumerate(basis)}
        nc = len(basis)
        p = [v for shift in lane["shifts"] for v in shift]
        s = [0.0] * (n * n)
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                s[i + n * j] = s[j + n * i] = lane["invariants"][k][0]
                k += 1
        before = self.counts()
        if precision == "dp":
            out = (self.h.Complex * (4 * nc))()
            status = self.lib.ml_tirust_evaluate_dp(self.context, n, rank,
                (c.c_double * len(p))(*p), len(p), (c.c_double * len(s))(*s), len(s),
                (self.h.Complex * n)(*(self.h.Complex(*z) for z in lane["masses"])), n,
                lane["mu2"], out, len(out), 0, 1, check, cache)
            values = [complex(z.re, z.im) for z in out]
            slots = [0j] * 4
            if status == 0:
                for alpha, coefficient in lane["coefficients"]:
                    coefficient = complex(*coefficient) * (-1) ** sum(alpha)
                    i = positions[tuple(alpha)]
                    for slot in range(4):
                        slots[slot] += coefficient * values[slot * nc + i]
                coefficients = [slots[3], slots[0] + slots[1], slots[2]]
        else:
            pp, ss, mm, scale = (self.h.records(v) for v in
                (p, s, [x for mass in lane["masses"] for x in mass], [lane["mu2"]]))
            out = c.create_string_buffer(8 * nc * 64)
            status = self.lib.ml_tirust_evaluate_mp(self.context, n, rank,
                pp, len(pp), ss, len(ss), mm, len(mm), scale, len(scale),
                out, len(out), 0, 1, check, cache)
            if status == 0:
                with localcontext() as ctx:
                    ctx.prec = 80
                    values = [Decimal(out.raw[i:i + 64].strip().decode()) for i in range(0, len(out), 64)]
                    slots = [[Decimal(0), Decimal(0)] for _ in range(4)]
                    for alpha, coefficient in lane["coefficients"]:
                        cr, ci = [Decimal.from_float(v) * (-1) ** sum(alpha) for v in coefficient]
                        i = positions[tuple(alpha)]
                        for slot in range(4):
                            re, im = values[2 * (slot * nc + i):2 * (slot * nc + i) + 2]
                            slots[slot][0] += cr * re - ci * im
                            slots[slot][1] += cr * im + ci * re
                    coefficients = [complex(*map(float, slots[3])),
                        complex(float(slots[0][0] + slots[1][0]), float(slots[0][1] + slots[1][1])),
                        complex(*map(float, slots[2]))]
        row = {"precision": precision, "check_pass": check, "cache": cache, "status": status,
               "cache_before": before}
        if status:
            message = c.create_string_buffer(4096)
            self.lib.ml_tirust_last_error(message, len(message))
            row["message"] = message.value.decode()
        else:
            truth = [complex(*z) for z in lane["truth"]]
            row["coefficients"] = [[z.real, z.imag] for z in coefficients]
            row["reference_relative_error"] = max(abs(a - b) for a, b in zip(coefficients, truth)) / max(map(abs, truth))
            row["symmetric_reference_error"] = [abs(a - b) / abs(a + b) if a + b else 0.0
                                                for a, b in zip(coefficients, truth)]
        row["cache_after"] = self.counts()
        return row

    def close(self):
        assert self.lib.ml_tirust_context_destroy(self.context) == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=Path(".tirust-build/bridge-qualification/libmadloop_tirust.so"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=Path(".tirust-build/stability-audit/libtirust_stability_audit.so"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new evidence filename")
    root = Path.cwd()
    spec = importlib.util.spec_from_file_location("tirust_fixture_helpers", root / "vendor/tirust/check_fixtures.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    source = root.parent / "references/TIRust/crates/iregi-rust/tests/fixtures/gate_passed_lanes.json"
    lane = next(v for v in json.loads(source.read_text())["lanes"] if v["label"] == LABEL)
    bridge = Bridge(args.library, helpers)
    report = {"label": LABEL, "fixture_sha256": digest(source), "library_sha256": digest(args.library),
        "script_sha256": digest(__file__), "truth": lane["truth"], "cases": {},
        "qp_contraction": "80-digit Decimal contraction of MPFR tensor transport, rounded to binary64 for report",
        "limitations": ["Synthetic complex tensor integral, not a physical squared matrix element.",
            "Re and Im projections are separately fed to the exact algebraic MadLoop real estimator.",
            "Raw supplied invariants retained; generated-adapter reconstruction is a separate diagnostic."]}
    try:
        cases = [("original_origin", lane), ("zero_shift_first", permute(lane, [3, 0, 1, 2]))]
        for name, primary in cases:
            report["cases"][name] = result = {}
            reversed_lane = permute(primary, [0, 3, 2, 1])
            for precision in ("dp", "qp"):
                bridge.clear()
                first = bridge.evaluate(primary, precision)
                result[precision] = rows = {"primary": first}
                rows["repeat_cached"] = bridge.evaluate(primary, precision)
                rows["reverse_check_bypass"] = bridge.evaluate(reversed_lane, precision, check=True)
                rows["reverse_cache_off"] = bridge.evaluate(reversed_lane, precision, cache=False)
                rows["reverse_shared_cache"] = bridge.evaluate(reversed_lane, precision)
                bridge.clear()
                rows["rotation_1"] = bridge.evaluate(rotate(primary, 1), precision, cache=False)
                rows["rotation_2"] = bridge.evaluate(rotate(primary, 2), precision, cache=False)
                rows["host_reversal_estimate"] = comparisons(first, [rows["reverse_check_bypass"]])
                rows["host_reversal_plus_rotations_estimate"] = comparisons(first,
                    [rows["reverse_check_bypass"], rows["rotation_1"], rows["rotation_2"]])
                print(name, precision, json.dumps(rows["host_reversal_estimate"]), flush=True)
    finally:
        bridge.close()
    adapter = Adapter(args.adapter, helpers)
    report["adapter_library_sha256"] = digest(args.adapter)
    try:
        # Also verify the independent Python formula against compiled production
        # Fortran for every raw diagnostic pair and four-evaluation comparison.
        for result in report["cases"].values():
            for rows in result.values():
                rows["host_reversal_estimate"] = comparisons(rows["primary"],
                    [rows["reverse_check_bypass"]], adapter)
                rows["host_reversal_plus_rotations_estimate"] = comparisons(rows["primary"],
                    [rows["reverse_check_bypass"], rows["rotation_1"], rows["rotation_2"]], adapter)
        normalized = translate_origin(permute(lane, [3, 0, 1, 2]))
        report["generated_adapter"] = result = {
            "input": normalized,
            "input_policy": "Near-zero last shift permuted first; q'=q+p0 and N'(q')=N(q'-p0). "
                "Polynomial translation uses Decimal80 then rounds coefficients once to binary64. "
                "PL uses binary64 cyclic shift differences. QP promotes these same binary64 PL/numerator inputs; "
                "the adapter reconstructs invariants and contracts the numerator in its working precision.",
            "normalization": "Unnormalized tensor diagnostic; no R2, counterterms, or physical Born amplitude."}
        for precision, mode in [("dp", 1), ("qp", 4)]:
            adapter.clear()
            result[precision] = rows = {"primary": adapter.evaluate(normalized, mode)}
            rows["reverse_check_bypass"] = adapter.evaluate(normalized, mode + 1)
            adapter.clear()
            rows["rotation_1"] = adapter.evaluate(rotate(normalized, 1), mode)
            adapter.clear()
            rows["rotation_2"] = adapter.evaluate(rotate(normalized, 2), mode)
            rows["host_reversal_estimate"] = comparisons(rows["primary"], [rows["reverse_check_bypass"]], adapter)
            rows["host_reversal_plus_rotations_estimate"] = comparisons(rows["primary"],
                [rows["reverse_check_bypass"], rows["rotation_1"], rows["rotation_2"]], adapter)
            print("generated_adapter", precision, json.dumps(rows["host_reversal_estimate"]), flush=True)
        for name in ("primary", "reverse_check_bypass", "rotation_1", "rotation_2"):
            dp, qp = result["dp"][name], result["qp"][name]
            if dp["status"] == qp["status"] == 0:
                pairs = [(complex(*a), complex(*b)) for a, b in zip(dp["coefficients"], qp["coefficients"])]
                dp["symmetric_error_against_same_input_qp"] = [abs(a-b)/abs(a+b) if a+b else 0.0 for a,b in pairs]
    finally:
        adapter.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
