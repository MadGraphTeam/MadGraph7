#!/usr/bin/env python3
"""Read-only, exact-arithmetic verification of the final dependency replay.

This verifies newly observed 256/384-bit convergence. Historical binary128
identity is reported separately and is not an acceptance requirement. No
numerical library, compiler, subprocess, or output writer is invoked.
"""
import argparse
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction as Q
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/numerical-controls-v1/manifest-v2.json"
SCHEMA = "final-release-numerical-capsule-v1"
NAMES = ("finite.re", "finite.im", "single.re", "single.im", "double.re", "double.im", "rational.re", "rational.im")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def show(value):
    with localcontext() as context:
        context.prec = 100
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def rounded113(value):
    """Nearest-even IEEE binary128 value as an exact rational (finite inputs)."""
    if not value:
        return value
    sign = -1 if value < 0 else 1
    value = abs(value)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < Q(2) ** exponent:
        exponent -= 1
    require(exponent <= 16383, "Binary128 overflow in retained output")
    quantum = Q(2) ** max(exponent - 112, -16494)
    quotient, remainder = divmod(*(value / quantum).as_integer_ratio())
    upward = 2 * remainder > (value / quantum).denominator or (
        2 * remainder == (value / quantum).denominator and quotient % 2)
    return sign * (quotient + bool(upward)) * quantum


def ascii113(text):
    return rounded113(Q(text)), text.lstrip().startswith("-")


def scaled(error, scale):
    require(bool(scale) or not error, "Nonzero difference at zero Laurent scale")
    return error / scale if scale else Q(0)


def observation_values(observation, bits):
    require(observation["status"] == 0 and observation["working_bits"] == bits,
            "Replay did not successfully evaluate at the required precision")
    require(observation["input_unchanged"] is True and observation["output_bytes"] == 512,
            "Replay input mutation or incomplete output commit")
    require(len(observation["raw"]) == 10 and len(observation["output"]) == 8,
            "Replay output dimensions differ")
    require(all(item["finite"] is True for item in observation["raw"]), "Nonfinite replay result")
    values = [Q(int(item["numerator"]), int(item["denominator"])) for item in observation["raw"]]
    require(all((item["negative"] is True) == (value < 0) for item, value in zip(observation["raw"], values) if value),
            "Raw replay signs differ from exact values")
    return values


def recompute(original, capture):
    """Verify the complete ordered stream and return independently derived facts."""
    require(len(original) == 1392 and len(capture) == 2785, "Incomplete final1392 replay stream")
    maxima = dict.fromkeys(("laurent", "CC", "R1", "split", "old_laurent", "old_R1", "ascii_rounding"), Q(0))
    changed = ascii_changed = checked_fields = 0
    components, shapes, all_shapes, ascii_rounding = Counter(), Counter(), Counter(), Counter()
    physical_bins, rational_bins = Counter(), Counter()
    minimum_precision = {256: 100000, 384: 100000}
    for index, (old, low, high) in enumerate(zip(original, capture[:1392], capture[1392:2784])):
        require(low["kind"] == "replay" and high["kind"] == "convergence", "Replay record kinds/order differ")
        require(low["sequence"] == high["sequence"] == index and low["input"] == high["input"] == old,
                "Replay input/order changed; filtering or sorting is forbidden")
        observations = (low["observed"], high["observed_384"])
        a, b = (observation_values(observed, bits) for observed, bits in zip(observations, (256, 384)))
        scale = max(map(abs, b[:6]))
        for name, indices in (("laurent", range(6)), ("CC", (6, 7)), ("R1", (8, 9))):
            error = scaled(max(abs(a[i] - b[i]) for i in indices), scale)
            require(error <= Q(1, 10**31), "New-source256/384 convergence failed for " + name)
            maxima[name] = max(maxima[name], error)
        split = scaled(max(abs(v[i] - v[6+i] - v[8+i]) for v in (a, b) for i in (0, 1)), scale)
        require(split <= Q(1, 10**60), "Complete finite is not CC + R1")
        maxima["split"] = max(maxima["split"], split)
        require([v["negative"] for v in observations[0]["raw"]] == [v["negative"] for v in observations[1]["raw"]],
                "Convergence changed a result sign")
        require(high["comparison"]["passed"] is True and high["comparison"]["sign_changed_indices"] == [],
                "Recorded convergence conclusion disagrees")
        for bits, observed, exact in zip((256, 384), observations, (a, b)):
            minimum_precision[bits] = min(minimum_precision[bits], *(item["precision"] for item in observed["raw"]))
            output_scale = max(map(abs, exact[:6]))
            for field, (text, raw_index) in enumerate(zip(observed["output"], (0, 1, 2, 3, 4, 5, 8, 9))):
                actual, sign = ascii113(text)
                require(sign == observed["raw"][raw_index]["negative"], "ASCII output sign differs")
                error = scaled(abs(actual - rounded113(exact[raw_index])), output_scale)
                require(error <= Q(1, 10**31), "ASCII output does not represent the raw Float result")
                checked_fields += 1
                if error:
                    ascii_rounding[f"{bits}/{NAMES[field]}"] += 1
                    maxima["ascii_rounding"] = max(maxima["ascii_rounding"], error)
        previous = list(map(ascii113, old["output"]))
        current = list(map(ascii113, observations[0]["output"]))
        different = [i for i, (x, y) in enumerate(zip(previous, current)) if x != y]
        require(low["historical_binary128_values_and_zero_signs_equal"] == (not different),
                "Historical binary128 classification disagrees")
        ascii_different = old["output"] != observations[0]["output"]
        require(low["historical_ascii_equal"] == (not ascii_different), "Historical ASCII classification disagrees")
        changed += bool(different)
        ascii_changed += ascii_different
        components.update(NAMES[i] for i in different)
        shape = f"n{old['n']}-r{old['rank']}"
        all_shapes[shape] += 1
        if different:
            shapes[shape] += 1
        deltas = [abs(x[0] - y[0]) for x, y in zip(previous, current)]
        physical_scale = max(abs(v[0]) for v in previous[:6] + current[:6])
        physical_error = scaled(max(deltas[:6]), physical_scale)
        rational_error = scaled(max(deltas[6:]), physical_scale)
        maxima["old_laurent"] = max(maxima["old_laurent"], physical_error)
        maxima["old_R1"] = max(maxima["old_R1"], rational_error)
        for exponent in (31, 20, 10, 7):
            label = f"1e-{exponent}"
            physical_bins[label] += physical_error > Q(1, 10**exponent)
            rational_bins[label] += rational_error > Q(1, 10**exponent)
    declared = capture[-1]
    require(declared["failed_binary128_replay_records"] == changed and
            declared["historical_ascii_changed_records_retained"] == ascii_changed and
            declared["failed_convergence_records"] == 0 and
            declared["sign_changed_convergence_records_require_review"] == 0,
            "Replay footer disagrees with the complete raw observations")
    return {"records_each_precision": 1392, "raw_records": 2785, "status_failures": 0,
            "failed_convergence_records": 0, "sign_changed_convergence_records": 0,
            "maximum_scaled_errors": {key: show(value) for key, value in maxima.items()},
            "minimum_tracked_precision_observed": {str(key): value for key, value in minimum_precision.items()},
            "historical_binary128_changed_records": changed, "historical_ascii_changed_records": ascii_changed,
            "historical_changed_component_counts": dict(components), "historical_changed_topology_rank_counts": dict(shapes),
            "all_topology_rank_counts": dict(all_shapes), "historical_complete_laurent_records_above_scaled_difference": dict(physical_bins),
            "historical_basis_rational_records_above_scaled_difference": dict(rational_bins),
            "ASCII_output_fields_examined": checked_fields, "ASCII_adjacent113bit_rounding_fields": dict(ascii_rounding),
            "historical_values_are_oracle": False, "historical_bit_identity_passed": changed == 0,
            "final_source_convergence_passed": True, "all1392_inputs_in_original_order_without_filtering": True}


def read_bound(record, inputs=None):
    path = ROOT / record["path"]
    require(path.resolve().is_relative_to(ROOT), "Evidence escaped repository")
    payload = path.read_bytes()
    require(hashlib.sha256(payload).hexdigest() == record["sha256"] and len(payload) == record["bytes"],
            "Changed numerical evidence: " + record["path"])
    if inputs is not None:
        inputs[record["path"]] = record["sha256"]
    if record.get("compression") == "gzip":
        payload = gzip.decompress(payload)
        require(hashlib.sha256(payload).hexdigest() == record["decoded_sha256"] and len(payload) == record["decoded_bytes"],
                "Changed decompressed numerical evidence: " + record["path"])
    return payload


def relocation_delta(original, relocated, replaced, label, sealed=False):
    """A metadata relocation cannot rewrite an observation or its scope."""
    actual = dict(relocated)
    for key in ("original_metadata", "publication_relocation", "publication_original_prepared", "publication_original_result"):
        actual.pop(key, None)
    expected = dict(original)
    if sealed:
        actual.pop("identity_sha256", None)
        expected.pop("identity_sha256", None)
    require(all(key in expected and key in actual for key in replaced), "Missing relocation field in " + label)
    for key in replaced:
        actual[key] = expected[key]
    require(actual == expected, "Metadata relocation changed numerical, runtime, source, or scope fields in " + label)


def verify_relocation(manifest, payloads, inputs=None):
    """Bind the portable views to the original successful actual execution."""
    def load(record):
        return json.loads(read_bound(record, inputs))

    original_unit = json.loads(payloads["unit-proof.json"])
    original_build = json.loads(payloads["build-fenv-record.json"])
    unit = load(manifest["portable_unit_proof"])
    build = load(manifest["portable_build_record"])
    require(unit["original_metadata"] == manifest["artifacts"]["unit-proof.json"] and
            build["original_metadata"] == manifest["artifacts"]["build-fenv-record.json"],
            "Portable unit/build view lost its original immutable metadata")
    relocation_delta(original_unit, unit, ("stdout", "stderr", "build_record"), "ordinary unit proof")
    relocation_delta(original_build, build, ("build_record", "build_stdout", "build_stderr", "retained_source_directory"), "Rust build proof")
    require(unit["build_record"] == manifest["portable_build_record"], "Portable unit proof refers to another build")
    for field, artifact in (("stdout", "ordinary-fenv.stdout"), ("stderr", "ordinary-fenv.stderr")):
        require(unit[field] == manifest["artifacts"][artifact] and
                hashlib.sha256(read_bound(unit[field], inputs)).hexdigest() == original_unit[field]["sha256"],
                "Portable ordinary log changed")
    for field, artifact in (("build_record", "build-fenv-process.json"), ("build_stdout", "build-fenv.stdout"), ("build_stderr", "build-fenv.stderr")):
        require(build[field] == manifest["artifacts"][artifact] and
                hashlib.sha256(read_bound(build[field], inputs)).hexdigest() == original_build[field]["sha256"],
                "Portable build log changed")
    require(build["retained_source_directory"] == str((ROOT / manifest["artifacts"]["source-v4/Cargo.toml"]["path"]).parent.relative_to(ROOT)),
            "Portable retained source directory differs from the complete bound snapshot")
    result = load(manifest["installed_precision_portable"])
    original_result = load(result["publication_original_result"])
    prepared = load(result["prepared"])
    original_prepared = load(prepared["publication_original_prepared"])
    relocation_delta(original_result, result, ("prepared",), "actual installed64 result", sealed=True)
    relocation_delta(original_prepared, prepared, ("unit_proof",), "actual installed64 preparation", sealed=True)
    require(original_result["prepared"] == prepared["publication_original_prepared"] and
            prepared["unit_proof"] == manifest["portable_unit_proof"],
            "Portable actual64 result/preparation chain differs from the original")
    original_installed_unit = load(original_prepared["unit_proof"])
    require(original_installed_unit == original_unit, "Actual64 original preparation used another ordinary suite")
    require(load(prepared["installation"]) == load(manifest["installation"]),
            "Portable actual64 proof and numerical capsule use different installations")
    # Verify the retained installed observation's sealed identities, independent
    # of whether the caller separately invokes precision.validate().
    for record in (result, original_result, prepared, original_prepared):
        content = {key: value for key, value in record.items() if key != "identity_sha256"}
        digest = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        require(digest == record["identity_sha256"], "Changed installed64 sealed metadata identity")
    for field in ("stdout", "stderr", "loaded_libraries"):
        read_bound(result[field], inputs)
    return True


def verify(manifest_path=DEFAULT, installation=None, inputs=None):
    """Verify portable raw authority; optionally bind to a concrete installation."""
    path = Path(manifest_path)
    if not path.is_absolute():
        path = ROOT / path
    manifest = json.loads(path.read_text())
    require(manifest["schema"] == SCHEMA, "Unknown final numerical evidence schema")
    require(manifest["verifier_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "Numerical verifier differs from the retained authority")
    if inputs is not None:
        inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        inputs[str(Path(__file__).resolve().relative_to(ROOT))] = manifest["verifier_sha256"]
    payloads = {name: read_bound(record, inputs) for name, record in manifest["artifacts"].items()}
    relocation_verified = verify_relocation(manifest, payloads, inputs)
    sources = manifest["bridge_source_sha256"]
    require(len(sources) == 45, "Incomplete tested source inventory")
    for name, checksum in sources.items():
        require(hashlib.sha256(payloads["source-v4/" + name]).hexdigest() == checksum, "Tested source snapshot changed")
    observed_install = json.loads(read_bound(manifest["installation"], inputs))
    require(observed_install["archived_bridge_source_sha256"] == sources, "Installed archive and tested source differ")
    if installation is not None:
        actual = installation if isinstance(installation, dict) else json.loads(Path(installation).read_text())
        require(actual["archived_bridge_source_sha256"] == sources and actual["files"] == observed_install["files"],
                "Requested installation is not the tested numerical archive/interface")
    process = json.loads(payloads["replay-convergence-final-process.json"])
    build = json.loads(payloads["build-fenv-record.json"])
    equivalence = json.loads(payloads["committed-source-equivalence.json"])
    require(build["bridge_source_sha256"] == sources and process["executable"] == build["executable"] and
            build["features"] == process["features"] == ["fenv-test-support"],
            "Replay executable and source-bound fenv build differ")
    require(equivalence["committed_revision"] == manifest["revision"] and
            equivalence["complete_source45_sha256"] == sources and
            equivalence["every_tested_snapshot_file_matches_git_object"] is True and
            equivalence["observations_relabelled"] is False,
            "Original dirty observation and committed-source equivalence differ")
    require(process["bridge_source_sha256"] == process["source_sha256_after"] == sources and
            process["source_sha256_before_equals_after"] is True and process["all_children_exited"] is True,
            "Replay lacks an unchanged source and terminal process binding")
    require(process["exit_code"] == 101, "Original historical bit-identity failure was relabelled")
    require(process["stdout"]["sha256"] == hashlib.sha256(payloads["replay-convergence-final.stdout"]).hexdigest() and
            process["stderr"]["sha256"] == hashlib.sha256(payloads["replay-convergence-final.stderr"]).hexdigest(),
            "Replay process/log binding differs")
    original = [json.loads(line) for line in payloads["original-replay.jsonl"].splitlines()]
    capture = [json.loads(line) for line in payloads["replay-convergence-final.jsonl"].splitlines()]
    result = recompute(original, capture)
    require(result == manifest["expected_recomputed"], "Published classification differs from exact raw replay")
    classification = json.loads(payloads["replay-exact-classification.json"])
    require(classification["input"]["sha256"] == hashlib.sha256(payloads["original-replay.jsonl"]).hexdigest() and
            classification["capture"]["sha256"] == hashlib.sha256(payloads["replay-convergence-final.jsonl"]).hexdigest(),
            "Original exact classification used different raw data")
    tail = json.loads(payloads["planar-sentinels-complete-process.json"])
    require(tail["exit_code"] == 101, "Known provider regression failure was relabelled")
    comparisons = [json.loads(line.split(" ", 1)[1]) for line in payloads["planar-sentinels-complete.stdout"].decode().splitlines()
                   if line.startswith("NINJARS_UPSTREAM_PLANAR_COMPARISON ")]
    require(len(comparisons) == 4 and sum(not row["accepted_or_rejected_safely"] for row in comparisons) == 2,
            "Known provider failures were omitted or relabelled")
    result.update(schema=SCHEMA, status="passed", revision=manifest["revision"], bridge_source_sha256=sources,
                  archive_sha256=observed_install["files"]["lib/libninjars.a"], original_rust_replay_exit_code=101,
                  historical_bit_identity_test_passed=False, known_provider_regression_test_passed=False,
                  metadata_only_relocation_verified=relocation_verified,
                  known_provider_unsafe_successes=2, known_provider_findings_scope="Direct scalar provider; the six retained full-bridge calls are independently accurate or safely rejected.",
                  numerical_scope="Focused bridge convergence only; complete MadLoop correctness and performance gates are separate.",
                  performance_claim=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", type=Path, default=DEFAULT)
    args = parser.parse_args()
    print(json.dumps(verify(args.manifest), indent=2))


if __name__ == "__main__":
    main()
