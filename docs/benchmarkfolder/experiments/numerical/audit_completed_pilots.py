#!/usr/bin/env python3
"""Independently audit retained numerical pilots; never execute native code.

Writes only completed-pilots-index.json beside this script. Frozen historical
protocols and recorded executable identities are used, never the current driver.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
REPORT = HERE.parents[1]
LABELS = ("qcd-n1", "qcd-n2", "qcd-n3", "ew-n1", "ew-n2", "mixed", "li-h", "li-hh")
NAMES = {1: "CutTools", 6: "Ninja", 7: "COLLIER", 8: "NinjaRS", 9: "TIRust"}
STATUS = {(2, 1): (1, "DP"), (3, 9): (1, "QP"), (2, 6): (6, "DP"),
          (3, 8): (6, "QP"), (2, 7): (7, "DP"), (2, 8): (8, "DP"),
          (3, 2): (8, "QP"), (2, 9): (9, "DP"), (3, 3): (9, "QP")}
FILES, ERRORS = {}, []


def require(condition, message):
    if not condition:
        ERRORS.append(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def read(path):
    data = path.read_bytes()
    FILES[path.relative_to(REPORT).as_posix()] = {"sha256": sha(data), "bytes": len(data)}
    return data


def load(path):
    return json.loads(read(path))


def parse_observation(directory, backend):
    raw = directory / "raw"
    ndim, nloop, nborn = map(int, read(raw / "dimensions.dat").split())
    require(ndim == max(nloop, nborn) and min(nloop, nborn) >= 0, f"{directory.name}: dimensions")
    rows, counts = [], Counter()
    for line in read(raw / "values.dat").decode().splitlines():
        fields = line.split()
        require(len(fields) == 8, f"{directory.name}: malformed value row")
        point, order, code = map(int, fields[:3])
        actual = STATUS.get((code // 100, code % 10))
        require(actual is not None and actual[0] == backend and code // 10 % 10 in (1, 2),
                f"{directory.name}: unexpected status {code}")
        values = list(map(float, fields[3:]))
        require(all(map(math.isfinite, values)) and values[-1] >= 0, f"{directory.name}: invalid values/accuracy")
        rows.append(dict(point=point, order=order, status=code, coefficients=values[:4],
                         accuracy=values[-1], actual_backend=actual[0], precision=actual[1]))
        if order == 0:
            counts[str(code)] += 1
    require([(r["point"], r["order"]) for r in rows] ==
            [(p, o) for p in range(1, 101) for o in range(ndim + 1)], f"{directory.name}: point/order coverage")
    for point in range(1, 101):
        require(len({r["status"] for r in rows if r["point"] == point}) == 1, f"{directory.name}: split status mismatch")
    timed = {}
    for line in read(raw / "timed_status.dat").decode().splitlines():
        code, number = map(int, line.split())
        require(str(code) not in timed and number > 0, f"{directory.name}: invalid histogram")
        timed[str(code)] = number
    require(timed == dict(counts) and sum(timed.values()) == 100, f"{directory.name}: histogram differs from 100 raw evaluations")
    usage = Counter()
    for code, number in counts.items():
        actual = STATUS[(int(code) // 100, int(code) % 10)]
        usage[f"{actual[0]}:{actual[1]}"] += number
    observation = load(raw / "observation.json")
    require(observation["rows"] == rows, f"{directory.name}: parsed rows differ")
    require(observation["status_counts"] == dict(counts) == observation["timed_status_counts"],
            f"{directory.name}: retained status counts differ")
    require(observation["backend_precision_counts"] == dict(usage), f"{directory.name}: retained precision counts differ")
    require(observation["native_dp_only"] == (dict(usage) == {f"{backend}:DP": 100}), f"{directory.name}: DP-only flag")
    require(observation["answer_dimensions"] == dict(allocated=ndim, loop_orders=nloop, born_orders=nborn),
            f"{directory.name}: retained dimensions differ")
    # Retain timing bytes for integrity, but do not calculate a runtime result.
    read(raw / "timing.dat")
    return rows, observation, counts, usage


def compare(reference, candidate, backend, recorded):
    require([(r["point"], r["order"]) for r in reference] == [(r["point"], r["order"]) for r in candidate],
            "comparison has different point/order keys")
    residuals, misses = [], []
    maximum_raw = maximum_relevant = maximum_finite = 0.0
    for left, right in zip(reference, candidate):
        finite_scale = max(abs(left["coefficients"][1]), abs(right["coefficients"][1]), 1e-99)
        for column, (a, b) in enumerate(zip(left["coefficients"], right["coefficients"])):
            magnitude = max(abs(a), abs(b))
            symmetric = abs(a-b)/abs(a+b) if a+b else (0.0 if magnitude == 0 else None)
            raw = abs(a-b)/magnitude if magnitude else 0.0
            fraction = magnitude/finite_scale if column >= 2 else None
            def accepted(threshold):
                return ((symmetric is not None and symmetric < threshold)
                        or (fraction is not None and fraction < threshold))
            residuals.append(dict(point=left["point"], order=left["order"], column=column,
                reference=a, candidate=b, relative_error=raw, symmetric_relative_error=symmetric,
                pole_to_finite=fraction, passed=accepted(3e-6)))
            if not accepted(1e-8):
                misses.append(dict(backend=backend, point=left["point"], order=left["order"], column=column,
                                   symmetric_relative_error=symmetric, pole_to_finite=fraction))
            maximum_raw = max(maximum_raw, raw)
            if column < 2 or fraction >= 3e-6:
                require(symmetric is not None, "undefined non-negligible comparison")
                maximum_relevant = max(maximum_relevant, symmetric if symmetric is not None else math.inf)
            if column == 1:
                maximum_finite = max(maximum_finite, symmetric if symmetric is not None else math.inf)
    passed = all(row["passed"] for row in residuals)
    require(recorded["residuals"] == residuals, "stored comparator residuals differ from independent recomputation")
    require(recorded["passed"] == passed and recorded["ordinary_target_passed"] == (not misses), "stored comparator pass flags differ")
    require(recorded["maximum_relative_error"] == maximum_raw and recorded["tolerance"] == 3e-6
            and recorded["ordinary_target_tolerance"] == 1e-8, "stored comparator maxima/tolerances differ")
    require(recorded["failures"] == [row for row in residuals if not row["passed"]], "stored comparator failures differ")
    return dict(regression_pass=passed, ordinary_target_pass=not misses,
                max_raw_relative_error=maximum_raw, max_symmetric_non_negligible_residual=maximum_relevant,
                max_finite_symmetric_residual=maximum_finite, ordinary_target_miss_count=len(misses)), misses


def audit_inputs(folder, source, backend, case, config, protocol_sha, baseline_source):
    inputs = load(folder / f"{source}-inputs.json")
    generation, compilation, settings = inputs["generation"], inputs["compilation"], inputs["settings"]
    require(generation["source"] == source and generation["case"] == case, f"{folder}: generation case/source")
    require(settings["mlreductionlib"] == str(backend) and settings["mlstabthres"] == 1e-6, f"{folder}: strict explicit backend settings")
    expected = dict(config["madloop_parameters"])
    if source == "candidate":
        expected.update(config["candidate_parameters"])
    require(all(settings.get(key.lower()) == value for key, value in expected.items()), f"{folder}: policy parameters")
    points = REPORT / "points" / (case["id"] + ".dat")
    point_bytes = read(points)
    require(sha(point_bytes) == inputs["points_sha256"] and int(point_bytes.splitlines()[0]) == 100,
            f"{folder}: point identity/count")
    require(len(point_bytes.splitlines()) == 1 + 100*(case["nfinal"]+2), f"{folder}: point stream length")
    card_sha = sha(read(REPORT / "inputs" / source / case["id"] / "param_card.dat"))
    require(card_sha == inputs["param_card_sha256"] == generation["parameter_card_sha256"], f"{folder}: physical parameter card identity")
    command_sha = sha(read(REPORT / "commands" / f"{source}-{case['id']}.mg5"))
    require(command_sha == generation["command_sha256"], f"{folder}: generation command identity")
    identity = generation["source_identity"]
    if source == "candidate":
        require(canonical(identity["files"]) == identity["files_sha256"], f"{folder}: source inventory digest")
        snapshot = load(REPORT / "inputs/source-identities" / (canonical(identity) + ".json"))
        require(snapshot == identity, f"{folder}: retained generator snapshot")
    else:
        require(identity["revision"] == baseline_source["revision"] == "9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f"
                and identity["files_sha256"] == canonical(baseline_source["files"])
                and identity["archive_sha256"] == baseline_source["archive_sha256"], f"{folder}: baseline source identity")
    require(compilation["numerical_build_from_sources"] is True and compilation["full_numerical_rebuild_this_invocation"] is True,
            f"{folder}: complete numerical source rebuild")
    require(compilation["worker_template_sha256"] == protocol_sha, f"{folder}: frozen worker template binding")
    rendered = (folder.parent / "tirust_benchmark_worker.f").read_text()
    for key, token in (("prefix", "PREFIX"), ("nexternal", "NEXTERNAL"), ("nborn", "NBORN")):
        rendered = rendered.replace(f"@{token}@", str(compilation["worker_render_parameters"][key]))
    require(sha(rendered.encode()) == compilation["driver_sha256"], f"{folder}: rendered compiled driver")
    driver_hashes = [value for name, value in compilation["sources"].items()
                     if re.fullmatch(r"SubProcesses/P[^/]+/check_sa\.f", name)]
    require(driver_hashes == [compilation["driver_sha256"]], f"{folder}: driver generated-source inventory")
    for name, text in compilation["settings"].items():
        require(sha(text.encode()) == compilation["sources"][name], f"{folder}: makefile source identity")
    require(compilation["link_inputs_before"] == compilation["link_inputs_after"], f"{folder}: link dependencies changed")
    for kind in ("archives", "metadata"):
        require(inputs["runtime"][kind] == compilation["link_inputs_after"][kind], f"{folder}: runtime link-input binding")
    require(re.fullmatch(r"[0-9a-f]{64}", compilation["executable_sha256"]) is not None, f"{folder}: missing executable identity")
    return inputs, dict(generator_revision=identity["revision"], generator_files_sha256=identity["files_sha256"],
        generator_dirty=identity["dirty"], executable_sha256=compilation["executable_sha256"],
        driver_sha256=compilation["driver_sha256"], generated_sources_sha256=canonical(compilation["sources"]),
        compiler_policy_sha256=canonical(compilation["effective_compiler_flags"]),
        numerical_settings_sha256=canonical(settings), runtime_libraries_sha256=canonical(inputs["runtime"]["libraries"]))


def main():
    read(Path(__file__))
    baseline_source = load(REPORT / "inputs/baseline.json")
    cases = []
    for suffix in LABELS:
        folder = HERE / ("strict-policy-" + suffix)
        # Hash every retained file, including raw timings that are not qualified.
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                require("__pycache__" not in path.parts and path.suffix != ".pyc", f"unexpected generated file: {path}")
                read(path)
        specification, original = load(folder / "specification.json"), load(folder / "results.json")
        config = load(folder / "cases.json")
        case = specification["case"]
        require(original["case"] == case and specification["performance_eligible"] is False
                and original["performance_eligible"] is False, f"{suffix}: scope")
        require(config["points_per_case"] == 100 and config["relative_tolerance"] == 3e-6, f"{suffix}: sampling/truth settings")
        configuration_sha = sha(read(folder / "cases.json"))
        if "configuration_sha256" in specification:
            require(configuration_sha == specification["configuration_sha256"], f"{suffix}: config digest")
        for name, checksum in specification["protocols"].items():
            require(sha(read(folder / name)) == checksum, f"{suffix}: frozen protocol {name}")
        expected = [("baseline", 1), *(("candidate", b) for b in NAMES)]
        require([(r["source"], r["backend"]) for r in original["results"]] == expected, f"{suffix}: explicit backend coverage")
        summaries, identities, target_misses, reference, frozen_inputs = [], {}, [], None, {}
        for (source, backend), top_result in zip(expected, original["results"]):
            run = folder / f"{source}-{backend}"
            require(not (run / "failure.json").exists(), f"{suffix}/{run.name}: failure retained in completed run")
            result = load(run / "result.json")
            require(result == top_result and result["status"] == "complete" and result["performance_eligible"] is False,
                    f"{suffix}/{run.name}: result scope/status/duplicate")
            rows, observation, status_counts, usage = parse_observation(run, backend)
            require(observation == result["observation"], f"{suffix}/{run.name}: duplicate observation")
            inputs, identity = audit_inputs(run, source, backend, case, config,
                specification["protocols"]["tirust_benchmark_worker.f"], baseline_source)
            if source in frozen_inputs:
                require(inputs["generation"] == frozen_inputs[source]["generation"]
                        and inputs["compilation"] == frozen_inputs[source]["compilation"], f"{suffix}: backend run source/build mismatch")
            frozen_inputs[source] = inputs
            identities[run.name] = identity
            if reference is None:
                reference = rows
            compared, misses = compare(reference, rows, backend, result["comparison"])
            row = dict(backend=backend, name=NAMES[backend], points=100, rows=len(rows),
                dp_count=usage.get(f"{backend}:DP", 0), qp_count=usage.get(f"{backend}:QP", 0),
                status_counts=dict(status_counts), **compared)
            if source == "baseline":
                baseline_summary = row
            else:
                summaries.append(row)
                target_misses.extend(misses)
        candidate, baseline = frozen_inputs["candidate"], frozen_inputs["baseline"]
        require(candidate["points_sha256"] == baseline["points_sha256"]
                and candidate["param_card_sha256"] == baseline["param_card_sha256"], f"{suffix}: source-to-source physical inputs differ")
        cases.append(dict(id=case["id"], process=case["process"], model=case["model"], nfinal=case["nfinal"],
            label=folder.name, points=100, candidate_backends=list(NAMES), baseline=baseline_summary, backends=summaries,
            all_five_regression_pass=all(row["regression_pass"] for row in summaries),
            all_five_ordinary_target_pass=all(row["ordinary_target_pass"] for row in summaries),
            ordinary_target_misses=target_misses,
            max_symmetric_non_negligible_residual=max(row["max_symmetric_non_negligible_residual"] for row in summaries),
            max_finite_symmetric_residual=max(row["max_finite_symmetric_residual"] for row in summaries),
            identities=dict(points_sha256=candidate["points_sha256"], param_card_sha256=candidate["param_card_sha256"],
                baseline_revision=baseline["generation"]["source_identity"]["revision"],
                candidate_revision=candidate["generation"]["source_identity"]["revision"],
                worker_template_sha256=specification["protocols"]["tirust_benchmark_worker.f"],
                configuration_sha256=configuration_sha,
                configuration_sha256_recorded_at_run="configuration_sha256" in specification, runs=identities),
            qualifiers=["Numerical pilot only; timings excluded from performance acceptance.",
                        "Historical frozen worker, source and executable identities; no assertion about the current driver.",
                        "DP/QP counts describe accepted point evaluations, not internal reduction attempts."] +
                       (["The earliest pilot did not record a configuration digest in specification.json; this audit hashes its retained cases.json and verifies every recorded effective parameter against it."] if "configuration_sha256" not in specification else []) +
                       (["QCD n3 retains the raw CutTools-reference ordinary-target misses; separate adjudication does not overwrite these comparisons."] if suffix == "qcd-n3" else []) +
                       (["Near-zero loop-induced poles may have large raw relative ratios; the frozen pole-to-finite exception is applied explicitly."] if suffix.startswith("li-") else [])))
    result = dict(schema="madloop-completed-numerical-pilots-v1", scope="Independent audit of eight historical physics pilots",
        performance_eligible=False, audit_passed=not ERRORS, audit_errors=ERRORS, cases=cases,
        comparison_definition={"regression_tolerance": 3e-6, "ordinary_target": 1e-8,
            "rule": "abs(a-b)/abs(a+b)<tolerance OR (pole column AND max(abs(a),abs(b))/max(abs(finite_a),abs(finite_b),1e-99)<tolerance)",
            "max_symmetric_non_negligible_residual": "Maximum symmetric ratio excluding poles accepted by the 3e-6 pole-size exception; Born and finite columns are always included.",
            "zero_zero": 0, "nonzero_opposite_sign_symmetric_ratio": None},
        totals=dict(cases=len(cases), runs=6*len(cases), candidate_evaluations=500*len(cases),
                    baseline_evaluations=100*len(cases), split_order_rows=sum(c["baseline"]["rows"]+
                    sum(r["rows"] for r in c["backends"]) for c in cases)),
        reproducibility="python3 docs/benchmarkfolder/experiments/numerical/audit_completed_pilots.py",
        files_base="docs/benchmarkfolder", files=FILES)
    target = HERE / "completed-pilots-index.json"
    target.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"audit_passed": result["audit_passed"], "errors": ERRORS, "totals": result["totals"],
                      "indexed_files": len(FILES), "index_sha256": sha(target.read_bytes())}, indent=2))
    if ERRORS:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
