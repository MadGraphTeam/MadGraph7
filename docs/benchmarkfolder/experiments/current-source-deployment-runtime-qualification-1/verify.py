"""Replay retained runtime evidence only; no native loading or external paths."""
import ast
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import posixpath

HERE = Path(__file__).resolve().parent
BUILD_SEAL = "5c5ae6bc61d99b4312f6e69296e44672e6907b40351a4a9bf01a1e8ac2bb1b70"
ADMISSION_SEAL = "6e1539f8e5004ff43c65688b7cf72c97ae7e0371a62ee86506809e84041d70f3"
RESULT_HASH = "14af24e03d891ea8f682e07b8e9dffc628dfde168f154b598cbea4682ab3ee7d"
EXPECTED = [("standalone", 1, 1), ("standalone", 8, 1), ("standalone", 9, 1),
            ("standalone", 9, 2), ("standalone", 9, 4), ("standalone", 9, 5),
            ("nlo", 1, 1), ("nlo", 9, 1), ("nlo", 9, 4)]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def sealed_packet(root, expected=None):
    index_path = root / "evidence-index.json"
    if expected:
        require(digest(index_path) == expected, "Dependency seal changed")
    index = read(index_path)
    actual = {str(p.relative_to(root)) for p in root.rglob("*")
              if p.is_file() and p.name != "evidence-index.json"}
    require(actual == set(index["files"]), "Packet file set changed")
    for name, checksum in index["files"].items():
        path = root / name
        require(path.resolve().is_relative_to(root) and not path.is_symlink(), "Packet path escape")
        require(digest(path) == checksum, "Packet bytes changed: " + name)
    return index


def card_values(path):
    lines = path.read_text().splitlines()
    values = {}
    for i, line in enumerate(lines):
        if line.startswith("#"):
            key = line[1:].strip()
            require(key not in values and i + 1 < len(lines), "Malformed card")
            values[key] = lines[i + 1].strip().lower()
    return values


def main():
    index = sealed_packet(HERE)
    summary = read(HERE / "qualification-summary.json")
    dependency = summary["build_packet"]
    require(dependency["relative_directory"] == "../current-source-deployment-qualification-1",
            "Unexpected build dependency")
    build = HERE.parent / "current-source-deployment-qualification-1"
    sealed_packet(build, BUILD_SEAL)
    admitted = read(build / "build/build-admitted.json")
    require(digest(build / "build/build-admitted.json") == dependency["build_admitted_sha256"],
            "Original build receipt changed")
    require(digest(HERE / "admission/source-index.json") == ADMISSION_SEAL, "Admission seal changed")
    source = read(HERE / "admission/source-index.json")
    for name, checksum in source["files"].items():
        require(digest(HERE / "admission" / name) == checksum, "Admission source changed")
    supplement = read(HERE / "admission/interpreter-supplement.json")
    for name, checksum in supplement["original_bindings"].items():
        path = build / ("build/" + name if name == "build-admitted.json" else name)
        require(digest(path) == checksum, "Additive supplement original binding")
    providers = dict(admitted["runtime_providers"])
    extra = supplement["additional_runtime_provider"]
    require(extra["path"] == extra["resolved"] and extra["path"] not in providers,
            "Supplement is not exactly additive")
    providers[extra["path"]] = extra["sha256"]
    for label, executable in supplement["executables"].items():
        require(executable["sha256"] == admitted["exports"][label]["executable_sha256"] and
                executable["pt_interp"]["path"] == extra["path"], "Executable/interpreter binding")

    # Only these three already-sealed pure helpers are compiled. No module imports,
    # native helpers, main function, absolute path lookups or subprocess code run.
    original = (build / "proposal/smoke.py").read_text()
    selected = [n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef)
                and n.name in {"number", "parse", "compare"}]
    require(len(selected) == 3, "Pure replay function set changed")
    namespace = {"Decimal": Decimal, "InvalidOperation": InvalidOperation, "require": require}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "retained-pure-parser", "exec"), namespace)

    def retained(name):
        entry = summary["runtime_files"][name]
        path = HERE / entry["path"]
        require(path.resolve().is_relative_to(HERE) and digest(path) == entry["sha256"],
                "Runtime file binding changed")
        return path

    for name in summary["runtime_files"]:
        retained(name)
    result_path = retained("runtime-result.json")
    require(digest(result_path) == RESULT_HASH == summary["runtime_result_sha256"], "Runtime result changed")
    result = read(result_path)
    require(result["complete"] and result["lock_acquired"] and result["protected_files_unchanged"],
            "Runtime did not complete cleanly")
    rows = result["records"]
    require(len(rows) == 10 and [(r["export"], r["backend"], r["mode"]) for r in rows[:9]] == EXPECTED,
            "Actual worker order differs from admitted modes")
    aliases = read(build / "provider-spellings.json")["aliases"]
    references, initialized, residuals = {}, set(), []
    for i, row in enumerate(rows):
        label = row["label"]
        require(row["cleanup_complete"] and not row["timeout"], "Unclean worker")
        require(row["cap_seconds"] <= (120 if i < 9 else 60), "Worker limit widened")
        execution = read(retained("results/" + label + "-execution.json"))
        require(all(row.get(k) == v for k, v in execution.items()), "Execution/aggregate mismatch")
        raw = retained(row["raw_result"])
        stdout = retained(row["stdout"])
        require(digest(raw) == row["raw_result_sha256"] and digest(stdout) == row["stdout_sha256"],
                "Worker raw/log hash mismatch")
        require(len(row["loader_logs"]) == 1, "Unexpected loader log count")
        log_name, checksum = next(iter(row["loader_logs"].items()))
        require(log_name == "logs/loader-" + label + "." + str(row["pid"]), "Wrong loader PID")
        log = retained(log_name)
        require(digest(log) == checksum == row["loader"]["log_sha256"], "Loader log hash mismatch")
        observed = {}
        for line in log.read_text().splitlines():
            if "calling init:" not in line:
                continue
            spelling = posixpath.normpath(line.split("calling init:", 1)[1].strip())
            require(spelling.startswith("/"), "Unresolved loader path")
            resolved = aliases[spelling]["resolved_path"] if spelling in aliases else spelling
            require(resolved in providers, "Unbound initialized provider")
            observed[resolved] = providers[resolved]
        require(len(observed) == 12 and observed == row["loader"]["initialized_providers"],
                "Initialized provider evidence mismatch")
        export_label = row["export"] if i < 9 else "nlo"
        dso = str(Path(admitted["exports"][export_label]["process"]).parents[
            1 if export_label == "standalone" else 2] / "lib/libmadloop_tirust.so")
        require(dso in observed and observed[dso] == summary["native_library_sha256"] and
                extra["path"] in observed, "Relocated DSO or actual interpreter not initialized")
        initialized.update(observed)
        card = card_values(retained("results/" + label + "-card.dat"))
        require(card["LoopPoleMode"] == "0" and card["TIRustGlobalCache"] == "-1" and
                card["TIRustCacheMaxBytes"] == "268435456" and
                card["UseQPIntegrandForTIRust"] == ".true.", "Pole/cache/QP card settings changed")
        if i < 9:
            require(row["returncode"] == 0 and card["MLReductionLib"] == str(row["backend"]) and
                    card["CTModeRun"] == card["CTModeInit"] == str(row["mode"]) and
                    card["CheckCycle"] == "1" and card["TIRustComputeFinite"] == ".true.", "Fixed card/status")
            parsed = namespace["parse"](raw.read_text(), row["mode"])
            require(parsed == row["result"], "Raw numerical replay changed")
            if export_label not in references:
                require(row["backend"] == 1, "Own-export CutTools control missing")
                references[export_label] = parsed
            comparison = namespace["compare"](references[export_label], parsed)
            require(comparison == row["comparison"] and
                    all(x["regression_passed"] and x["ordinary_target_passed"] for x in comparison.values()),
                    "Own-export numerical comparison failed")
            residuals.extend(Decimal(x["symmetric"]) for x in comparison.values())
        else:
            require(label == "nlo-partial" and row["returncode"] == 1 and row["partial_guard_passed"] and
                    card["MLReductionLib"] == "9" and card["CTModeRun"] == card["CTModeInit"] == "1" and
                    card["TIRustComputeFinite"] == ".false." and raw.read_bytes() == b"" and
                    "Poles-only tensors are not complete amplitudes" in stdout.read_text(), "NLO partial guard")
    require(len(initialized) == 13 and str(max(residuals)) == summary["maximum_symmetric_residual"],
            "Derived runtime summary mismatch")
    execution = read(retained("runtime-controller-execution.json"))
    require(execution["terminal_returncode"] == 0 and execution["native_groups_idle"] and
            execution["common_lock_probed_free"] and execution["elapsed_seconds"] == 8.900 and
            execution["elapsed_seconds"] < 900 and execution["runtime_result_sha256"] == RESULT_HASH,
            "Whole-controller execution receipt")
    require((HERE / "runtime/controller-terminal.txt").read_text() == "CONTROLLER_ELAPSED_SECONDS=8.900\n",
            "Captured timer output changed")
    require(len(execution["groups"]) == 10 and all(x["cleanup_complete"] and not x["group_alive_now"]
                                                  for x in execution["groups"]), "Recorded cleanup not idle")
    for group, row in zip(execution["groups"], rows, strict=True):
        require(all(group[key] == row[key] for key in ["label", "pid", "returncode", "cleanup_complete"]),
                "Cleanup/result worker identity mismatch")
    post = read(HERE / "runtime/postflight.json")
    require(post["private_source_prefixes_absent"], "Copied source prefixes remain")
    require(set(post["cards"]) == {"standalone", "nlo"}, "Missing restored card")
    for label, card in post["cards"].items():
        require(card["matches_original_build_card"] and card["path"] == admitted["exports"][label]["runtime_card"] and
                card["sha256"] == admitted["files"][card["path"]], "Runtime card was not restored")
    print(json.dumps({"retained_runtime_replay_passed": True, "retained_files": len(index["files"]),
                      "fixed_numerical_workers": 9, "partial_guard_workers": 1,
                      "maximum_symmetric_residual": str(max(residuals)), "all_1e8_passed": True,
                      "initialized_provider_paths": len(initialized), "external_files_rehashed": 0,
                      "native_processes_launched": 0, "performance_or_stability_qualification": False}, indent=2))


if __name__ == "__main__":
    main()
