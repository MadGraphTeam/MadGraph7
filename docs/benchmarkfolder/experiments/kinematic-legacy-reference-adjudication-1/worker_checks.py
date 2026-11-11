"""Pure retained-worker checks for the closed legacy-reference adjudication.

Callbacks provide authenticated packet bytes; no recorded absolute path is
opened. Selected sealed comparison functions run with a byte-only virtual
Path and a coefficient-comparison sentinel. Their original filesystem loaders,
native controllers, and materialized-card reads are never executed.
"""

import ast
import hashlib
import json
import re
from types import SimpleNamespace


SOURCE = "0a0dba9baf12ff5a988e4d459141a48140f285a396658ee10f7f09099848dbd8"
CLOSURE = "b94dca1a42b7acd2c835ce6b1bf04fe9036b0d64755ec1348240235a76b06462"
BUILD = "08a5b485954e20dc25fcffc2c26edddd73520b982e9daa380e1d5eca06709f22"
OLD_SOURCE = "ccbb3dff88f673043251dac5d5099d184f709fdffdba1e5f2c3c1dc0aca37a74"
OLD_COMPARE = "ecce1e1f99f0de7d1798880cc9137251b4e553accc708f3cee357ed794b99272"
COMPARE = "34277061d0a367e156f39519e2cad973adebaf89e9ee4c1c794313238f93cba4"
USAGE = "b4fb17b7d7efba6acbfe261e62bf13a885b41c9b648487e1e5fc28e55457f1ce"
ROOT = "/shared/external/madloop_tirust/madgraph7_madloop_tirust/"
WORK = ".tirust-build/kinematic-invariant-reference-adjudication-executor-1/"
PROPOSAL = ".tirust-build/kinematic-invariant-reference-adjudication-proposal-1/"
EXE = {"bytes": 87020816, "sha256": "34d32dc8fa47e00d600e471df0d75d63eafcb0120cecde31099cc809bfed6373"}
OUTPUTS = ("bootstrap-values.dat", "driver-state.dat", "driver-bits.dat", "qualification.dat",
           "dimensions.dat", "values.dat", "timed_status.dat", "timing.dat", "trace.dat")
DECIMAL_TOKEN = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?\Z")
BINDINGS = {
    "ordinary_comparison": ("results/ordinary-comparison.json", "6d4f37f0b9b0cc0b3297d31b7167d176db640bbca21cd33f8861a0e4b9750451"),
    "point20_comparison": ("results/root-point20-comparison.json", "7b2390b98193dd11c66ae8eae64ec7b36f5497accd9e2e355de03ccfacda5d6d"),
    "physical_card": ("proposal/inputs/cms-param_card.dat", "96158fedb398f3a7957bff10db0a7101281eb57ef0e9973e9c6e43c727efb8ec"),
    "point1": ("proposal/inputs/cms-point1.dat", "ed760e231007f76fa0f6b80aad72be2291a5e5cef8343e14c69e40dd6a50233b"),
    "point20": ("proposal/inputs/cms-point20.dat", "57d7f677e925d7bc258e65fe440880d2876f0317de3904c348060e08c48edd1d"),
}
CARD_SHAS = {
    6: "2c6e8317324ae0d0194b5c8b1840545c64adc0732658abdd23adc06c399a6d2d",
    8: "5dad1b91bdb0ebf81b82f1b42561e45387adf85e7cfa10ed44cd6311499d810e",
}


def require(condition, message):
    if not condition:
        raise ValueError("Reference worker evidence: " + message)


def pin(raw):
    require(isinstance(raw, bytes), "callback did not return bytes")
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def checked(data, name, expected):
    raw = data(name)
    actual = pin(raw)
    wanted = {key: expected[key] for key in ("bytes", "sha256")} if isinstance(expected, dict) else expected
    require(actual == wanted if isinstance(wanted, dict) else actual["sha256"] == wanted,
            "retained payload differs: " + name)
    return raw


def safe(name):
    require(isinstance(name, str) and name and not name.startswith("/") and "\\" not in name
            and all(part not in ("", ".", "..") for part in name.split("/")), "unsafe logical path")
    return name


def selected(raw, names, constants=()):
    """Extract only exact definitions/literal constants, never imports/top level."""
    tree = ast.parse(raw)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    require(len(functions) == len(names) and {node.name for node in functions} == set(names),
            "pure function selection differs")
    for node in functions:
        require(not node.decorator_list, "unexpected decorator")
        for item in [*node.args.defaults, *[v for v in node.args.kw_defaults if v is not None]]:
            ast.literal_eval(item)
    literals = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in constants:
                literals[node.targets[0].id] = ast.literal_eval(node.value)
    require(set(literals) == set(constants), "selected constant missing")
    return ast.Module(body=functions, type_ignores=[]), literals


def card_value(raw, name):
    lines = raw.decode("ascii").splitlines()
    found = [index for index, line in enumerate(lines) if line.strip() == "#" + name]
    require(len(found) == 1, "missing/duplicate card option: " + name)
    return lines[found[0] + 1].split("!", 1)[0].strip().upper()


def verify_maps(raw, capture, expected):
    require(pin(raw)["sha256"] == capture["sha256"] and capture["elf"] == expected,
            "raw map/provider receipt differs")
    extra = capture["data"]
    for name, identity in extra.items():
        require(name.startswith("/nix/store/") and name.endswith("/lib/locale/locale-archive")
                and type(identity["bytes"]) is int and identity["bytes"] > 0
                and re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]), "unexpected non-ELF data mapping")
    observed, inodes = set(), {}
    for line in raw.decode("ascii").splitlines():
        fields = line.split(None, 5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        name = fields[5]
        require(not name.endswith(" (deleted)") and int(fields[4]) > 0
                and (name in expected or name in extra), "unadmitted/deleted raw map path")
        if name in extra:
            require(fields[1] == "r--p", "locale mapping permissions differ")
        observed.add(name)
        inodes.setdefault(name, set()).add((fields[3], fields[4]))
    require(observed == set(expected) | set(extra) and all(len(ids) == 1 for ids in inodes.values()),
            "raw provider-map coverage differs")


def verify_workers(data, document, parent_data, parent_document):
    """Verify explicit closed bytes; the caller authenticates the c04 parent.

    Own logical paths are source/* and results/*. Parent paths use the c04
    capsule layout. Build-artifact and overall process-closure joins belong to
    the enclosing verifier. No external/materialized worker files are read.
    """
    def doc(name, expected=None):
        raw = data(name) if expected is None else checked(data, name, expected)
        value = json.loads(raw)
        require(value == document(name), "own document/byte callback differs")
        return value

    def parent_doc(name, expected=None):
        raw = parent_data(name) if expected is None else checked(parent_data, name, expected)
        value = json.loads(raw)
        require(value == parent_document(name), "parent document/byte callback differs")
        return value

    source = doc("source/index.json", SOURCE)
    require(source["payload_count"] == len(source["files"]) == 29, "source inventory differs")

    def source_data(name):
        safe(name)
        require(name in source["files"], "unsealed source payload")
        return checked(data, "source/" + name, source["files"][name])

    closure = doc("results/run-closure.json", CLOSURE)
    failure = "ValueError('Reference eligibility/agreement failed; evidence retained')"
    require(closure["source_index_sha256"] == SOURCE and closure["reviewed_build_sha256"] == BUILD
            and closure["stage"] == "run" and closure["passed"] is False and closure["failure"] == failure
            and closure["postcheck_failure"] is None and closure["owned_command_count"] == 4
            and closure["loader_query_count"] == 2 and closure["host_stability_qualified"] is False
            and closure["performance_measurement"] is False and len(closure["payloads"]) == 43,
            "closed run/source/build scope differs")
    allowed = {"results/" + safe(name): identity for name, identity in closure["payloads"].items()}

    def retained(name):
        safe(name)
        require(name in allowed, "non-closed worker read: " + name)
        return checked(data, name, allowed[name])

    def retained_doc(name):
        raw = retained(name)
        value = json.loads(raw)
        require(value == document(name), "closed document/byte callback differs")
        return value

    plan = json.loads(source_data("validation-plan.json"))
    requests = plan["requests"]
    account = retained_doc("results/run-requests.json")
    require(account == {"requests": requests, "completed": [item["id"] for item in requests],
                       "failure": failure, "not_completed": [], "maximum_outer_entries": 4,
                       "performance_measurement": False}, "request closure accounting differs")
    bindings = json.loads(source_data("bindings.json"))
    require(bindings["root"] == ROOT.rstrip("/"), "recorded source root differs")
    old_index = parent_doc("proposal/index.json", OLD_SOURCE)
    old_source = checked(parent_data, "proposal/amplitude_compare.py", OLD_COMPARE)
    require(pin(old_source) == old_index["files"]["amplitude_compare.py"], "old comparator source member differs")
    reference_source = source_data("reference_compare.py")
    require(pin(reference_source)["sha256"] == COMPARE, "reference comparison source changed")
    usage_source = source_data("usage.py")
    require(pin(usage_source)["sha256"] == USAGE, "usage parser source changed")

    class RetainedPath:
        def __init__(self, value):
            self.logical = safe(value.logical if isinstance(value, RetainedPath) else value)

        def __truediv__(self, other):
            return RetainedPath(self.logical + "/" + safe(other))

        def __str__(self):
            return self.logical

        @property
        def name(self):
            return self.logical.rsplit("/", 1)[-1]

        def read_bytes(self):
            return retained(self.logical)

    usage_tree, _ = selected(usage_source, {"parse"})
    usage_ns = {}
    exec(compile(usage_tree, "sealed-reference-usage", "exec"), usage_ns)
    usage_module = SimpleNamespace(parse=usage_ns["parse"])
    legacy_anchor = "        require(not raw['trace.dat'],'Unexpected diagnostic trace output')"
    require(old_source.decode().count(legacy_anchor) == 1, "legacy usage adaptation anchor differs")
    adapted = old_source.decode().replace(legacy_anchor,
        "        # REF usage stream checked by outer reference gate.", 1)
    legacy_tree, legacy_constants = selected(adapted, {"require", "pin", "_states", "_coverage", "_decimal_shape", "inspect_worker"},
                                             {"OUTPUTS"})
    require(legacy_constants["OUTPUTS"] == OUTPUTS, "legacy output scope differs")
    legacy_ns = {"hashlib": hashlib, "json": json, "re": re, "Path": RetainedPath,
                 "OUTPUTS": OUTPUTS, "DECIMAL_TOKEN": DECIMAL_TOKEN}
    exec(compile(legacy_tree, "sealed-reference-structural-parsers", "exec"), legacy_ns)
    arithmetic_calls = []

    def forbid_coefficients(*args, **kwargs):
        arithmetic_calls.append(True)
        raise RuntimeError("This closed failed-reference replay must never compare coefficients")

    legacy_module = SimpleNamespace(**{name: legacy_ns[name] for name in
        ("_states", "_coverage", "inspect_worker", "DECIMAL_TOKEN")}, compare_coefficients=forbid_coefficients)

    def binding(key, sha):
        require(key in BINDINGS and BINDINGS[key][1] == sha, "unadmitted saved binding")
        logical = BINDINGS[key][0]
        item = bindings["files"][key]
        raw = checked(parent_data, logical, item)
        require(pin(raw)["sha256"] == sha, "saved binding source differs")
        expected_prefix = ".tirust-build/kinematic-invariant-amplitude-"
        require(item["path"].startswith(expected_prefix), "saved binding is not historical amplitude input")
        return raw, {"path": item["path"], **pin(raw)}

    def cards_and_points(directory, request):
        raw = (directory / "specification.json").read_bytes()
        spec = json.loads(raw)
        require(spec["request"] == request and spec["source_index_sha256"] == SOURCE,
                "reference source/request differs")
        numerical_raw = source_data(request["card"])
        require(pin(numerical_raw)["sha256"] == CARD_SHAS[request["backend"]], "reference card changed")
        expected = {}
        for key in ("physical_card", "point1", "point20"):
            old_raw, record = binding(key, BINDINGS[key][1])
            own_name = {"physical_card": "inputs/cms-param_card.dat", "point1": "inputs/cms-point1.dat",
                        "point20": "inputs/cms-point20.dat"}[key]
            require(source_data(own_name) == old_raw, "copied physical input differs")
            expected[key] = pin(old_raw)
        require(spec["numerical_card"] == pin(numerical_raw) and spec["physical_card"] == expected["physical_card"]
                and spec["bootstrap_pin"] == expected["point1"] and spec["point_pin"] == expected["point20"],
                "retained input specification differs")
        return {"specification.json": pin(raw), "numerical_card": pin(numerical_raw), **expected}

    reference_tree, constants = selected(reference_source,
        {"require", "pin", "checked", "_structural_evidence", "inspect_reference_worker", "load_saved",
         "_phase_view", "validate_requests", "compare_reference_stage"},
        {"OLD_COMPARE_SHA", "USAGE_SHA", "SAVED_SHAS", "INPUT_SHAS", "CARD_SHAS"})
    require(constants == {"OLD_COMPARE_SHA": OLD_COMPARE, "USAGE_SHA": USAGE,
                "SAVED_SHAS": {key: BINDINGS[key][1] for key in ("ordinary_comparison", "point20_comparison")},
                "INPUT_SHAS": {key: BINDINGS[key][1] for key in ("physical_card", "point1", "point20")},
                "CARD_SHAS": CARD_SHAS}, "reference comparison constants differ")
    reference_ns = {"hashlib": hashlib, "json": json, "Path": RetainedPath, **constants,
                    "_legacy": lambda: legacy_module, "_usage": lambda: usage_module,
                    "_binding": binding, "_cards_and_points": cards_and_points}
    exec(compile(reference_tree, "sealed-reference-comparison", "exec"), reference_ns)
    replayed = reference_ns["compare_reference_stage"]("results/runs", requests)
    recorded_raw = retained("results/reference-comparison.json")
    require(json.loads(recorded_raw) == replayed == document("results/reference-comparison.json")
            and recorded_raw == (json.dumps(replayed, sort_keys=True, indent=2) + "\n").encode("ascii"),
            "complete reference comparison does not reproduce")
    require(not arithmetic_calls and replayed["execution_complete"] is True and replayed["outer_entries_observed"] == 4
            and replayed["ready_done_maps"] == 4 and replayed["workers_requested"] == 2
            and all(replayed[key] is False for key in ("passed", "native_usage_gate_passed", "bootstrap_controls_passed",
                "reference_gate_passed", "references_agree", "same_native_input_established", "winner_selected",
                "independent_accuracy_qualified", "normal_host_stability_qualified", "performance_measurement")),
            "reference replay changed failure/qualification scope")
    require(len(replayed["bootstrap_comparisons"]) == 2 and len(replayed["measurement_comparisons"]) == 7
            and all(row["interpretation_blocked"] is True and row["eligible"] is False and row["passed"] is False
                    for row in replayed["bootstrap_comparisons"] + replayed["measurement_comparisons"]),
            "failed worker coefficients were promoted")

    baseline = parent_doc("results/builds/original/build.json")
    providers = baseline["runtime"]["files"]
    require(len(providers) == 15, "expected provider inventory differs")
    command_names, native_pids, summaries = set(), set(), []
    for position, request in enumerate(requests, 1):
        backend, identifier = request["backend"], request["id"]
        require(backend in (8, 6), "unadmitted backend")
        prefix = "results/runs/" + safe(identifier) + "/"
        spec, native = retained_doc(prefix + "specification.json"), retained_doc(prefix + "result.json")
        observation = replayed["observations"][str(backend)]
        require(observation["eligible"] is False and observation["bootstrap_control_passed"] is False
                and observation["measurement_interpretation_allowed"] is False
                and native["numerically_qualified"] is False and native["performance_measurement"] is False,
                "worker qualification differs")
        pid = native["native_pid"]
        require(type(pid) is int and pid > 0 and pid not in native_pids, "native identity missing/reused")
        native_pids.add(pid)
        cwd = ROOT + WORK + "work/workers/" + identifier + "/SubProcesses/P0_epem_mupmum"
        require(spec["cwd"] == cwd and spec["executable"] == cwd + "/check" and spec["executable_pin"] == EXE
                and spec["evidence"] == ROOT + WORK + prefix.rstrip("/")
                and spec["bootstrap_file"] == cwd + "/bootstrap-input.dat"
                and spec["point_file"] == cwd + "/measurement-input.dat", "worker path/build identity differs")
        expected_runtime = {**providers, spec["executable"]: EXE["sha256"]}
        require(spec["expected_runtime"] == expected_runtime, "worker runtime admission differs")
        for capture in native["runtime_maps"]:
            require(safe(capture["file"]) in ("maps-0.txt", "maps-1.txt"), "unexpected map file")
            verify_maps(retained(prefix + capture["file"]), capture, expected_runtime)
        require([capture["file"] for capture in native["runtime_maps"]] == ["maps-0.txt", "maps-1.txt"],
                "map capture sequence differs")
        require(set(native["outputs"]) == set(OUTPUTS), "raw worker output set differs")
        for name in OUTPUTS:
            checked(retained, prefix + name, native["outputs"][name])
        raw_rows = legacy_ns["_coverage"](retained(prefix + "driver-bits.dat").decode("ascii"), 20, 4)
        states = legacy_ns["_states"](retained(prefix + "driver-state.dat").decode("ascii"), 20, 4)
        require(states == observation["states"] and all(row["checkphase"] == 1 and row["helicity_checked"] == 0 for row in states),
                "restored raw state differs")
        require(all(row["fpe_dp"] == row["fpe_qp"] == 0 for row in states[:-1])
                and (states[-1]["fpe_dp"], states[-1]["fpe_qp"]) == ({8: (0, 1), 6: (1, 0)}[backend]),
                "software contribution flags differ")
        qualification = [line.split() for line in retained(prefix + "qualification.dat").decode("ascii").splitlines()]
        require(qualification == [["NATIVE_QUALIFICATION_PENDING"], ["RESTORED_STATE_IS_NOT_EFFECTIVE_CALL_STATE"],
            ["BOOTSTRAP_RESTORED_STATE", "1", "120", "0", "4"], ["MEASUREMENT_RESTORED_STATE", "120", "0", "4"],
            ["STATUS_ALONE_IS_NOT_NUMERICAL_VALIDATION"]], "qualification text differs")
        require(retained(prefix + "dimensions.dat").split() == [b"1", b"1", b"1"]
                and retained(prefix + "timed_status.dat").split() == [b"120", b"1"], "dimensions/status count differs")
        usage = usage_ns["parse"](retained(prefix + "trace.dat").decode("ascii"), backend)
        require(usage == observation["native_usage"] and usage["complete"] is True
                and usage["actual_dp_calls"] == 0 and usage["coefficient_accuracy_proven"] is False
                and usage["native_completed_per_phase"] == {"0": 26, "1": 26}
                and len(usage["records"]) == 104, "native usage observation differs")
        for phase in (0, 1):
            returned = [row for row in usage["records"] if row["phase"] == phase and row["step"] == 1]
            require([row["group"] for row in returned] == list(range(1, 27)), "native group coverage differs")
        failures = [(row["phase"], row["group"], row["status"]) for row in usage["nonzero_native_statuses"]]
        require(failures == ([(1, group, 1) for group in (3, 4, 8, 12, 24)] if backend == 8 else [])
                and usage["passed"] is (backend == 6) and observation["native_usage_gate_passed"] is (backend == 6),
                "native status and software gate distinction differs")
        nonfinite = [{"phase": row["phase"], "order": row["order"], "column": column, "word": word}
                     for row in raw_rows for column, word in enumerate(row["words"])
                     if ((int(word, 16) >> 52) & 2047) == 2047]
        require((not nonfinite) if backend == 8 else
                [(row["phase"], row["order"], row["column"]) for row in nonfinite] ==
                [(1, order, column) for order in (0, 1) for column in (1, 2, 3)],
                "raw nonfinite-word classification differs")
        card = source_data(request["card"])
        expected_card = {"MLReductionLib": str(backend), "CTModeRun": "4", "LoopPoleMode": "0",
                         "TIRustComputeFinite": ".TRUE.", "UseQPIntegrandForNinja": ".TRUE.",
                         "UseQPIntegrandForTIRust": ".TRUE.", "UseQPIntegrandForNinjaRS": ".TRUE.",
                         "TIRustGlobalCache": "0", "TIRustCacheMaxBytes": "268435456"}
        echoes = {}
        console = retained(prefix + "stdout.txt").decode("utf-8", errors="replace")
        for key, value in re.findall(r"^\s*>\s*(\w+)\s*=\s*(.*?)\s*$", console, re.M):
            echoes.setdefault(key, []).append(" ".join(value.split()).upper())
        for key, value in expected_card.items():
            require(card_value(card, key) == value and echoes.get(key) == ["T" if value == ".TRUE." else value],
                    "source/native card echo differs: " + key)
        warnings = [line for line in console.splitlines() if "##W03" in line]
        require(bool(warnings) is (backend == 8), "incomplete contribution warning differs")
        admission = retained_doc("results/runtime/" + identifier + ".json")
        require(admission["passed"] is True and admission["failure"] is None and admission["label"] == identifier
                and admission["loader_query_started"] is True and admission["executable"] == spec["executable"]
                and admission["executable_pin"] == EXE and admission["runtime"]["files"] == providers
                and admission["actual_interpreter"] == admission["runtime"]["loader"] == baseline["runtime"]["loader"]
                and admission["runtime"]["loader_sha256"] == baseline["runtime"]["loader_sha256"],
                "runtime query/provider evidence differs")
        query_name = "results/execution/run-query-run-" + f"{position:04d}"
        worker_name = "results/execution/run-worker-" + identifier
        for name in (query_name, worker_name):
            command_names.add(name + ".json")
            command, log = retained_doc(name + ".json"), retained(name + ".txt")
            require(command["returncode"] == 0 and command["exception"] is None and command["cleanup_exception"] is None
                    and command["cleanup"]["survivors"] == [] and command["cpu"] == 12 and command["cpu_exclusive"] is False
                    and command["log_sha256"] == pin(log)["sha256"], "owned command closure differs")
            if name == query_name:
                require(command["argv"][-3:] == [admission["actual_interpreter"], "--list", spec["executable"]]
                        and log.decode("utf-8") == admission["runtime"]["loader_listing"], "loader transcript differs")
            else:
                require(command["cwd"] == cwd and command["argv"][-4:] == [ROOT + PROPOSAL + "worker_entry.py",
                    ROOT + WORK + prefix + "specification.json", pin(retained(prefix + "specification.json"))["sha256"], SOURCE]
                    and any(item["pid"] == pid for item in command["cleanup"]["observed_owned_processes"]),
                    "worker command/specification/ownership differs")
        summaries.append({"id": identifier, "backend": backend, "requested_mode": 4, "fixed_unchecked_return": 120,
            "protocol_complete": True, "eligible": False, "native_usage_gate_passed": usage["passed"],
            "qp_api_width": 128, "qp_api_significand_digits": 113, "native_completed_per_phase": {"0": 26, "1": 26},
            "native_nonzero_statuses": usage["nonzero_native_statuses"], "recorded_dp_api_calls": 0,
            "measurement_software_flags": {"dp": states[-1]["fpe_dp"], "qp": states[-1]["fpe_qp"]},
            "nonfinite_words": nonfinite, "incomplete_contribution_warnings": warnings,
            "bootstrap_comparison_performed": False, "measurement_comparison_performed": False})
    require(command_names == {"results/" + name for name in closure["payloads"]
                             if name.startswith("execution/") and name.endswith(".json")}, "closed command scope differs")
    return {"workers": summaries, "reference_comparison_serialized_bytes_reproduced": True,
            "native_workers_completed": 2, "outer_returns_recorded": 4, "maps_verified": 4,
            "native_api_calls_completed": 104, "integer_usage_records": 208,
            "coefficient_comparisons_executed": len(arithmetic_calls), "failed_coefficient_arithmetic": False,
            "raw_fifo_transcript_available": False, "materialized_worker_inputs_rehashed": False,
            "recorded_dp_api_calls": 0, "qp_bits_describe_api_transport_only": True,
            "internal_reducer_working_precision_qualified": False, "same_native_inputs_established": False,
            "independent_accuracy_qualified": False, "host_stability_qualified": False,
            "performance_measurement": False, "new_native_executions": 0, "external_binary_reads": 0}
