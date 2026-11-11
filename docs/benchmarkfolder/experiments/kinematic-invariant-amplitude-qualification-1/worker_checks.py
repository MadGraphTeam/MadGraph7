"""Pure checks of explicitly closed fixed-mode amplitude workers.

All bytes come from supplied retained-payload callbacks. No filesystem/native
operation or retained controller is executed. The enclosing verifier seals the
packet, sibling dependencies and supplied stage closures before this call.
FIFO messages survive as parsed records only; driver states/bits, stdout and
maps are retained raw and are checked separately below. Unsuccessful output
words are never converted into numerical amplitude coefficients.
"""

import ast
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import re


SOURCE_SEAL = "ccbb3dff88f673043251dac5d5099d184f709fdffdba1e5f2c3c1dc0aca37a74"
BUILD_SEAL = "8cab371e9affd5297c0f48c30184361959d20f3d3a48c6dbd6109b86bf49bacb"
ORDINARY_SEAL = "b364bd8535d0f5850cb5a21390c5678cbb1c1dd8fddfae6f0174cf50d261f7a7"
STRESS_SEAL = "c696421eb4948b4d2de5993cc84455b1ac04d0f659b245341729714f91ae8482"
POINT20_COMPARISON_SHA = "7b2390b98193dd11c66ae8eae64ec7b36f5497accd9e2e355de03ccfacda5d6d"
COMPARATOR_SHA = "ecce1e1f99f0de7d1798880cc9137251b4e553accc708f3cee357ed794b99272"
ROOT = "/shared/external/madloop_tirust/madgraph7_madloop_tirust/"
WORK = ".tirust-build/kinematic-invariant-amplitude-executor-2/"
PROPOSAL = ".tirust-build/kinematic-invariant-amplitude-proposal-2/"
SUBPROCESS = "SubProcesses/P0_epem_mupmum/"
OUTPUTS = ("bootstrap-values.dat", "driver-state.dat", "driver-bits.dat", "qualification.dat",
           "dimensions.dat", "values.dat", "timed_status.dat", "timing.dat", "trace.dat")
EQUAL_FILES = ("bootstrap-values.dat", "driver-state.dat", "driver-bits.dat", "qualification.dat",
               "dimensions.dat", "values.dat", "timed_status.dat")
SOURCE_PINS = {
    "tools/tirust_benchmark.py": {"bytes": 96661, "sha256": "bc405b44d8d794cc4396ef3da3d48af238ac4dfb619c58901d7bab910a7e1370"},
    "tests/parallel_tests/loop_me_comparator.py": {"bytes": 74750, "sha256": "b6da773dadbe1e202639a865cbe3f29f65af8147a250a8bb05e756d537548354"},
    ".tirust-build/kinematic-provenance-current-executor-1/source/parity.py": {
        "bytes": 8270, "sha256": "a9c91ea520280e8094e404ac88aa44fa0adebf39dd042905cc9a97080c41ff40"},
}


def require(condition, message):
    if not condition:
        raise ValueError("Amplitude worker evidence: " + message)


def pin(raw):
    require(isinstance(raw, bytes), "payload callback did not return bytes")
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def checked(data, name, expected):
    raw = data(name)
    actual = pin(raw)
    wanted = {key: expected[key] for key in ("bytes", "sha256")} if isinstance(expected, dict) else expected
    require(actual == wanted if isinstance(wanted, dict) else actual["sha256"] == wanted,
            "retained bytes differ: " + name)
    return raw


def document_checked(data, document, name, expected=None):
    raw = data(name) if expected is None else checked(data, name, expected)
    result = json.loads(raw)
    require(result == document(name), "byte/document callback mismatch: " + name)
    return result


def safe_name(name):
    require(isinstance(name, str) and name and not name.startswith("/") and "\\" not in name
            and all(part not in ("", ".", "..") for part in name.split("/")), "unsafe logical path")
    return name


def comparator_namespace(raw, read_closed):
    """Select ten pinned definitions and substitute a byte-only virtual Path."""
    require(pin(raw)["sha256"] == COMPARATOR_SHA, "comparison implementation changed")
    tree = ast.parse(raw)
    wanted = {"require", "pin", "decode64", "wire", "_states", "_coverage", "_decimal_shape",
              "inspect_worker", "compare_coefficients", "compare_stage"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    require(len(functions) == len(wanted) and {node.name for node in functions} == wanted,
            "comparison function selection differs")
    for node in functions:
        require(not node.decorator_list, "unexpected function decorator")
        for value in [*node.args.defaults, *[item for item in node.args.kw_defaults if item is not None]]:
            ast.literal_eval(value)
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"SOURCE_PINS", "OUTPUTS", "EQUAL_FILES", "COLUMNS"}:
                constants[name] = ast.literal_eval(node.value)
    require(constants == {"SOURCE_PINS": SOURCE_PINS, "OUTPUTS": OUTPUTS,
                          "EQUAL_FILES": EQUAL_FILES,
                          "COLUMNS": ("Born", "finite", "single_pole", "double_pole")},
            "comparison constants/source provenance differ")

    class RetainedPath:
        """No stat, resolve, open, directory discovery or external path support."""
        def __init__(self, value):
            self.logical = safe_name(value.logical if isinstance(value, RetainedPath) else value)

        def __truediv__(self, other):
            return RetainedPath(self.logical + "/" + safe_name(other))

        @property
        def name(self):
            return self.logical.rsplit("/", 1)[-1]

        def read_bytes(self):
            return read_closed(self.logical)

    namespace = {"Decimal": Decimal, "localcontext": localcontext, "F": Fraction,
                 "hashlib": hashlib, "json": json, "re": re, "Path": RetainedPath,
                 "SOURCE_PINS": SOURCE_PINS, "OUTPUTS": OUTPUTS, "EQUAL_FILES": EQUAL_FILES,
                 "COLUMNS": constants["COLUMNS"], "TOL": Fraction(3, 1000000),
                 "TARGET": Fraction(1, 100000000),
                 "DECIMAL_TOKEN": re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?\Z"),
                 "verify_sources": lambda: SOURCE_PINS}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "sealed-retained-amplitude-comparison", "exec"), namespace)
    return namespace


def card_value(raw, name):
    lines = raw.decode("ascii").splitlines()
    found = [index for index, line in enumerate(lines) if line.strip() == "#" + name]
    require(len(found) == 1, "missing/duplicate numerical card setting: " + name)
    index = found[0] + 1
    while index < len(lines) and (not lines[index].strip() or lines[index].lstrip().startswith("!")):
        index += 1
    require(index < len(lines), "missing numerical card value: " + name)
    return lines[index].split("!", 1)[0].strip().upper()


def verify_maps(raw, capture, expected):
    require(pin(raw)["sha256"] == capture["sha256"] and capture["elf"] == expected,
            "map byte/provider receipt differs")
    observed, inode_keys = set(), {}
    data_files = capture["data"]
    for name, identity in data_files.items():
        require(name.startswith("/nix/store/") and name.endswith("/lib/locale/locale-archive")
                and type(identity["bytes"]) is int and identity["bytes"] > 0
                and re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]), "unexpected non-ELF map receipt")
    for line in raw.decode("ascii").splitlines():
        fields = line.split(None, 5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        name = fields[5]
        require(not name.endswith(" (deleted)") and int(fields[4]) > 0,
                "deleted/unbacked file mapping")
        require(name in expected or name in data_files, "unexpected raw map path: " + name)
        if name in data_files:
            require(fields[1] == "r--p", "locale mapping permissions differ")
        observed.add(name)
        inode_keys.setdefault(name, set()).add((fields[3], fields[4]))
    require(observed == set(expected) | set(data_files)
            and all(len(values) == 1 for values in inode_keys.values()),
            "raw map path/inode coverage differs")
    return len(data_files)


def inspect_incomplete(retained, prefix, request, native):
    """Check the one retained timeout without interpreting coefficient words."""
    require(request["id"] == "cms-p21-mode1-lib1-original" and request["point"] == 21
            and request["backend"] == request["requested_mode"] == 1,
            "unexpected incomplete worker")
    require(native["orchestration_complete"] is False and native["runtime_map_verified"] is False
            and native["trace_complete"] is False and native["native_returncode"] == -15
            and native["stdout_overflow"] is False and native["collection_errors"] == []
            and native["cleanup_failure"] is None
            and native["failure"] == {"class": "ProtocolError", "message":
                "Worker protocol timeout; see " + ROOT + WORK + prefix + "stdout.txt"},
            "incomplete worker termination differs")
    require(len(native["close_receipts"]) == 2 and all(item == {
        "complete": True, "graceful_requested": False, "pid": native["native_pid"],
        "returncode": -15, "worker_closed": True} for item in native["close_receipts"]),
        "incomplete worker close receipts differ")
    protocol = native["protocol"]
    require(len(protocol) == 1 and len(protocol[0]) == 7
            and protocol[0][:6] == ["READY", "1", "1", "1", "1", "1"]
            and re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?", protocol[0][6]),
            "incomplete worker is not READY-only")
    names = {"bootstrap-values.dat", "dimensions.dat", "driver-bits.dat", "driver-state.dat",
             "qualification.dat", "trace.dat"}
    require(set(native["outputs"]) == names, "incomplete output inventory differs")
    raw = {name: checked(retained, prefix + name, native["outputs"][name]).decode("ascii") for name in names}
    require(raw["trace.dat"] == "" and raw["dimensions.dat"].split() == ["1", "1", "1"],
            "incomplete worker dimensions/trace differ")
    state_rows = [line.split() for line in raw["driver-state.dat"].splitlines()
                  if line.strip() and not line.startswith("#")]
    require(len(state_rows) == 3, "incomplete state record count differs")
    states = []
    expected = [("B", 0, 1, 1, -987654321, -999), ("A", 0, 1, 1, 110, 1),
                ("B", 1, 2, 21, -987654321, -999)]
    for fields, (stage, phase, call, point, code, actual) in zip(state_rows, expected):
        require(len(fields) == 16 and fields[:2] == ["STATE", stage]
                and list(map(int, fields[2:12])) == [phase, call, point, 1, code, actual, 1, 0, 0, 0]
                and fields[12:] == ["00000001", "00000000", "00000000", "00000000"],
                "incomplete call markers/flags differ")
        states.append({"stage": stage, "phase": phase, "call": call, "point": point,
                       "requested_mode": 1, "code": code, "actual_mode": actual,
                       "checkphase": 1, "helicity_checked": 0, "fpe_dp": 0, "fpe_qp": 0})
    bit_rows = [line.split() for line in raw["driver-bits.dat"].splitlines()
                if line.strip() and not line.startswith("#")]
    require(len(bit_rows) == 2, "incomplete raw-bit coverage differs")
    rows = []
    for order, fields in enumerate(bit_rows):
        require(len(fields) == 10 and list(map(int, fields[:5])) == [0, 1, 1, order, 110]
                and all(re.fullmatch(r"[0-9A-F]{16}", item) for item in fields[5:]),
                "incomplete raw-bit schema differs")
        rows.append({"phase": 0, "call": 1, "point": 1, "order": order,
                     "code": 110, "words": fields[5:]})
    decimal_rows = [line.split() for line in raw["bootstrap-values.dat"].splitlines()
                    if line.strip() and not line.startswith("#")]
    require(len(decimal_rows) == 2 and all(len(row) == 9 for row in decimal_rows)
            and [list(map(int, row[:4])) for row in decimal_rows] == [[1, 1, 0, 110], [1, 1, 1, 110]]
            and all(re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?", word)
                    for row in decimal_rows for word in row[4:]), "incomplete bootstrap text differs")
    return {"states": states, "raw_rows": rows, "eligible": False,
            "failures": ["Measurement did not return before the retained protocol timeout"],
            "incomplete_contribution_warnings": [line for line in retained(prefix + "stdout.txt").decode(
                "utf-8", errors="replace").splitlines() if "##W03" in line]}


def verify_workers(data, document, build_data, build_document, closures):
    """Verify only ordinary/stress stages explicitly supplied as closed records.

    Main source paths use proposal/, current build/stage paths results/.
    build_data/build_document access the immutable earlier build sibling.
    Stage failure and candidate-gate failure are deliberately separate outputs.
    """
    require(closures and set(closures) <= {"ordinary", "stress"} and "ordinary" in closures,
            "unsupported/missing closed stage")
    source = document_checked(data, document, "proposal/index.json", SOURCE_SEAL)
    require(source["payload_files"] == len(source["files"]) == 55, "source inventory differs")

    def source_bytes(name):
        safe_name(name)
        require(name in source["files"], "unsealed source payload")
        return checked(data, "proposal/" + name, source["files"][name])

    compare_source = source_bytes("amplitude_compare.py")
    plan = json.loads(source_bytes("validation-plan.json"))
    for name, expected in SOURCE_PINS.items():
        if name.startswith(".tirust-build/"):
            raw = checked(build_data, "source/parity.py", expected)
        else:
            raw = checked(data, "comparison-sources/" + name, expected)
        require(pin(raw) == expected, "comparison source attribution differs")
    build_closure = document_checked(data, document, "results/build-closure.json", BUILD_SEAL)
    require(build_closure["source_index_sha256"] == SOURCE_SEAL and build_closure["passed"],
            "closed build/source mismatch")
    builds = {}
    for variant in ("original", "candidate"):
        name = "builds/" + variant + "/build.json"
        builds[variant] = document_checked(data, document, "results/" + name, build_closure["payloads"][name])
    require(builds["original"]["runtime"]["files"] == builds["candidate"]["runtime"]["files"],
            "original/candidate providers differ")
    all_pids, summary = set(), {}
    for stage in ("ordinary", "stress"):
        if stage not in closures:
            continue
        closure = closures[stage]
        checksum = ORDINARY_SEAL if stage == "ordinary" else STRESS_SEAL
        saved = document_checked(data, document, "results/" + stage + "-closure.json", checksum)
        require(saved == closure and closure["stage"] == stage
                and closure["source_index_sha256"] == SOURCE_SEAL
                and closure["reviewed_build_sha256"] == BUILD_SEAL,
                "closed stage source/build binding differs")
        if stage == "stress":
            require(closure["reviewed_ordinary_sha256"] == ORDINARY_SEAL,
                    "stress did not follow the reviewed ordinary stage")
        allowed = {"results/" + safe_name(name): expected for name, expected in closure["payloads"].items()}

        def retained(name):
            safe_name(name)
            require(name in allowed, "comparison attempted a non-closed payload read: " + name)
            return checked(data, name, allowed[name])

        def retained_document(name):
            raw = retained(name)
            result = json.loads(raw)
            require(result == document(name), "closed document/bytes mismatch")
            return result

        requests = [item for item in plan["requests"] if item["stage"] == stage]
        expected_count = 8 if stage == "ordinary" else 16
        require(len(requests) == expected_count and len({item["id"] for item in requests}) == expected_count,
                "sealed worker matrix differs")
        account = retained_document("results/" + stage + "-requests.json")
        completed_requests = requests if stage == "ordinary" else requests[:8]
        launched_requests = requests if stage == "ordinary" else requests[:9]
        require(account["requests"] == requests and account["completed"] == [item["id"] for item in completed_requests]
                and account["not_completed"] == requests[len(completed_requests):]
                and account["maximum_outer_entries"] == 2 * expected_count
                and account["performance_measurement"] is False, "request completion accounting differs")
        expected_failure = None if stage == "ordinary" else "RuntimeError('Command returned 1')"
        require(account["failure"] == closure["failure"] == expected_failure
                and closure["postcheck_failure"] is None and closure["passed"] is (stage == "ordinary"),
                "closed stage outcome differs")
        if stage == "stress":
            require(all(item["point"] == 20 for item in completed_requests)
                    and all(item["point"] == 21 for item in requests[8:])
                    and "stress-comparison.json" not in closure["payloads"],
                    "partial stress comparison scope differs")
            for request in requests[9:]:
                require(not any(name.startswith("runs/" + request["id"] + "/")
                                or name == "runtime/" + request["id"] + ".json"
                                or name.startswith("execution/stress-worker-" + request["id"] + ".")
                                for name in closure["payloads"]), "unlaunched request has retained worker evidence")
        namespace = comparator_namespace(compare_source, retained)
        recomputed = namespace["compare_stage"]("results/runs", completed_requests)
        comparison_name = "results/ordinary-comparison.json" if stage == "ordinary" else "results/root-point20-comparison.json"
        comparison_bytes = retained(comparison_name) if stage == "ordinary" else checked(data, comparison_name, POINT20_COMPARISON_SHA)
        recorded = json.loads(comparison_bytes)
        require(recorded == document(comparison_name), "comparison byte/document callbacks differ")
        expected_record = dict(recomputed)
        if stage == "stress":
            expected_record["subset_scope"] = {"closed_parent_stress_sha256": STRESS_SEAL,
                "closed_parent_passed": False, "subgroup_point": 20, "point21_unqualified": True,
                "native_executions": 0, "not_a_replacement_full_stress_comparison": True}
        require(expected_record == recorded, "full comparison record does not reproduce")
        require(comparison_bytes ==
                (json.dumps(expected_record, sort_keys=(stage == "ordinary"), indent=2) + "\n").encode("ascii"),
                "comparison serialized bytes differ")
        require(recomputed["workers_requested"] == recomputed["workers_with_complete_protocol"] == len(completed_requests)
                and recomputed["outer_entries_observed"] == recomputed["ready_done_maps"] == 2 * len(completed_requests)
                and recomputed["fixed_returns_are_unchecked"] is True
                and recomputed["normal_host_stability_qualified"] is False
                and recomputed["independent_accuracy_qualified"] is False
                and recomputed["performance_measurement"] is False, "comparison scope/accounting differs")
        worker_summaries, command_names, locale_maps = [], set(), 0
        for position, request in enumerate(launched_requests, 1):
            identifier = safe_name(request["id"])
            require("/" not in identifier, "worker id contains a directory")
            prefix = "results/runs/" + identifier + "/"
            spec = retained_document(prefix + "specification.json")
            native = retained_document(prefix + "result.json")
            incomplete = stage == "stress" and position == 9
            observation = inspect_incomplete(retained, prefix, request, native) if incomplete else recomputed["observations"][identifier]
            if incomplete:
                require(all(prefix + name not in allowed for name in set(OUTPUTS) - set(native["outputs"]))
                        and prefix + "maps-1.txt" not in allowed, "incomplete worker has unexpected measurement evidence")
            require(spec["request"] == native["request"] == request and spec["source_index_sha256"] == SOURCE_SEAL,
                    "worker specification/request mismatch")
            mode, backend = request["requested_mode"], request["backend"]
            require(mode in (1, 4) and backend in (1, 9) and request["bootstrap_calls"] == request["measurement_calls"] == 1
                    and request["outer_entries_max"] == 2
                    and request["expected_fixed_unchecked_return"] == {1: 110, 4: 120}[mode], "worker call/precision scope differs")
            cwd = ROOT + WORK + "work/workers/" + identifier + "/" + SUBPROCESS.rstrip("/")
            require(spec["cwd"] == cwd and spec["evidence"] == ROOT + WORK + "results/runs/" + identifier
                    and spec["executable"] == cwd + "/check"
                    and spec["bootstrap_file"] == cwd + "/bootstrap-input.dat"
                    and spec["point_file"] == cwd + "/measurement-input.dat", "worker paths differ")
            for source_key, pin_key in (("bootstrap", "bootstrap_pin"), ("measurement", "point_pin"),
                                        ("card", "numerical_card"), ("physical_card", "physical_card")):
                require(pin(source_bytes(request[source_key])) == spec[pin_key], "worker source/card pin differs")
            card = source_bytes(request["card"])
            expected_card = {"MLReductionLib": str(backend), "CTModeRun": str(mode), "LoopPoleMode": "0",
                             "TIRustComputeFinite": ".TRUE.", "UseQPIntegrandForTIRust": ".TRUE.",
                             "UseQPIntegrandForCutTools": ".TRUE.", "TIRustGlobalCache": "0",
                             "TIRustCacheMaxBytes": "268435456"}
            for key, value in expected_card.items():
                require(card_value(card, key) == value, "fixed worker card mode/options differ: " + key)
            stdout = retained(prefix + "stdout.txt").decode("utf-8", errors="replace")
            echoes = {}
            for key, value in re.findall(r"^\s*>\s*(\w+)\s*=\s*(.*?)\s*$", stdout, re.M):
                echoes.setdefault(key, []).append(" ".join(value.split()).upper())
            for key, value in expected_card.items():
                expected_echo = "T" if value == ".TRUE." else value
                require(echoes.get(key) == [expected_echo], "native card echo differs: " + key)
            code = request["expected_fixed_unchecked_return"]
            qualification = [line.split() for line in retained(prefix + "qualification.dat").decode("ascii").splitlines()]
            require(len(qualification) == (3 if incomplete else 5) and qualification[0] == ["NATIVE_QUALIFICATION_PENDING"]
                    and qualification[1] == ["RESTORED_STATE_IS_NOT_EFFECTIVE_CALL_STATE"]
                    and qualification[2][:3] == ["BOOTSTRAP_RESTORED_STATE", "1", str(code)]
                    and len(qualification[2]) == 5 and qualification[2][3] == "0"
                    and qualification[2][4] == str(mode), "bootstrap restored-state qualification differs")
            if not incomplete:
                require(qualification[4] == ["STATUS_ALONE_IS_NOT_NUMERICAL_VALIDATION"]
                        and qualification[3] == ["MEASUREMENT_RESTORED_STATE", str(code), "0", str(mode)],
                        "measurement restored-state qualification differs")
            require(native["numerically_qualified"] is False and native["performance_measurement"] is False,
                    "worker overstates numerical qualification")
            pid = native["native_pid"]
            require(type(pid) is int and pid > 0 and pid not in all_pids, "native PID is absent/reused")
            all_pids.add(pid)
            built = builds[request["variant"]]
            providers = built["runtime"]["files"]
            require(len(providers) == 15 and spec["executable_pin"] == built["executable_pin"],
                    "worker executable/provider build differs")
            expected_runtime = {**providers, spec["executable"]: built["executable_pin"]["sha256"]}
            require(spec["expected_runtime"] == expected_runtime, "worker runtime admission differs")
            require([capture["boundary"] for capture in native["runtime_maps"]] == (["READY"] if incomplete else ["READY", "DONE"]),
                    "runtime map boundaries differ")
            for capture in native["runtime_maps"]:
                name = safe_name(capture["file"])
                require("/" not in name, "unsafe map filename")
                locale_maps += verify_maps(retained(prefix + name), capture, expected_runtime)
            admission = retained_document("results/runtime/" + identifier + ".json")
            require(admission["label"] == identifier and admission["passed"] is True and admission["failure"] is None
                    and admission["loader_query_started"] is True and admission["executable"] == spec["executable"]
                    and admission["executable_pin"] == spec["executable_pin"]
                    and admission["actual_interpreter"] == built["runtime"]["loader"]
                    and admission["runtime"]["files"] == providers
                    and admission["runtime"]["loader"] == built["runtime"]["loader"]
                    and admission["runtime"]["loader_sha256"] == built["runtime"]["loader_sha256"],
                    "per-worker loader/provider record differs")
            query_name = "results/execution/" + stage + "-query-" + stage + f"-{position:04d}"
            worker_name = "results/execution/" + stage + "-worker-" + identifier
            for name in (query_name, worker_name):
                command_names.add(name + ".json")
                command = retained_document(name + ".json")
                log = retained(name + ".txt")
                failed_command = incomplete and name == worker_name
                require(pin(log)["sha256"] == command["log_sha256"] and command["returncode"] == (1 if failed_command else 0)
                        and command["exception"] == (expected_failure if failed_command else None)
                        and command["cleanup_exception"] is None
                        and command["cleanup"]["survivors"] == [] and command["cpu"] == 12
                        and command["cpu_exclusive"] is False, "owned command did not close cleanly")
                if name == query_name:
                    require(command["argv"][-3:] == [admission["actual_interpreter"], "--list", spec["executable"]]
                            and log.decode("utf-8") == admission["runtime"]["loader_listing"],
                            "loader query transcript/command differs")
                else:
                    require(command["cwd"] == cwd and command["argv"][-4:] == [
                        ROOT + PROPOSAL + "worker_entry.py", ROOT + WORK + prefix + "specification.json",
                        pin(retained(prefix + "specification.json"))["sha256"], SOURCE_SEAL],
                        "worker entry/specification command differs")
                    require(any(item["pid"] == pid for item in command["cleanup"]["observed_owned_processes"]),
                            "native PID is absent from owned descendants")
            states = observation["states"]
            require(len(states) == (3 if incomplete else 4) and all(item["checkphase"] == 1 and item["helicity_checked"] == 0 for item in states),
                    "restored driver state flags differ")
            require(all(item["fpe_dp"] == item["fpe_qp"] == 0 for item in states if item["phase"] == 0),
                    "bootstrap contribution flags are not clean")
            worker_summaries.append({"id": identifier, "requested_backend": backend, "requested_mode": mode,
                                     "expected_fixed_unchecked_return": code,
                                     "measurement_return_recorded": not incomplete,
                                     "fixed_unchecked_return": None if incomplete else code, "eligible": observation["eligible"],
                                     "failures": observation["failures"],
                                     "measurement_flags": [{key: item[key] for key in
                                         ("stage", "checkphase", "helicity_checked", "fpe_dp", "fpe_qp")}
                                         for item in states if item["phase"] == 1],
                                     "raw_accuracy_words": sorted({row["words"][4] for row in observation["raw_rows"]}),
                                     "incomplete_contribution_warnings": observation["incomplete_contribution_warnings"]})
        actual_commands = {"results/" + name for name in closure["payloads"]
                           if name.startswith("execution/") and name.endswith(".json")}
        require(command_names == actual_commands and len(command_names) == 2 * len(launched_requests)
                and closure["owned_command_count"] == 2 * len(launched_requests)
                and closure["loader_query_count"] == len(launched_requests), "closed command scope differs")
        if stage == "ordinary":
            require(recomputed["passed"] and recomputed["candidate_gate_passed"]
                    and recomputed["ordinary_target_passed"] and recomputed["workers_eligible"] == 8,
                    "ordinary qualification changed")
        incomplete_count = len(launched_requests) - len(completed_requests)
        summary[stage] = {"workers_requested": expected_count, "workers_launched": len(launched_requests),
                          "workers_complete": len(completed_requests), "workers_not_launched": expected_count - len(launched_requests),
                          "precall_markers": 2 * len(launched_requests),
                          "recorded_returns": 2 * len(completed_requests) + incomplete_count,
                          "recorded_measurement_returns": len(completed_requests),
                          "retained_fifo_protocol_records_verified": len(launched_requests),
                          "raw_fifo_transcript_available": False, "raw_state_and_bit_files_verified": 2 * len(launched_requests),
                          "maps_verified": 2 * len(completed_requests) + incomplete_count, "locale_map_records": locale_maps,
                          "comparison_serialized_bytes_reproduced": True,
                          "stage_passed": closure["passed"], "stage_failure": closure["failure"],
                          "comparison_scope": "full ordinary matrix" if stage == "ordinary" else "completed point-20 subset only",
                          "comparison_passed": recomputed["passed"], "candidate_gate_passed": recomputed["candidate_gate_passed"],
                          "ordinary_target_passed": recomputed["ordinary_target_passed"],
                          "eligible_workers": recomputed["workers_eligible"],
                          "original_failures": recomputed["original_failures"],
                          "candidate_failures": recomputed["candidate_failures"], "worker_usage": worker_summaries}
    return {"stages": summary, "fixed_returns_are_unchecked": True,
            "failed_coefficient_words_interpreted": False, "host_stability_qualified": False,
            "independent_accuracy_qualified": False, "performance_measurement": False,
            "native_executions": 0, "external_binary_reads": 0}
