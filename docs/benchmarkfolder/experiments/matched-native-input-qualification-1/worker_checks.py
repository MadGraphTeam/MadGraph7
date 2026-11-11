"""Pure replay of the closed matched-input and outer-amplitude gates.

Only supplied retained-byte callbacks are read. No native binary, historical
absolute file, materialized worker input or controller is executed. Scientific
decoder definitions are selected unchanged from their authenticated source.
The caller authenticates sibling packet seals and the build/source provenance.
"""
import ast
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
import re
from types import ModuleType

SCIENCE = "357e116b51879cda1e1145c9d040194732c21ffd9fcd2557d9f68451901b262b"
CONTROLLER = "9076a51a0be56c57fb7dda32cfec3ef9b038bec2e73dc5e751bf31dc5ac3bc41"
RUN = "52faa470d8be22b81f8636edba21f5fba6c2d13be1f0241a8212a7b2055839a7"
BUILD = "c09358a689b2b856b81580a7d0371b73a3d13edb3a173e9b8aed4b8f143ad00d"
OLD_SOURCE = "ccbb3dff88f673043251dac5d5099d184f709fdffdba1e5f2c3c1dc0aca37a74"
OLD_COMPARE = "ecce1e1f99f0de7d1798880cc9137251b4e553accc708f3cee357ed794b99272"
REFERENCE = "0a0dba9baf12ff5a988e4d459141a48140f285a396658ee10f7f09099848dbd8"
STRUCTURAL = "34277061d0a367e156f39519e2cad973adebaf89e9ee4c1c794313238f93cba4"
COMPARE = "bec9ac7757e99ca76cba4e5c1d7fdb52fb3f7407e0bc700dae03098318815a04"
DECODER = "5b3c88d636efb3fb2e91bdf0afab030fd834c6e669317bac2872f27c0af8998d"
ROOT = "/shared/external/madloop_tirust/madgraph7_madloop_tirust/"
WORK = ".tirust-build/kinematic-matched-native-input-executor-2/"
PROPOSAL = ".tirust-build/kinematic-matched-native-input-controller-2/"
OUTPUTS = ("bootstrap-values.dat", "driver-state.dat", "driver-bits.dat", "qualification.dat",
           "dimensions.dat", "values.dat", "timed_status.dat", "timing.dat", "trace.dat")
DECIMAL_TOKEN = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?\Z")


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


def verify_workers(data, document, amplitude_data, amplitude_document, legacy_data, legacy_document):
    """Reproduce retained source-bound inputs and the gated amplitude report.

    The enclosing verifier independently authenticates all sibling packets and
    source/build derivation. The source-preflight receipt is joined here as
    historical provenance; executable/provider bytes are never reread.
    """
    def doc(callback, documents, name, expected=None):
        raw = callback(name) if expected is None else checked(callback, name, expected)
        value = json.loads(raw)
        require(value == documents(name), "document/byte callback differs: " + name)
        return value

    science_index = doc(data, document, "science/index.json", SCIENCE)
    controller_index = doc(data, document, "controller/index.json", CONTROLLER)
    require(len(science_index["files"]) == 35 and len(controller_index["files"]) == 31
            and controller_index["scientific_source_sha256"] == SCIENCE, "source inventory/ancestry differs")

    def source(prefix, index, name):
        safe(name)
        require(name in index["files"], "unsealed source path")
        return checked(data, prefix + "/" + name, index["files"][name])

    science_bytes = {name: source("science", science_index, name) for name in science_index["files"]}
    controller_bytes = {name: source("controller", controller_index, name) for name in controller_index["files"]}
    closure = doc(data, document, "results/run-closure.json", RUN)
    build_closure = doc(data, document, "results/build-closure.json", BUILD)
    require(closure["source_index_sha256"] == build_closure["source_index_sha256"] == CONTROLLER
            and closure["reviewed_build_sha256"] == BUILD and closure["stage"] == "run"
            and closure["passed"] is True and closure["failure"] is None and closure["postcheck_failure"] is None
            and closure["owned_command_count"] == 4 and closure["loader_query_count"] == 2
            and closure["performance_measurement"] is False and closure["host_stability_qualified"] is False
            and len(closure["payloads"]) == 43 and build_closure["passed"] is True,
            "closed source/build/run scope differs")
    allowed = {"results/" + safe(name): value for name, value in closure["payloads"].items()}

    def retained(name):
        safe(name)
        require(name in allowed, "non-closed runtime payload: " + name)
        return checked(data, name, allowed[name])

    def retained_doc(name):
        return doc(retained, document, name)

    build_raw = checked(data, "results/builds/matched/build.json", build_closure["payloads"]["builds/matched/build.json"])
    build = json.loads(build_raw)
    require(build == document("results/builds/matched/build.json") and build["variant"] == "matched",
            "retained matched build differs")
    build_pin = pin(build_raw)
    require(build_pin == {"bytes": 411303, "sha256": "0bf8c1549e31dfec1f41e15380e23e95376bfd9ded4aa55474171643313e4c7c"},
            "matched build receipt identity differs")
    old_index = doc(amplitude_data, amplitude_document, "proposal/index.json", OLD_SOURCE)
    old_raw = checked(amplitude_data, "proposal/amplitude_compare.py", OLD_COMPARE)
    require(pin(old_raw) == old_index["files"]["amplitude_compare.py"], "old comparator membership differs")
    reference_index = doc(legacy_data, legacy_document, "source/index.json", REFERENCE)
    reference_raw = checked(legacy_data, "source/reference_compare.py", STRUCTURAL)
    require(pin(reference_raw) == reference_index["files"]["reference_compare.py"], "structural source membership differs")
    require(pin(science_bytes["matched_inputs.py"])["sha256"] == DECODER
            and pin(controller_bytes["compare.py"])["sha256"] == COMPARE, "decoder/comparison source changed")
    plan = json.loads(controller_bytes["validation-plan.json"])
    requests = plan["requests"]
    require(retained_doc("results/run-requests.json") == {"requests": requests,
        "completed": [request["id"] for request in requests], "failure": None,
        "not_completed": [], "maximum_outer_entries": 4, "performance_measurement": False},
        "request completion account differs")

    # Absolute paths stay provenance strings. This virtual path exposes only
    # joined logical retained paths; it has no filesystem or resolve operation.
    class RetainedPath:
        def __init__(self, value):
            self.logical = safe(value.logical if isinstance(value, RetainedPath) else value)

        def __truediv__(self, value):
            return RetainedPath(self.logical + "/" + safe(value))

        def __str__(self):
            return self.logical

        @property
        def name(self):
            return self.logical.rsplit("/", 1)[-1]

        def read_bytes(self):
            if self.logical.startswith("amplitude/"):
                name = self.logical[len("amplitude/"):]
                require(name in ("results/ordinary-closure.json", "results/ordinary-comparison.json"),
                        "unadmitted historical amplitude read")
                return amplitude_data(name)
            return retained(self.logical)

    # Scientific definitions and assignments are compiled unchanged. The only
    # excluded definition is the external-filesystem verify_source_contract.
    science_tree = ast.parse(science_bytes["matched_inputs.py"])
    science_names = {"require", "integers", "_routes", "environment_valid", "expected_wire", "parse_trace", "join_inputs"}
    science_functions = [node for node in science_tree.body if isinstance(node, ast.FunctionDef) and node.name in science_names]
    require(len(science_functions) == 7 and {node.name for node in science_functions} == science_names,
            "scientific function selection differs")
    scientific_nodes = [node for node in science_tree.body if isinstance(node, ast.Assign)
                        or isinstance(node, ast.FunctionDef) and node.name in science_names]
    scientific = ModuleType("retained_strict_matched_inputs")
    scientific.__dict__.update(hashlib=hashlib, math=math, re=re)
    exec(compile(ast.Module(body=scientific_nodes, type_ignores=[]), "sealed-strict-matched-inputs", "exec"), scientific.__dict__)
    routes = {int(key): value for key, value in json.loads(science_bytes["expected-routes.json"]).items()}
    scientific._routes(routes)
    preflight = json.loads(science_bytes["audit/final-source-preflight.json"])
    require(preflight["sealed_index"] is None and preflight["passed"] is True and preflight["native_actions"] == 0
            and preflight["routes"] == 26 and preflight["generated_products"] == 12,
            "historical source preflight differs")
    preflight = {**preflight, "sealed_index": SCIENCE}
    science_evidence = {"scientific_index_sha256": SCIENCE, "payloads_verified": len(science_bytes),
        "matched_inputs.py": science_index["files"]["matched_inputs.py"],
        "expected-routes.json": science_index["files"]["expected-routes.json"]}

    # The only legacy assertion adaptation is the exact sealed controller's
    # trace gate. A real module makes its attribute and function-global state
    # identical, preserving when the controller opens the gate.
    anchor = "        require(not raw['trace.dat'],'Unexpected diagnostic trace output')"
    require(old_raw.decode().count(anchor) == 1, "legacy trace guard differs")
    adapted = old_raw.decode().replace(anchor,
        "        require(_MATCHED_GATE_OPEN,'Matched native-input gate is not open')", 1)
    legacy_tree, constants = selected(adapted, {"require", "pin", "decode64", "wire", "_states", "_coverage",
        "_decimal_shape", "inspect_worker", "compare_coefficients"}, {"OUTPUTS", "COLUMNS"})
    require(constants["OUTPUTS"] == OUTPUTS and constants["COLUMNS"] == ("Born", "finite", "single_pole", "double_pole"),
            "legacy output columns differ")
    legacy = ModuleType("retained_matched_amplitude")
    legacy.__dict__.update(hashlib=hashlib, json=json, re=re, Path=RetainedPath, F=Fraction,
        Decimal=Decimal, localcontext=localcontext, DECIMAL_TOKEN=DECIMAL_TOKEN,
        TOL=Fraction(3, 1000000), TARGET=Fraction(1, 100000000), _MATCHED_GATE_OPEN=False, **constants)
    exec(compile(legacy_tree, "sealed-matched-amplitude-functions", "exec"), legacy.__dict__)
    structural_tree, _ = selected(reference_raw, {"require", "pin", "checked", "_structural_evidence"})
    structural = ModuleType("retained_matched_structural")
    structural.__dict__.update(hashlib=hashlib, json=json, Path=RetainedPath, _legacy=lambda: legacy)
    exec(compile(structural_tree, "sealed-matched-structural-functions", "exec"), structural.__dict__)
    compare_tree, compare_constants = selected(controller_bytes["compare.py"],
        {"require", "pin", "checked", "_saved_control", "validate_requests", "_phase_view", "_finite_inputs", "compare_stage"},
        {"MATCHED_SHA", "OLD_SHA", "OLD_COMPARE_SHA", "REFERENCE_COMPARE_SHA", "ORDINARY_CLOSURE_SHA", "ORDINARY_COMPARE_SHA"})
    require(compare_constants == {"MATCHED_SHA": SCIENCE, "OLD_SHA": OLD_SOURCE, "OLD_COMPARE_SHA": OLD_COMPARE,
        "REFERENCE_COMPARE_SHA": STRUCTURAL,
        "ORDINARY_CLOSURE_SHA": "b364bd8535d0f5850cb5a21390c5678cbb1c1dd8fddfae6f0174cf50d261f7a7",
        "ORDINARY_COMPARE_SHA": "6d4f37f0b9b0cc0b3297d31b7167d176db640bbca21cd33f8861a0e4b9750451"},
        "controller comparison constants differ")

    def inspected(directory, request, controller_sha, supplied_build, supplied_pin, structure):
        """Portable form of the file-access boundary, preserving field checks."""
        identifier = request["id"]
        require(directory.logical == "results/runs/" + identifier and controller_sha == CONTROLLER
                and supplied_build == build and supplied_pin == build_pin, "inspection arguments differ")
        prefix = directory.logical + "/"
        raw = retained(prefix + "specification.json")
        spec = json.loads(raw)
        require(spec["request"] == request and spec["source_index_sha256"] == CONTROLLER
                and spec["scientific_source_index_sha256"] == SCIENCE
                and spec["build_receipt_sha256"] == build_pin["sha256"], "worker specification source/build differs")
        cwd = ROOT + WORK + "work/workers/" + identifier + "/SubProcesses/P0_epem_mupmum"
        require(spec["cwd"] == cwd and spec["executable"] == cwd + "/check"
                and spec["evidence"] == ROOT + WORK + directory.logical
                and spec["executable_pin"] == build["executable_pin"], "worker executable/path identity differs")
        expected_runtime = {**build["runtime"]["files"], spec["executable"]: build["executable_pin"]["sha256"]}
        require(spec["expected_runtime"] == expected_runtime, "worker admitted provider set differs")
        for source_key, pin_key in (("card", "numerical_card"), ("physical_card", "physical_card"),
                                     ("bootstrap", "bootstrap_pin"), ("measurement", "point_pin")):
            name = request[source_key]
            require(science_bytes[name] == controller_bytes[name] and spec[pin_key] == pin(science_bytes[name]),
                    "retained literal card/point identity differs")
        require(spec["bootstrap_file"] == cwd + "/bootstrap-input.dat"
                and spec["point_file"] == cwd + "/measurement-input.dat", "literal point path differs")
        obs = {"id": identifier, "request": request, "eligible": False, "failures": [], "raw_rows": [], "states": [],
            "source_evidence": {"specification.json": pin(raw)}, "native_pid": None,
            "protocol_complete": False, "structural_complete": False, "outer_entries_observed": 0,
            "ready_done_maps": 0, "measurement_interpretation_allowed": False}
        native_raw = retained(prefix + "result.json")
        native = json.loads(native_raw)
        require(native["request"] == request, "native request differs")
        obs["source_evidence"]["result.json"] = pin(native_raw)
        structure._structural_evidence(directory, request, native, obs)
        require(all(capture["elf"] == expected_runtime for capture in native["runtime_maps"]), "runtime map admission differs")
        trace = checked(retained, prefix + "trace.dat", native["outputs"]["trace.dat"])
        obs["source_evidence"]["trace.dat"] = pin(trace)
        return obs, trace

    def retained_build(runs, provided):
        require(runs.logical == "results/runs" and provided == build, "provided build differs")
        return build, build_pin

    def retained_controller(expected):
        require(expected == CONTROLLER, "controller source seal differs")
        return pin(data("controller/index.json"))

    outer = ModuleType("retained_matched_outer")
    outer.__dict__.update(hashlib=hashlib, json=json, Path=RetainedPath, **compare_constants,
        ORDINARY=RetainedPath("amplitude/results"), _controller_source=retained_controller,
        _scientific_source=lambda: {"module": scientific, "routes": routes, "preflight": preflight,
                                    "source_evidence": science_evidence},
        _build=retained_build, _inspect=inspected, _legacy=lambda: legacy, _reference=lambda: structural)
    exec(compile(compare_tree, "sealed-matched-outer-gates", "exec"), outer.__dict__)
    arithmetic = []
    original_compare = legacy.compare_coefficients

    def tracked_compare(reference, candidate):
        require(legacy._MATCHED_GATE_OPEN is True and reference["eligible"] is True and candidate["eligible"] is True,
                "coefficient arithmetic reached a closed gate")
        arithmetic.append((reference["id"], candidate["id"]))
        return original_compare(reference, candidate)

    legacy.compare_coefficients = tracked_compare
    replayed = outer.compare_stage("results/runs", requests, controller_source_sha=CONTROLLER, build_receipt=build)
    raw_comparison = retained("results/matched-comparison.json")
    require(replayed == json.loads(raw_comparison) == document("results/matched-comparison.json")
            and raw_comparison == (json.dumps(replayed, sort_keys=True, indent=2) + "\n").encode("ascii"),
            "complete strict input/outer comparison does not reproduce")
    require(arithmetic == [("cms-p1-mode4-lib9-candidate", "cms-p20-mode4-lib8-matched"),
                           ("cms-p1-mode4-lib9-candidate", "cms-p20-mode4-lib9-matched"),
                           ("cms-p20-mode4-lib8-matched", "cms-p20-mode4-lib9-matched")],
            "arithmetic call sequence differs from two bootstraps then measurement")
    require(replayed["failures"] == [] and replayed["outer_entries_observed"] == replayed["ready_done_maps"] == 4
            and all(replayed[key] is True for key in ("passed", "execution_complete", "native_input_gate_passed",
                "native_inputs_finite", "outer_output_gate_passed", "bootstrap_controls_passed", "measurement_interpretation_allowed"))
            and all(replayed[key] is False for key in ("accuracy_qualified", "normal_host_stability_qualified",
                                                       "performance_measurement", "winner_selected")), "outer gate/scope differs")
    joined = replayed["input_match"]
    require(all(joined[key] is True for key in ("exact_input_match", "certified_geometry_match", "caller_binding_verified",
                "qualified_input_match", "native_status_qualified", "native_input_gate_passed"))
            and all(joined[key] == [] for key in ("mismatches", "geometry_failures", "native_status_failures"))
            and joined["finite_comparison_admitted"] is False and joined["accuracy_qualified"] is False,
            "scientific input match changed qualification scope")
    providers = build["runtime"]["files"]
    require(len(providers) == 15, "runtime provider scope differs")
    commands, pids, worker_summary = set(), set(), []
    for position, request in enumerate(requests, 1):
        backend, identifier = request["backend"], request["id"]
        prefix = "results/runs/" + identifier + "/"
        spec, native = retained_doc(prefix + "specification.json"), retained_doc(prefix + "result.json")
        obs = replayed["observations"][str(backend)]
        require(obs["eligible"] is True and obs["failures"] == [] and obs["measurement_interpretation_allowed"] is True
                and native["numerically_qualified"] is False and native["performance_measurement"] is False,
                "worker outer eligibility/scope differs")
        require(native["native_pid"] not in pids, "native PID reused")
        pids.add(native["native_pid"])
        require(set(native["outputs"]) == set(OUTPUTS), "worker output inventory differs")
        for name in OUTPUTS:
            checked(retained, prefix + name, native["outputs"][name])
        expected_runtime = {**providers, spec["executable"]: build["executable_pin"]["sha256"]}
        require([capture["file"] for capture in native["runtime_maps"]] == ["maps-0.txt", "maps-1.txt"], "map sequence differs")
        for capture in native["runtime_maps"]:
            verify_maps(retained(prefix + capture["file"]), capture, expected_runtime)
        require(all(state["checkphase"] == 1 and state["helicity_checked"] == state["fpe_dp"] == state["fpe_qp"] == 0
                    for state in obs["states"]), "restored contribution/unchecked state differs")
        qualification = [line.split() for line in retained(prefix + "qualification.dat").decode("ascii").splitlines()]
        require(qualification == [["NATIVE_QUALIFICATION_PENDING"], ["RESTORED_STATE_IS_NOT_EFFECTIVE_CALL_STATE"],
            ["BOOTSTRAP_RESTORED_STATE", "1", "120", "0", "4"], ["MEASUREMENT_RESTORED_STATE", "120", "0", "4"],
            ["STATUS_ALONE_IS_NOT_NUMERICAL_VALIDATION"]], "qualification text differs")
        card = science_bytes[request["card"]]
        settings = {"MLReductionLib": str(backend), "CTModeRun": "4", "LoopPoleMode": "0",
                    "TIRustComputeFinite": ".TRUE.", "UseQPIntegrandForNinjaRS": ".TRUE.",
                    "UseQPIntegrandForTIRust": ".TRUE.", "TIRustGlobalCache": "0",
                    "TIRustCacheMaxBytes": "268435456"}
        echoes = {}
        console = retained(prefix + "stdout.txt").decode("utf-8", errors="replace")
        require("##W03" not in console, "incomplete contribution warning")
        for key, value in re.findall(r"^\s*>\s*(\w+)\s*=\s*(.*?)\s*$", console, re.M):
            echoes.setdefault(key, []).append(" ".join(value.split()).upper())
        for key, value in settings.items():
            require(card_value(card, key) == value and echoes.get(key) == ["T" if value == ".TRUE." else value],
                    "literal/native card setting differs: " + key)
        parsed = joined["parsed"][position - 1]
        require(parsed["backend"] == backend and len(parsed["calls"]) == 52 and parsed["phase_call_counts"] == [26, 26],
                "native input coverage differs")
        require(all(call["status"] == 0 and call["complete"] == call["unchanged"] == 1
                    and call["mode"] == 4 and call["width"] == 128 and call["digits"] == 113 for call in parsed["calls"]),
                "actual native usage/status/precision differs")
        for phase in (0, 1):
            require([call["group"] for call in parsed["calls"] if call["phase"] == phase] == list(range(1, 27)),
                    "native group order differs")
        admission = retained_doc("results/runtime/" + identifier + ".json")
        require(admission["passed"] is True and admission["failure"] is None and admission["label"] == identifier
                and admission["loader_query_started"] is True and admission["executable"] == spec["executable"]
                and admission["executable_pin"] == build["executable_pin"] and admission["runtime"]["files"] == providers
                and admission["actual_interpreter"] == admission["runtime"]["loader"] == build["runtime"]["loader"]
                and admission["runtime"]["loader_sha256"] == build["runtime"]["loader_sha256"], "runtime admission differs")
        query_name = "results/execution/run-query-run-" + f"{position:04d}"
        worker_name = "results/execution/run-worker-" + identifier
        for name in (query_name, worker_name):
            commands.add(name + ".json")
            command, log = retained_doc(name + ".json"), retained(name + ".txt")
            require(command["returncode"] == 0 and command["exception"] is None and command["cleanup_exception"] is None
                    and command["cleanup"]["survivors"] == [] and command["cpu"] == 12 and command["cpu_exclusive"] is False
                    and command["log_sha256"] == pin(log)["sha256"], "owned command closure differs")
            if name == query_name:
                require(command["argv"][-3:] == [admission["actual_interpreter"], "--list", spec["executable"]]
                        and log.decode("utf-8") == admission["runtime"]["loader_listing"], "loader query transcript differs")
            else:
                require(command["cwd"] == spec["cwd"] and command["argv"][-4:] == [ROOT + PROPOSAL + "worker_entry.py",
                    ROOT + WORK + prefix + "specification.json", pin(retained(prefix + "specification.json"))["sha256"], CONTROLLER]
                    and any(item["pid"] == native["native_pid"] for item in command["cleanup"]["observed_owned_processes"]),
                    "worker source/specification/ownership command differs")
        worker_summary.append({"backend": backend, "id": identifier, "fixed_unchecked_return": 120,
            "native_calls": 52, "outer_returns": 2, "map_boundaries": ["READY", "DONE"],
            "qp_api_storage_bits": 128, "qp_api_significand_bits": 113, "software_flags": [0, 0],
            "numerator_wire_role": "native NinjaRS numerator" if backend == 8 else "Fortran contraction buffer; not tensor API argument"})
    require(commands == {"results/" + name for name in closure["payloads"] if name.startswith("execution/") and name.endswith(".json")},
            "closed command scope differs")
    require(replayed["measurement_comparison"]["maximum_relative_error"] == {
        "decimal": "2.01363060398197313E-16", "numerator": "1", "denominator": "4966154159668068"},
        "retained measurement residual differs")
    return {"comparison_serialized_bytes_reproduced": True, "strict_scientific_decoder_unchanged": True,
        "workers": worker_summary, "exact_matched_call_pairs": 52, "native_calls": 104, "outer_returns": 4,
        "maps_verified": 4, "coefficient_comparisons": {"bootstrap": 2, "measurement": 1},
        "measurement_maximum_relative_error": replayed["measurement_comparison"]["maximum_relative_error"],
        "private_adapter_mu_alignment": "both promote MU_R to QP before squaring",
        "ordered_q0_shifts_and_exact_numerator_bits_compared": True, "failed_legacy_coefficients_read": False,
        "raw_fifo_transcript_available": False, "historical_source_preflight_preserved": True,
        "fresh_source_derivation_replayed_by_worker_helper": False, "materialized_inputs_or_binaries_rehashed": False,
        "internal_reducer_working_precision_qualified": False, "independent_accuracy_qualified": False,
        "normal_host_stability_qualified": False, "production_promoted": False, "performance_measurement": False,
        "new_native_executions": 0, "external_binary_reads": 0}
