#!/usr/bin/env python3
"""Independent forced-DP self-consistency observations, never a timing gate.

The source observer records each reduction immediately after dispatch. Only
private generated copies are instrumented; stripping markers restores every
original source byte. This module imports no MadGraph or numerical engine.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "ninjars-dp-stability-summary-v1"
RAW_SCHEMA = "ninjars-forced-dp-raw-v1"
BEGIN = "C DPSTAB_OBSERVER_BEGIN\n"
END = "C DPSTAB_OBSERVER_END\n"
TRANSFORMS = ("original_forward", "original_reverse", "z90_forward")
TOOLS = {1: "CutTools", 6: "Ninja", 7: "COLLIER", 8: "NinjaRS"}
COUNTER_NAMES = ("calls", "rejected", "nonfinite", "higher_precision",
                 *[f"backend_{i}" for i in range(1,9)], "fpe_dp_seen",
                 "fpe_qp_seen", "nrs_compensated_dp", "ieee_invalid_seen",
                 "ieee_divzero_seen", "ieee_overflow_seen")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fixed_statements(source):
    """Yield fixed-form statement spans, preserving comments and source bytes."""
    lines = source.splitlines(keepends=True)
    current = None
    for index, line in enumerate(lines):
        if not line.strip() or line[0] in "Cc*!":
            continue
        continuation = len(line) > 5 and line[5] not in " 0\t\n"
        if continuation:
            if current is None:
                raise ValueError("Orphan fixed-form continuation")
            current[1] = index + 1
            current[2] += line[6:].strip()
        else:
            if current is not None:
                yield tuple(current)
            current = [index, index + 1, line[6:].strip()]
    if current is not None:
        yield tuple(current)


def arguments(text):
    result, start, depth = [], 0, 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            result.append(text[start:index].strip())
            start = index + 1
        if depth < 0:
            raise ValueError("Unbalanced call")
    if depth:
        raise ValueError("Unbalanced call")
    result.append(text[start:].strip())
    return result


def strip_observer(source):
    return re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), "", source,
                  flags=re.S)


def instrument_interface(source):
    """Add observers after every supported generic LOOP_n dispatch, fail closed."""
    if BEGIN in source or END in source:
        raise ValueError("Already instrumented interface")
    statements = list(fixed_statements(source))
    active, prefix, insertions, dispatches = False, None, {}, []
    supported = {"CTLOOP", "NINJA_LOOP", "MP_NINJA_LOOP", "NINJARS_LOOP",
                 "MP_NINJARS_LOOP", "TIRLOOP"}
    expected_args = {"TIRLOOP": 9}
    for start, end, statement in statements:
        unit = re.match(r"SUBROUTINE\s+(\w+)\(", statement, re.I)
        if unit:
            match = re.fullmatch(r"(\w*?)LOOP_\d+(?:_\d+)*", unit[1], re.I)
            active = bool(match)
            if match:
                if prefix is not None and prefix.upper() != match[1].upper():
                    raise ValueError("Multiple process prefixes")
                prefix = match[1]
        elif re.fullmatch(r"END(?:\s+SUBROUTINE(?:\s+\w+)?)?", statement, re.I):
            active = False
        call = re.fullmatch(r"CALL\s+(\w+)\((.*)\)", statement, re.I)
        if not active or not call:
            continue
        callee = call[1]
        if not callee.upper().startswith(prefix.upper()):
            raise ValueError("Unrecognized call inside reduction dispatcher")
        name = callee[len(prefix):].upper()
        if name == "CHOOSE_LOOPLIB":
            continue
        if name not in supported:
            raise ValueError(f"Unsupported dispatcher call: {callee}")
        args = arguments(call[2])
        if len(args) != expected_args.get(name, 6):
            raise ValueError("Unexpected reduction call signature")
        result, stable = args[-2:]
        if not re.fullmatch(r"(?:LOOPRES|AMPL)\(1,[A-Z0-9_,+]+\)", result, re.I):
            raise ValueError("Unknown result-array argument")
        if not re.fullmatch(r"S\((?:SQUAREDSOINDEX|1),LOOPNUM\)", stable, re.I):
            raise ValueError("Unknown stable-array argument")
        required_stable = ("S(SQUAREDSOINDEX,LOOPNUM)" if result.upper().startswith(
            "LOOPRES(") else "S(1,LOOPNUM)")
        if stable.upper() != required_stable:
            raise ValueError("Result/stable indices differ")
        if name == "TIRLOOP" and args[2].upper() != "I_LIB":
            raise ValueError("TIR library index differs")
        observer = (BEGIN + f"      CALL {prefix}DPSTAB_CAPTURE({stable},\n"
                    f"     $ {result},MLREDUCTIONLIB(I_LIB),CTMODE)\n" + END)
        insertions[end] = observer
        dispatches.append({"callee": callee, "line": start+1,
                           "result": result, "stable": stable})
    if not dispatches or not prefix:
        raise ValueError("No generic reduction dispatcher calls")
    lines = source.splitlines(keepends=True)
    result = "".join(line + insertions.get(index+1, "")
                     for index, line in enumerate(lines))
    if strip_observer(result) != source:
        raise ValueError("Observer changes original numeric source")
    return result, {"prefix": prefix, "dispatches": dispatches,
                    "original_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "instrumented_sha256": hashlib.sha256(result.encode()).hexdigest(),
                    "original_bytes_restored_after_stripping": True}


def render_driver(prefix, nexternal, nso):
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", prefix):
        raise ValueError("Invalid process prefix")
    if nexternal < 3 or nso < 1:
        raise ValueError("Invalid process dimensions")
    text = (ROOT / "tools/ninjars_dp_stability_driver.f").read_text()
    for key, value in {"PREFIX": prefix, "NEXTERNAL": nexternal, "NSO": nso}.items():
        text = text.replace("@"+key+"@", str(value))
    if "@" in text:
        raise ValueError("Unexpanded driver parameter")
    return text


def recorder_control_source(prefix="DPT_"):
    """Compiled later: exact production observer with forced failure/zero/FPE controls."""
    helper = render_driver(prefix, 3, 1).split(
        "C     Observer only:", 1)[1]
    helper = "C     Observer only:" + helper
    return ("      PROGRAM RECORDER_CONTROL\n"
            "      USE, INTRINSIC :: IEEE_ARITHMETIC\n"
            "      USE, INTRINSIC :: IEEE_EXCEPTIONS\n"
            "      IMPLICIT NONE\n"
            "      INTEGER*8 COUNTS(18)\n"
            f"      COMMON/{prefix}DPSTAB_COUNTS/COUNTS\n"
            "      COMPLEX*16 RES(3),SAVED(3)\n"
            "      LOGICAL STABLE,FPE_DP,FPE_QP,FLAG\n"
            f"      COMMON/{prefix}FPE_IN_REDUCTION/FPE_DP,FPE_QP\n"
            "      LOGICAL NRS_RETRY_MODE,NRS_RETRY_USED\n"
            f"      COMMON/{prefix}NRS_RETRY_FLAGS/"
            "NRS_RETRY_MODE,NRS_RETRY_USED\n"
            "      INTEGER LIB,I\n"
            "      DO I=1,4\n"
            "        LIB=1\n"
            "        IF(I.EQ.2) LIB=6\n"
            "        IF(I.EQ.3) LIB=7\n"
            "        IF(I.EQ.4) LIB=8\n"
            "        COUNTS=0\n"
            "        FPE_DP=.FALSE.\n"
            "        FPE_QP=.FALSE.\n"
            "        NRS_RETRY_MODE=.FALSE.\n"
            "        NRS_RETRY_USED=.FALSE.\n"
            "        RES=(0D0,0D0)\n"
            "        SAVED=RES\n"
            "        STABLE=.FALSE.\n"
            f"        CALL {prefix}DPSTAB_CAPTURE(STABLE,RES,LIB,1)\n"
            "        IF(COUNTS(2).NE.1) STOP 1\n"
            "        IF(STABLE.OR.ANY(RES.NE.SAVED)) STOP 2\n"
            "        STABLE=.TRUE.\n"
            f"        CALL {prefix}DPSTAB_CAPTURE(STABLE,RES,LIB,2)\n"
            "        IF(COUNTS(2).NE.1) STOP 3\n"
            "        IF(COUNTS(4+LIB).NE.2) STOP 4\n"
            "        IF(.NOT.STABLE.OR.ANY(RES.NE.SAVED)) STOP 5\n"
            "        RES(1)=CMPLX(IEEE_VALUE(0D0,IEEE_QUIET_NAN),0D0,8)\n"
            f"        CALL {prefix}DPSTAB_CAPTURE(STABLE,RES,LIB,1)\n"
            "        IF(COUNTS(3).NE.1) STOP 6\n"
            f"        CALL {prefix}DPSTAB_CAPTURE(STABLE,SAVED,LIB,4)\n"
            "        IF(COUNTS(4).NE.1) STOP 7\n"
            "        FPE_DP=.TRUE.\n"
            f"        CALL {prefix}DPSTAB_CAPTURE(STABLE,SAVED,LIB,1)\n"
            "        IF(COUNTS(13).NE.1) STOP 8\n"
            "        NRS_RETRY_MODE=.TRUE.\n"
            f"        CALL {prefix}DPSTAB_CAPTURE(STABLE,SAVED,LIB,1)\n"
            "        IF(LIB.EQ.8.AND.COUNTS(15).NE.1) STOP 9\n"
            "        IF(LIB.NE.8.AND.COUNTS(15).NE.0) STOP 10\n"
            "        COUNTS(16:18)=0\n"
            "        CALL IEEE_SET_FLAG(IEEE_INVALID,.TRUE.)\n"
            "        CALL IEEE_SET_FLAG(IEEE_DIVIDE_BY_ZERO,.TRUE.)\n"
            "        CALL IEEE_SET_FLAG(IEEE_OVERFLOW,.TRUE.)\n"
            f"        CALL {prefix}DPSTAB_CAPTURE(STABLE,SAVED,LIB,1)\n"
            "        CALL IEEE_GET_FLAG(IEEE_INVALID,FLAG)\n"
            "        COUNTS(16)=MERGE(1,0,FLAG)\n"
            "        CALL IEEE_GET_FLAG(IEEE_DIVIDE_BY_ZERO,FLAG)\n"
            "        COUNTS(17)=MERGE(1,0,FLAG)\n"
            "        CALL IEEE_GET_FLAG(IEEE_OVERFLOW,FLAG)\n"
            "        COUNTS(18)=MERGE(1,0,FLAG)\n"
            "        IF(ANY(COUNTS(16:18).NE.1)) STOP 11\n"
            "        CALL IEEE_SET_FLAG(IEEE_INVALID,.FALSE.)\n"
            "        CALL IEEE_SET_FLAG(IEEE_DIVIDE_BY_ZERO,.FALSE.)\n"
            "        CALL IEEE_SET_FLAG(IEEE_OVERFLOW,.FALSE.)\n"
            "      ENDDO\n"
            "      WRITE(*,*) 'DPSTAB_RECORDER_CONTROLS_PASS'\n"
            "      END\n" + helper)


def _number(value):
    try:
        parsed = float(value.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def parse_raw(path, expected_points, expected_tool, expected_nso):
    """Retain failed/incomplete slots; reject malformed or duplicate authority."""
    lines = Path(path).read_text().splitlines()
    header, warm, done, active = None, None, None, None
    observed = {}
    for line in lines:
        fields = line.split()
        if not fields:
            continue
        tag = fields[0]
        if tag == "DPSD_HEADER":
            if header is not None or len(fields) != 8:
                raise ValueError("Invalid raw header")
            header = list(map(int, fields[1:]))
            if (header[0], header[1], header[3], header[5]) != (
                    1, expected_points, expected_nso, expected_tool):
                raise ValueError("Raw header differs from specification")
            if header[2] < expected_nso or header[6] != 1:
                raise ValueError("Invalid dimensions or initial floating-point rounding")
        elif tag == "WARM":
            if warm is not None or len(fields) != 3:
                raise ValueError("Invalid warm marker")
            warm = {"calls": int(fields[1]), "seconds": _number(fields[2])}
        elif tag == "BEGIN":
            if len(fields) != 3 or active is not None:
                raise ValueError("Invalid observation start")
            key = tuple(map(int, fields[1:]))
            if key in observed or not (1 <= key[0] <= expected_points and 1 <= key[1] <= 3):
                raise ValueError("Duplicate or out-of-range observation")
            active = {"point": key[0], "transform": key[1], "rows": {},
                      "complete": False, "metadata": None}
            observed[key] = active
        elif tag == "OBS":
            if active is None or active["metadata"] is not None or len(fields) != 26:
                raise ValueError("Malformed observation metadata")
            numbers = list(map(int, fields[1:]))
            if numbers[:2] != [active["point"], active["transform"]]:
                raise ValueError("Observation key mismatch")
            active["metadata"] = numbers[2:]
        elif tag == "ROW":
            if active is None or len(fields) != 9:
                raise ValueError("Malformed numerical row")
            point, transform, order = map(int, fields[1:4])
            if (point, transform) != (active["point"], active["transform"]):
                raise ValueError("Numerical key mismatch")
            if order in active["rows"] or not 0 <= order <= expected_nso:
                raise ValueError("Duplicate or invalid split order")
            active["rows"][order] = {"raw": fields[4:],
                                     "values": [_number(v) for v in fields[4:]]}
        elif tag == "ENDOBS":
            if active is None or tuple(map(int, fields[1:])) != (
                    active["point"], active["transform"]):
                raise ValueError("Invalid observation completion")
            active["complete"] = (active["metadata"] is not None and
                                   len(active["rows"]) == expected_nso+1)
            active = None
        elif tag == "DONE":
            if done is not None or active is not None or len(fields) != 3:
                raise ValueError("Invalid campaign completion")
            done = {"observations": int(fields[1]), "seconds": _number(fields[2])}
        else:
            raise ValueError(f"Unrecognized raw record: {tag}")
    if header is None:
        raise ValueError("Missing raw header")
    complete = (warm is not None and warm["seconds"] is not None and
                done is not None and done["seconds"] is not None and
                done["observations"] == 3*expected_points and
                len(observed) == 3*expected_points and all(x["complete"] for x in observed.values()))
    return {"header": header, "warm": warm, "done": done,
            "complete": complete, "observations": observed}


def observation_failure(observation, tool):
    if not observation or not observation["complete"]:
        return "missing_or_incomplete_output"
    mode, code, fpdp, fpqp, nearest, *counts = observation["metadata"]
    if mode != (2 if observation["transform"] == 2 else 1):
        return "unexpected_mode"
    if len(counts) != len(COUNTER_NAMES) or any(c < 0 for c in counts):
        return "invalid_recorder_counts"
    total, rejected, nonfinite, qp = counts[:4]
    backends = counts[4:12]
    if not nearest:
        return "changed_rounding_mode"
    if counts[14]:
        return "unexpected_compensated_dp"
    if sum(backends) != total or any(c for i,c in enumerate(backends,1) if i != tool):
        return "unexpected_backend"
    if qp or fpqp or counts[13] or (code//10)%10 == 2:
        return "unexpected_higher_precision"
    if (code//10)%10 != 1:
        return "initialization_or_invalid_status"
    if code//100 != 1:
        return "unexpected_status"
    if rejected:
        return "reduction_rejected"
    if nonfinite or any(v is None for row in observation["rows"].values() for v in row["values"][:4]):
        return "nonfinite_result"
    if fpdp or counts[12]:
        return "reduction_fpe"
    return None


def estimator(observations, tool, indices, order=0, coefficient=1):
    reasons = [observation_failure(observations.get(i), tool) for i in indices]
    if any(reasons):
        return {"classification": next(r for r in reasons if r), "delta": None,
                "unbounded": True, "values": [None if not observations.get(i) or
                 order not in observations[i]["rows"] else observations[i]["rows"][order]["values"][coefficient] for i in indices]}
    values = [observations[i]["rows"][order]["values"][coefficient] for i in indices]
    scale = max(abs(v) for v in values)
    if scale == 0:
        return {"classification": "all_zero", "delta": 0., "unbounded": False,
                "values": values, "mean": 0., "spread": 0.}
    normalized = [v/scale for v in values]
    mean = math.fsum(normalized)/len(values)
    spread = max(normalized)-min(normalized)
    if mean == 0:
        return {"classification": "zero_mean_nonzero_spread", "delta": None,
                "unbounded": True, "values": values}
    delta = spread/abs(mean)
    if not math.isfinite(delta):
        return {"classification": "unbounded_relative_spread", "delta": None,
                "unbounded": True, "values": values}
    return {"classification": "finite", "delta": delta, "unbounded": False,
            "values": values, "normalized_mean": mean, "normalized_spread": spread}


def interval95(k, n):
    if not 0 <= k <= n or n <= 0:
        raise ValueError("Invalid binomial count")
    z = 1.959963984540054
    p, d = k/n, 1+z*z/n
    center = (p+z*z/(2*n))/d
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0.,center-half), min(1.,center+half)]


def tail(scores, thresholds):
    n = len(scores)
    if n == 0:
        raise ValueError("Empty stability sample")
    result = []
    for threshold in thresholds:
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("Invalid stability threshold")
        k = sum(s["unbounded"] or s["delta"] > threshold for s in scores)
        result.append({"threshold": threshold, "k": k, "N": n,
                       "fraction": k/n, "interval95": interval95(k,n),
                       "zero_event_upper95": -math.expm1(math.log(.05)/n) if k == 0 else None})
    return result


def _bound(root, record, inputs):
    if not isinstance(record, dict) or not {"path", "sha256"}.issubset(record):
        raise ValueError("Unbound artifact")
    path = Path(record["path"])
    if not path.is_absolute():
        path = root/path
    if not path.is_file() or digest(path) != record["sha256"]:
        raise ValueError(f"Artifact hash mismatch: {path}")
    if inputs is not None:
        inputs.add(path)
    return path


def validate_run_sources(run, inputs=None):
    """Recompute source instrumentation rather than trust a success flag."""
    required = {"original_interface", "instrumented_interface", "driver", "points",
                "parameter_card", "madloop_card", "executable", "build", "execution",
                "stdout", "stderr"}
    records = run.get("artifacts", {})
    if not required.issubset(records):
        raise ValueError("Incomplete DP producer/build/input artifact bindings")
    paths = {name:_bound(ROOT,record,inputs) for name,record in records.items()}
    instrumented, receipt = instrument_interface(paths["original_interface"].read_text())
    if paths["instrumented_interface"].read_text() != instrumented:
        raise ValueError("Private interface differs from exact observer transformation")
    expected_driver = render_driver(receipt["prefix"],int(run["nexternal"]),int(run["nso"]))
    if paths["driver"].read_text() != expected_driver:
        raise ValueError("Private DP driver differs from bound producer")
    with paths["executable"].open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise ValueError("DP executable is not an ELF build")
    tokens = paths["points"].read_text().split()
    if int(tokens[0]) != int(run["N"]) or len(tokens) != 1+int(run["N"])*int(run["nexternal"])*4:
        raise ValueError("DP point stream dimensions differ")
    if not all(_number(x) is not None for x in tokens[1:]):
        raise ValueError("Nonfinite point stream")
    execution = json.loads(paths["execution"].read_text())
    if execution.get("requested_tool") != run["tool"] or execution.get("helicity") != run["helicity"]:
        raise ValueError("Execution selector differs")
    if execution.get("thread_environment") != {
            "OMP_NUM_THREADS":"1", "OPENBLAS_NUM_THREADS":"1",
            "RAYON_NUM_THREADS":"1", "MKL_NUM_THREADS":"1"}:
        raise ValueError("DP execution thread environment is not bound")
    if not isinstance(execution.get("cpu"),int):
        raise ValueError("DP execution CPU is missing")
    build = json.loads(paths["build"].read_text())
    for field,artifact in [("driver_sha256","driver"),
            ("instrumented_interface_sha256","instrumented_interface"),
            ("executable_sha256","executable")]:
        if build.get(field) != records[artifact]["sha256"]:
            raise ValueError("DP compilation record differs from source/executable")
    return receipt, paths, execution


def report_snapshot(manifest_or_summary, expected=None, inputs=None):
    """Always re-read bound raw observations; a supplied summary is not authority."""
    path = Path(manifest_or_summary).resolve()
    data = json.loads(path.read_text())
    if data.get("schema") == SCHEMA:
        source = _bound(ROOT, data["manifest"], inputs)
        return report_snapshot(source, expected=expected, inputs=inputs)
    if data.get("schema") != RAW_SCHEMA:
        raise ValueError("Unsupported DP stability authority")
    if data.get("stream") != "forced_dp_self_consistency" or data.get("rescue_enabled") is not False:
        raise ValueError("Rescued or unspecified stream cannot supply DP self-consistency")
    if inputs is not None:
        inputs.add(path)
    if expected is not None:
        for key, value in expected.items():
            if data.get(key) != value:
                raise ValueError(f"DP manifest identity differs: {key}")
    for record in data.get("bindings", []):
        _bound(ROOT, record, inputs)
    if not data.get("bindings") or not data.get("cohort") or not data.get("instrumentation"):
        raise ValueError("DP authority lacks source/runtime/instrumentation bindings")
    required_cohort = {"numerical_revision", "generation_revision", "generator_identity_sha256",
                       "runtime_identity_sha256", "archive_sha256"}
    if not required_cohort.issubset(data["cohort"]):
        raise ValueError("DP authority lacks full cohort identities")
    cohort_validated = False
    if expected is not None and not data.get("cohort_record"):
        raise ValueError("An actual consumer requires a bound explicit cohort file")
    if data.get("cohort_record"):
        cohort_path = _bound(ROOT,data["cohort_record"],inputs)
        import ninjars_benchmark_policy_v3 as policy_v3
        retained = {}
        release = policy_v3.validate_cohort(cohort_path,retained)
        if any(data["cohort"][key] != release[key] for key in required_cohort):
            raise ValueError("Declared DP cohort differs from validated explicit cohort")
        for name,sha256 in retained.items():
            _bound(ROOT,{"path":name,"sha256":sha256},inputs)
        cohort_validated = True
    if data["instrumentation"] != "per-dispatch-observer-v1":
        raise ValueError("Unknown DP instrumentation")
    if not isinstance(data.get("runs"),list) or not data["runs"]:
        raise ValueError("DP authority has no observed runs")
    thresholds = data.get("thresholds", [10.**(-16+i/5) for i in range(81)])
    if not thresholds or thresholds != sorted(set(thresholds)):
        raise ValueError("DP threshold grid must be nonempty and strictly increasing")
    cases, all_complete = {}, True
    seen = set()
    for run in data["runs"]:
        case, tool, n, nso = run["case"], int(run["tool"]), int(run["N"]), int(run["nso"])
        if tool not in TOOLS or (case,tool) in seen or n <= 0:
            raise ValueError("Invalid or duplicate DP run")
        seen.add((case,tool))
        receipt, runpaths, execution = validate_run_sources(run,inputs)
        rawpath = _bound(ROOT, run["raw"], inputs)
        parsed = parse_raw(rawpath,n,tool,nso)
        if parsed["header"][4] != run["helicity"]:
            raise ValueError("Raw helicity differs from bound selector")
        if parsed["complete"] and execution.get("exit_code") != 0:
            raise ValueError("Completed DP raw stream has a failed process")
        all_complete = all_complete and parsed["complete"]
        groups = {}
        for name, indices in {"combined": (1,2,3), "direction": (1,2), "lorentz": (1,3)}.items():
            scores = []
            for point in range(1,n+1):
                observations = {t: parsed["observations"].get((point,t)) for t in range(1,4)}
                scores.append({"point": point, **estimator(observations,tool,indices)})
            groups[name] = {"point_estimators": scores,
                            "classification_counts": dict(Counter(s["classification"] for s in scores)),
                            "failure_categories": dict(Counter(s["classification"] for s in scores if s["unbounded"])),
                            "thresholds": tail(scores,thresholds)}
        entry = {"N": n, "tool": TOOLS[tool], "complete": parsed["complete"],
                 "status": "complete" if parsed["complete"] else "incomplete",
                 "raw": run["raw"], "contrasts": groups,
                 "scores": {name:g["point_estimators"] for name,g in groups.items()},
                 "failure_categories": groups["combined"]["failure_categories"],
                 "point_estimators": groups["combined"]["point_estimators"],
                 "thresholds": groups["combined"]["thresholds"],
                 "normalization": "unnormalized finite Born interference or loop-induced squared amplitude",
                 "estimator_scope": "forced DP; rescue disabled; per-dispatch rejection/FPE capture",
                 "classification_counts": groups["combined"]["classification_counts"],
                 "helicity": parsed["header"][4], "warm": parsed["warm"], "done": parsed["done"],
                 "point_stream": run["artifacts"]["points"], "instrumentation":receipt,
                 "execution":run["artifacts"]["execution"]}
        entry["hardware_exception_observations"] = {
            name:sum(bool(obs["metadata"][5+index]) for obs in parsed["observations"].values()
                     if obs["metadata"] is not None)
            for index,name in enumerate(COUNTER_NAMES) if index >= 15}
        group = cases.setdefault(case,{"id":case,"tools":{},"group":run.get("group","unspecified"),
                 "process":run.get("process",case),"sample_kind":run.get("sample_kind","fixed-core-pilot"),
                 "N":n,"helicity":run["helicity"]})
        if group["tools"] and any(v["N"] != n for v in group["tools"].values()):
            raise ValueError("Different point denominators for tools")
        if group["tools"] and any(v["point_stream"]["sha256"] != run["artifacts"]["points"]["sha256"] or
                v["helicity"] != run["helicity"] for v in group["tools"].values()):
            raise ValueError("Tools use different points or helicities")
        group["tools"][str(tool)] = entry
    for case in cases.values():
        if set(case["tools"]) != {str(t) for t in TOOLS}:
            all_complete = False
    return {"schema": SCHEMA, "manifest": {"path":str(path),"sha256":digest(path)},
            "stream":"forced_dp_self_consistency", "rescue_enabled":False,
            "requested_precision":"native-double", "thresholds":thresholds,
            "floating_point_rounding":"nearest observed before/after every call",
            "cohort": data["cohort"], "observations_complete": all_complete,
            "cohort_record":data.get("cohort_record"), "cohort_validated":cohort_validated,
            "complete": False, "physical_qualified": False,
            "physical_qualification_pending": "Independent actual cohort/runtime target binding and instrumented-process parity controls",
            "qualification": "diagnostic only; does not replace correctness or timing gates",
            "estimator": "(max(v)-min(v))/abs(mean(v)); failures count above every finite threshold",
            "interval_method": "Wilson two-sided 95%; exact one-sided 95% zero-event upper bound",
            "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    inspect = sub.add_parser("instrument")
    inspect.add_argument("source",type=Path)
    inspect.add_argument("destination",type=Path)
    inspect.add_argument("--receipt",type=Path,required=True)
    driver = sub.add_parser("driver")
    driver.add_argument("destination",type=Path)
    driver.add_argument("--prefix",required=True)
    driver.add_argument("--nexternal",type=int,required=True)
    driver.add_argument("--nso",type=int,required=True)
    control = sub.add_parser("recorder-control")
    control.add_argument("destination",type=Path)
    summary = sub.add_parser("summarize")
    summary.add_argument("manifest",type=Path)
    summary.add_argument("destination",type=Path)
    args = parser.parse_args()
    if args.command == "instrument":
        if args.destination.exists() or args.source.resolve()==args.destination.resolve():
            raise ValueError("Instrument only a new private destination")
        text, receipt = instrument_interface(args.source.read_text())
        args.destination.parent.mkdir(parents=True,exist_ok=True)
        args.destination.write_text(text)
        dump(args.receipt,receipt)
    elif args.command == "driver":
        args.destination.write_text(render_driver(args.prefix,args.nexternal,args.nso))
    elif args.command == "recorder-control":
        args.destination.write_text(recorder_control_source())
    else:
        dump(args.destination,report_snapshot(args.manifest))


if __name__ == "__main__":
    main()
