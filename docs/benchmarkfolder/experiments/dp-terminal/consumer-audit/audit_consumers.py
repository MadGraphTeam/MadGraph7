#!/usr/bin/env python3
"""Source-only, fail-closed contract for the frozen QCD n=1 DP experiment.

This intentionally recognizes one immutable export and two reviewed helpers.
It neither invokes a compiler nor proves arbitrary Fortran programs safe.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex

FROZEN_INPUTS_SHA256 = "1dbf671783cfb56762b612140693d34c781e0c46cdef88096c9861a944fa41b4"
PROCESS = "SubProcesses/P0_uux_z"
CALLER = PROCESS + "/coef_construction_1.f"
POLYNOMIAL = PROCESS + "/polynomial.f"
HELPERS = {
    "a9644682d62de62dfd3f33c80b51fd916383c443701507cc6fc5c95c81abe95e": "elemental",
    "0f3af055f42e7fea0334b97fb0765451b9257e5cb0aec5ea48320043fb27a730": "packed",
}
SCOPE_DIRS = {"DHELAS": "Source/DHELAS", "MODEL": "Source/MODEL", "process": PROCESS}
SOURCE_SUFFIXES = {".f", ".for", ".f77", ".f90", ".f95", ".f03", ".f08",
                   ".inc", ".h", ".c", ".cc", ".cpp", ".s"}
LINKS = {
    "Source/coupl.inc": "MODEL/coupl.inc", "Source/ldme.inc": "MODEL/ldme.inc",
    **{PROCESS + "/" + name: "../" + name for name in (
        "MadLoopCommons.f", "MadLoopParamReader.f", "MadLoopParams.inc", "coupl.inc",
        "cts_mpc.h", "cts_mprec.h", "global_specs.inc", "makefile", "mg5_citation.f",
        "mp_coupl.inc", "mp_coupl_same_name.inc")},
    PROCESS + "/coef_specs.inc": "../../Source/DHELAS/coef_specs.inc",
    **{"SubProcesses/" + name: "../Source/MODEL/" + name for name in (
        "coupl.inc", "ldme.inc", "mp_coupl.inc", "mp_coupl_same_name.inc")},
}


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_pins(path):
    data = Path(path).read_bytes()
    require(sha(data) == FROZEN_INPUTS_SHA256, "Frozen candidate-inputs identity changed")
    value = json.loads(data)
    require(len(value["compilation"]["sources"]) == 111, "Unexpected frozen source count")
    return value


def relative_name(name):
    path = Path(name)
    require(not path.is_absolute() and ".." not in path.parts and "\\" not in name,
            "Unsafe source name: " + name)
    return path.as_posix()


def source_bytes(root, name):
    name = relative_name(name)
    path = root / name
    require(path.is_file(), "Missing source: " + name)
    require(path.resolve().is_relative_to(root), "Source escapes export: " + name)
    for parent in path.parents:
        if parent == root:
            break
        require(not parent.is_symlink(), "Aliased source directory: " + name)
    if path.is_symlink():
        require(name in LINKS and str(path.readlink()) == LINKS[name],
                "Unrecognized source symlink: " + name)
    return path.read_bytes()


def source_inventory(root, pins, transformed=None):
    """Reject extra source/include/build files, including files absent from recipes."""
    root = Path(root).resolve()
    expected = pins["compilation"]["sources"]
    for path in root.rglob("*"):
        name = path.relative_to(root).as_posix()
        if (path.is_symlink() and path.is_dir() and name.startswith(("Source/", "SubProcesses/"))
                and name != PROCESS + "/MadLoop5_resources"):
            raise AuditError("Aliased source directory: " + name)
        relevant = (path.suffix.lower() in SOURCE_SUFFIXES or
                    path.name.lower().startswith("makefile") or
                    path.name.lower() == "gnumakefile" or
                    path.name == "MadLoop_makefile_definitions" or path.name == "make_opts")
        if relevant:
            require(name in expected, "Unknown source/include/build input: " + name)
    content = {}
    for name, checksum in sorted(expected.items()):
        data = source_bytes(root, name)
        if transformed and name in transformed:
            require(data == transformed[name], "Unexpected transform delta: " + name)
        else:
            require(sha(data) == checksum, "Frozen source hash mismatch: " + name)
        content[name] = data
    return content


def compile_units(pins):
    """Enumerate frozen object recipes, independently cross-checking their argv."""
    units = {}
    scopes = pins["compilation"]["effective_compiler_flags"]["scopes"]
    require(set(scopes) == set(SCOPE_DIRS), "Unknown compiler scope")
    for scope, directory in SCOPE_DIRS.items():
        recipes = scopes[scope]["recipes"]
        found = {}
        for line in scopes[scope]["expanded_make_output"].splitlines():
            args = shlex.split(line)
            if "-c" not in args:
                require(not any(Path(word).suffix.lower() in {".f", ".f90"} for word in args),
                        "Unrecognized compile recipe: " + line)
                continue
            names = [word for word in args if Path(word).suffix.lower() in {".f", ".f90"}]
            require(len(names) == 1 and names[0] in recipes, "Unknown compiled unit")
            name = names[0]
            require(name not in found and "/" not in name, "Duplicate or aliased compiled unit")
            includes = []
            index = 1
            while index < len(args):
                word = args[index]
                if word == "-I":
                    index += 1
                    includes.append(args[index])
                elif word.startswith("-I"):
                    includes.append(word[2:])
                index += 1
            found[name] = {"scope": scope, "directory": directory,
                           "include_dirs": includes, "argv": args}
        require(set(found) == set(recipes), "Compile recipe/source enumeration differs")
        for name, record in found.items():
            relative = directory + "/" + name
            require(relative in pins["compilation"]["sources"], "Compiled unit lacks a source pin")
            units[relative] = record
    require(len(units) == 52, "Unexpected compiled recipe unit count")
    return units


def actual_compile_units(pins, units, log_directory):
    actual, logs = {}, {}
    for scope, directory in SCOPE_DIRS.items():
        path = Path(log_directory) / ("compilation-" + scope + ".log")
        data = path.read_bytes()
        require(sha(data) == pins["compilation"]["build_logs"][scope], "Frozen build log changed: " + scope)
        logs[scope] = sha(data)
        for line in data.decode().splitlines():
            if not re.search(r"(?:^|\s)-c(?:\s|$)", line):
                continue
            args = shlex.split(line)
            if "-c" not in args:
                continue
            names = [word for word in args if Path(word).suffix.lower() in {".f", ".f90"}]
            require(len(names) == 1, "Unrecognized actual compile command")
            name = directory + "/" + names[0]
            require(name in units and args == units[name]["argv"], "Actual compile command differs from frozen recipe")
            require(name not in actual, "Repeated actual compilation")
            actual[name] = units[name]
    require(len(actual) == 49, "Unexpected actual compiled unit count")
    return actual, logs


def strip_comment(text):
    quote = None
    index = 0
    while index < len(text):
        char = text[index]
        if quote:
            if char == quote:
                if index + 1 < len(text) and text[index + 1] == quote:
                    index += 1
                else:
                    quote = None
        elif char in "'\"":
            quote = char
        elif char == "!":
            return text[:index]
        index += 1
    return text


def fixed_line(raw):
    if raw.startswith("\t"):
        return ("     $" + raw[2:] if raw[1:2] and raw[1:2] in "123456789"
                else "      " + raw[1:])
    return raw


def logical_rows(physical):
    result = []
    for name, number, raw in physical:
        if not raw.strip() or raw[:1] in "cC*!":
            continue
        require(not raw.startswith("#"), "Preprocessed source is outside this contract: " + name)
        raw = fixed_line(raw)
        require(len(raw) >= 6 and "\t" not in raw[:6], "Unrecognized fixed-form source: " + name)
        line = strip_comment(raw[6:132]).strip()
        if not line:
            continue
        if raw[5] not in " 0":
            require(bool(result), "Orphan continuation: " + name)
            result[-1][2] += line
        else:
            result.append([name, number, line])
    return result


def statements(data, name):
    return logical_rows([(name, number, raw) for number, raw in enumerate(data.decode().splitlines(), 1)])


def expanded_units(content, units):
    """Resolve every INCLUDE in each frozen compile context; no external include admitted."""
    result, edges = {}, []
    for unit, recipe in sorted(units.items()):
        def expand(name, stack):
            require(name not in stack, "Cyclic INCLUDE: " + name)
            output = []
            for line, raw in enumerate(content[name].decode().splitlines(), 1):
                origin = name
                text = strip_comment(fixed_line(raw)[6:132]).strip() if raw[:1] not in "cC*!" else ""
                match = re.fullmatch(r"INCLUDE\s*['\"]([^'\"]+)['\"]", text, re.I)
                if match:
                    target = match[1]
                    require(not Path(target).is_absolute() and "\\" not in target,
                            "Unsafe INCLUDE: " + target)
                    directories = [str(Path(name).parent), recipe["directory"]]
                    directories.extend(os.path.normpath(str(Path(recipe["directory"]) / folder))
                                       for folder in recipe["include_dirs"] if not Path(folder).is_absolute())
                    choices = [os.path.normpath(folder + "/" + target) for folder in directories]
                    require(all(not Path(choice).is_absolute() and ".." not in Path(choice).parts
                                for choice in choices), "INCLUDE escapes export: " + target)
                    selected = next((choice for choice in choices if choice in content), None)
                    require(selected is not None, "Unpinned or external INCLUDE: " + target)
                    edges.append({"unit": unit, "from": origin, "line": line, "include": selected})
                    output.extend(expand(selected, stack + [name]))
                else:
                    require(not re.match(r"INCLUDE\b", text, re.I), "Unrecognized INCLUDE syntax")
                    output.append([origin, line, raw])
            return output
        result[unit] = logical_rows(expand(unit, []))
    return result, edges


def compact(text):
    return re.sub(r"\s+", "", text.upper())


def validate_consumers(expanded):
    """Prove the known DP block and its one seed-only alias from expanded code."""
    expected_blocks = {PROCESS + "/" + name: alias for name, alias in (
        ("coef_construction_1.f", "WL"), ("helas_calls_ampb_1.f", "WL"),
        ("loop_CT_calls_1.f", "WL"), ("loop_matrix.f", "WL"),
        ("mp_compute_loop_coefs.f", "DP_WL"))}
    seen, observations = {}, []
    expected_calls = [
        "CALLML5_0_UPDATE_WL_0_0(WL(1,0,1,0),4,COEFS,4,4,WL(1,0,1,1))",
        "CALLML5_0_UPDATE_WL_0_1(WL(1,0,1,1),4,COEFS,4,4,WL(1,0,1,2))",
        "CALLML5_0_UPDATE_WL_1_1(WL(1,0,1,2),4,COEFS,4,4,WL(1,0,1,3))",
        "CALLML5_0_CREATE_LOOP_COEFS(WL(1,0,1,3),2,4,1,1,1,2,H)",
    ]
    for unit, rows in expanded.items():
        block_rows = [row for row in rows if "/ML5_0_WL/" in compact(row[2])]
        if not block_rows:
            continue
        require(unit in expected_blocks and len(block_rows) == 1, "Unknown or repeated DP COMMON consumer: " + unit)
        alias = expected_blocks[unit]
        common = "COMMON/ML5_0_WL/" + alias + (",DP_PL" if alias == "DP_WL" else ",PL")
        require(block_rows[0][0] == unit and compact(block_rows[0][2]) == common,
                "Unknown DP COMMON alias/layout: " + unit)
        declaration = "COMPLEX*16" + alias + "(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE,-1:NLOOPWAVEFUNCS)"
        momentum = "DP_PL" if alias == "DP_WL" else "PL"
        momentum_declaration = "COMPLEX*16" + momentum + "(0:3,-1:NLOOPWAVEFUNCS)"
        require(sum(compact(row[2]) == declaration and row[0] == unit for row in rows) == 1,
                "DP COMMON wavefunction declaration changed: " + unit)
        require(sum(compact(row[2]) == momentum_declaration and row[0] == unit for row in rows) == 1,
                "DP COMMON momentum declaration changed: " + unit)
        uses = []
        for origin, line, text in rows:
            normalized = compact(text)
            if not re.search(r"\b" + alias + r"\b", text.upper()):
                continue
            require(origin == unit, "Included DP COMMON consumer: " + origin)
            if normalized in (common, declaration):
                continue
            uses.append(normalized)
            observations.append({"unit": unit, "line": line, "alias": alias, "statement": normalized})
        if unit == CALLER:
            require(uses == expected_calls, "DP update/terminal consumer graph changed")
        else:
            seeds = {alias + suffix for suffix in (
                "(I,J,K,-1)=(0.0D0,0.0D0)", "(I,J,K,0)=(1.0D0,0.0D0)",
                "(I,J,K,0)=(0.0D0,0.0D0)", "(1,0,2,-1)=DCMPLX(-1.0D0,0.0D0)",
                "(2,0,1,-1)=DCMPLX(1.0D0,0.0D0)", "(3,0,4,-1)=DCMPLX(1.0D0,0.0D0)",
                "(4,0,3,-1)=DCMPLX(-1.0D0,0.0D0)")}
            require(all(use in seeds for use in uses), "Non-seed or escaped DP COMMON use: " + unit)
            if unit.endswith(("/loop_matrix.f", "/mp_compute_loop_coefs.f")):
                require(len(uses) == 7 and set(uses) == seeds, "Seed initialization contract changed")
            else:
                require(not uses, "Unexpected DP COMMON use: " + unit)
        seen[unit] = alias
    require(seen == expected_blocks, "Expected DP COMMON declaration is missing")
    polynomial = [compact(row[2]) for row in expanded[POLYNOMIAL]]
    def body(name):
        starts = [index for index, line in enumerate(polynomial) if line.startswith("SUBROUTINE" + name + "(")]
        require(len(starts) == 1, "Missing or repeated merger routine")
        start = starts[0]
        end = polynomial.index("END", start)
        return polynomial[start:end + 1]
    merger = body("ML5_0_MERGE_WL")
    require(merger == [
        "SUBROUTINEML5_0_MERGE_WL(WL,R,LCUT_SIZE,CONST,OUT)",
        "USEML5_0_POLYNOMIAL_CONSTANTS", "INTEGERI,J",
        "COMPLEX*16WL(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)", "INTEGERR,LCUT_SIZE",
        "COMPLEX*16CONST", "COMPLEX*16OUT(0:LOOPMAXCOEFS-1)", "DOI=1,LCUT_SIZE",
        "DOJ=0,NCOEF_R(R)-1", "OUT(J)=OUT(J)+WL(I,J,I)*CONST", "ENDDO", "ENDDO", "END"],
        "Diagonal merger implementation changed")
    create = body("ML5_0_CREATE_LOOP_COEFS")
    uses = [line for line in create if "LOOP_WF" in line]
    require(len(uses) == 3 and uses[-1].startswith("CALLML5_0_MERGE_WL(LOOP_WF,RANK,LCUT_SIZE,CONST(I),"),
            "CREATE_LOOP_COEFS has an unknown wavefunction consumer")
    return {"terminal_wavefunction": 3, "producer": 2, "shape": [4, 4, 4],
            "rank": [1, 1], "common_aliases": seen, "executable_uses": observations,
            "merger_statements_sha256": sha("\n".join(merger).encode())}


def once(text, old, new):
    require(text.count(old) == 1, "Expected unique transform anchor: " + repr(old))
    return text.replace(old, new)


def expected_terminal_caller(data, trace_admission=False):
    text = data.decode()
    text = once(text, "      USE ALOHA_OBJECT\n", "      USE ALOHA_OBJECT\n"
                "      USE, INTRINSIC :: IEEE_EXCEPTIONS, ONLY: IEEE_GET_FLAG,\n"
                "     $ IEEE_INEXACT\n")
    text = once(text, "      INTEGER I,J,K\n", "      INTEGER I,J,K\n"
                "      LOGICAL DP_TERMINAL_ENTRY_INEXACT,DP_TERMINAL_FAST\n")
    text = once(text, "      CALL ML5_0_UPDATE_WL_1_1(WL(1,0,1,2),4,COEFS,4,4,WL(1,0,1,3))\n",
                "      CALL IEEE_GET_FLAG(IEEE_INEXACT,DP_TERMINAL_ENTRY_INEXACT)\n"
                "      CALL ML5_0_DP_TERMINAL_1_1(WL(1,0,1,2),4,COEFS,4,4,\n"
                "     $ WL(1,0,1,3),DP_TERMINAL_ENTRY_INEXACT,DP_TERMINAL_FAST)\n")
    if trace_admission:
        anchor = "      CALL ML5_0_CREATE_LOOP_COEFS(WL(1,0,1,3),2,4,1,1,1,2,H)\n"
        text = once(text, anchor, "      WRITE(*,*) 'DP_TERMINAL_ADMISSION',\n"
                    "     $ DP_TERMINAL_ENTRY_INEXACT,DP_TERMINAL_FAST\n" + anchor)
    return text.encode()


def expected_helper(data):
    require(sha(data) in HELPERS, "Unreviewed terminal helper identity")
    text = once(data.decode(), "subroutine DP_TERMINAL(", "subroutine ML5_0_DP_TERMINAL_1_1(")
    text = once(text, "call DP_DENSE(", "call ML5_0_UPDATE_WL_1_1(")
    output = []
    for line in text.splitlines():
        if line.lstrip().startswith("!"):
            output.append("C " + line.lstrip()[1:])
        elif line.strip():
            label = re.match(r"^(\d+)\s+(.*)$", line)
            prefix = "      "
            if label:
                prefix, line = label[1].rjust(5) + " ", label[2]
            output.append(prefix + line[:66])
            line = line[66:]
            while line:
                output.append("     $" + line[:66])
                line = line[66:]
    return ("\n".join(output) + "\n").encode()


def audit(baseline_root, candidate_inputs_path, transformed_root=None,
          helper_path=None, helper_sha256=None, build_logs_dir=None, trace_admission=False):
    pins = load_pins(candidate_inputs_path)
    baseline = source_inventory(baseline_root, pins)
    recipes = compile_units(pins)
    if build_logs_dir is None:
        build_logs_dir = Path(candidate_inputs_path).resolve().parents[3] / "inputs/candidate/qcd_n1_uux_z"
    units, logs = actual_compile_units(pins, recipes, build_logs_dir)
    expanded, includes = expanded_units(baseline, units)
    consumers = validate_consumers(expanded)
    closure = sorted(set(units) | {edge["include"] for edge in includes})
    base = Path(baseline_root).resolve()
    resolved = {name: (base / name).resolve().relative_to(base).as_posix() for name in closure}
    result = {
        "schema": "madloop-dp-terminal-consumer-audit-v1", "passed": True,
        "scope": "Frozen QCD n=1 generated-source and exact transform proof only",
        "native_execution": False, "performance_eligible": False,
        "frozen_candidate_inputs_sha256": FROZEN_INPUTS_SHA256,
        "source_count": len(baseline), "compile_recipe_unit_count": len(recipes),
        "actual_compiled_unit_count": len(units), "build_logs_sha256": logs,
        "uncompiled_recipe_units": sorted(set(recipes) - set(units)),
        "compile_units": units, "include_edges": includes,
        "include_file_count": len({edge["include"] for edge in includes}),
        "distinct_include_edge_count": len({(edge["unit"], edge["from"], edge["include"]) for edge in includes}),
        "textual_closure_count": len(closure), "resolved_closure_count": len(set(resolved.values())),
        "logical_to_resolved_closure": resolved,
        "external_module_boundary": ["COLLIER", "MNINJA", "MNINJARS", "MPMODULE", "MTIRUST"],
        "source_sha256": {name: sha(data) for name, data in baseline.items()},
        "consumer_proof": consumers, "transformed": False,
        "limitations": ["Compile units are enumerated from hash-verified historical build logs; no compiler is run.",
                        "External USE modules and their .mod files are outside the textual source closure.",
                        "Native linkage and external archive identity are separate build-proof obligations.",
                        "Source identity pins intentionally reject arbitrary/custom exports."]}
    if transformed_root is not None:
        require(helper_path is not None and helper_sha256 is not None, "Transformed audit requires pinned helper")
        helper = Path(helper_path).read_bytes()
        require(sha(helper) == helper_sha256, "Helper hash differs from requested identity")
        changes = {CALLER: expected_terminal_caller(baseline[CALLER], trace_admission),
                   POLYNOMIAL: baseline[POLYNOMIAL] + b"\n" + expected_helper(helper)}
        transformed = source_inventory(transformed_root, pins, changes)
        # Exact byte deltas preserve every old routine and all old includes.
        # The reviewed helper has no INCLUDE or COMMON statement.
        require(not re.search(rb"(?im)^\s*(include|common)\b", helper), "Helper introduces hidden storage/input")
        result.update(transformed=True, helper_variant=HELPERS[sha(helper)],
                      admission_trace=trace_admission,
                      trace_is_diagnostic_only=trace_admission,
                      helper_sha256=sha(helper),
                      changed_sources={name: sha(transformed[name]) for name in changes},
                      original_merger_and_create_bytes_unchanged=True,
                      entry_flag_capture_immediately_before_terminal_call=True)
    else:
        require(helper_path is None and helper_sha256 is None, "Helper supplied without transformed root")
        require(not trace_admission, "Trace supplied without transformed root")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-inputs", type=Path, required=True)
    parser.add_argument("--transformed-root", type=Path)
    parser.add_argument("--helper", type=Path)
    parser.add_argument("--helper-sha256")
    parser.add_argument("--build-logs-dir", type=Path)
    parser.add_argument("--trace-admission", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.baseline_root, args.candidate_inputs, args.transformed_root,
                   args.helper, args.helper_sha256, args.build_logs_dir, args.trace_admission)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
