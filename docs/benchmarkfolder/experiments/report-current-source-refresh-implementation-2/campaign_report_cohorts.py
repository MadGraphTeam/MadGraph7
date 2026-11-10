"""Portable raw cohort reparse, separate from ordinary/acceptance timing."""
import ast
import importlib.util
import json
from pathlib import Path
import types


def helper():
    spec = importlib.util.spec_from_file_location("campaign_report_pairs", Path(__file__).with_name("campaign_report_pairs.py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def card_settings(path, expected):
    """Parse explicit saved MadLoop values without importing a live model."""
    result, name, ended = {}, None, False
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("!"):
            continue
        if ended:
            raise ValueError("Unexpected content after numerical-card terminator")
        if line == "/* End of param file */":
            if name is not None:
                raise ValueError("Numerical-card terminator interrupts a value")
            ended = True
            continue
        if line.startswith("#"):
            name = line[1:].strip().lower()
            if name in result or name not in expected:
                raise ValueError("Duplicate or unknown saved numerical parameter")
            continue
        if name is None:
            raise ValueError("Numerical value lacks a parameter name")
        prototype = expected[name]
        if isinstance(prototype, bool):
            if line.lower() not in (".true.", ".false."):
                raise ValueError("Invalid logical parameter")
            result[name] = line.lower() == ".true."
        elif isinstance(prototype, int):
            result[name] = int(line)
        elif isinstance(prototype, float):
            result[name] = float(line.lower().replace("d", "e"))
        else:
            result[name] = line
        name = None
    if name is not None or set(result) != set(expected):
        raise ValueError("Incomplete saved numerical card")
    return result


def reparse(evidence, directory, manifest):
    pairs = helper()
    spec = evidence.json(directory + "/specification.json")
    binding = {"id": manifest["id"], "manifest_sha256": pairs.sha(evidence.add("campaigns/" + manifest["id"] + "/manifest.json"))}
    if spec.get("campaign_binding") != binding or spec.get("trace_schema") != 2:
        raise ValueError("Cohort campaign/observed-flag schema differs")
    if evidence.exists(directory + "/failure.json"):
        failure = evidence.json(directory + "/failure.json")
        if failure.get("campaign_binding") != binding:
            raise ValueError("Cohort failure has no matching campaign identity")
        return {"status": "failed", "failure": failure, "cohorts": {}}
    if not evidence.exists(directory + "/results.json"):
        return {"status": "pending", "cohorts": {}}
    inputs = evidence.json(directory + "/inputs.json")
    compiled = evidence.json(directory + "/compilation.json")
    baseline_record = evidence.json("campaigns/" + manifest["id"] + "/inputs/baseline.json",
                                    manifest["baseline"]["record_sha256"])
    with pairs.retained_protocol(evidence, directory, manifest) as (bench, sources):
        bench.campaign_guard = lambda stage, inputs=None: pairs.validate_identity(
            {**inputs, "compilation": compiled["original"]}, manifest, binding, baseline_record) if inputs is not None else None
        source = sources.get("tirust_benchmark_cohorts.py")
        if source is None:
            raise ValueError("Cohort parser source is absent")
        tree = ast.parse(source)
        class PureCard(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                return None if node.module == "madgraph.various.banner" else node
        tree = PureCard().visit(tree)
        tree.body = [node for node in tree.body if isinstance(node,
            (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign))]
        module = types.ModuleType("retained_cohort_report")
        module.__file__ = str(evidence.path(directory + "/tirust_benchmark_cohorts.py"))
        module.MadLoopParam = lambda path: card_settings(path, inputs["settings"])
        exec(compile(tree, module.__file__, "exec"), module.__dict__)
        # A frozen directory view ignores later append-only sibling batches.
        module.Path = pairs.snapshot_paths(evidence)
        result = module.validate_report(module.Path(evidence.path(directory)))
    return result


def collect_cohorts(evidence, campaign_dir, manifest, config):
    prefix = campaign_dir + "/experiments/cohorts"
    paths = evidence.tree(prefix) if evidence.exists(prefix) else []
    directories = sorted({str(Path(path).parent) for path in paths if path.endswith("/specification.json")})
    groups = {}
    for directory in directories:
        spec = evidence.json(directory + "/specification.json")
        key = (spec["case"]["id"], spec["source"], spec["backend"])
        case = next((row for row in config["cases"] if row["id"] == key[0]), None)
        if case != spec["case"] or key[1] not in ("baseline", "candidate") or key[2] not in config["backends"]:
            raise ValueError("Cohort refers to an unconfigured source/process/backend")
        try:
            value = reparse(evidence, directory, manifest)
        except (ValueError, KeyError, OSError, RuntimeError) as error:
            value = {"status": "failed", "error": str(error), "cohorts": {}}
        groups.setdefault(key, []).append({"directory": directory, **value})
    rows = []
    for case in config["cases"]:
        for backend in config["backends"]:
            attempts = groups.get((case["id"], "candidate", backend), [])
            status = "pending"
            if attempts:
                passed = []
                for attempt in attempts:
                    cohorts = attempt.get("cohorts", {})
                    passed.append(attempt["status"] == "diagnostic"
                        and all(name in cohorts for name in ("dp", "compensated_dp", "qp", "failed"))
                        and not cohorts["failed"]["indices"]
                        and all(not row["indices"] or row.get("eligible_diagnostic") is True
                                for name, row in cohorts.items() if name != "failed"))
                status = "passed" if all(passed) else "failed" if any(
                    a["status"] == "failed" or (a["status"] == "diagnostic" and not passed[index])
                    for index, a in enumerate(attempts)) else "pending"
            rows.append({"case": case["id"], "source": "candidate", "backend": backend,
                         "status": status, "attempts": attempts})
    return {"status": "passed" if all(r["status"] == "passed" for r in rows) else
            "failed" if any(r["status"] == "failed" for r in rows) else "pending",
            "rows": rows, "baseline_diagnostics": [
                {"case": key[0], "backend": key[2], "attempts": value}
                for key, value in groups.items() if key[1] == "baseline"],
            "performance_acceptance_eligible": False,
            "scope": "Traced full-amplitude subset diagnostics. DP includes internal method selection; compensated DP and naturally rescued QP are separate. Per-call tracing adds overhead. Different frozen subsets cannot be used as backend speed ratios. The diagnostic 90% duration rule is distinct from the main paired 95% rule."}
