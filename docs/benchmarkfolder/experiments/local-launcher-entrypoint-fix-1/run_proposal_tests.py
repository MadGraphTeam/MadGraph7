"""Exercise proposed replacements without editing/importing them as production."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import types
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "tools"))


def load_tool(path):
    spec = importlib.util.spec_from_file_location("tirust_benchmark_local", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_tests(tool):
    sys.modules["tirust_benchmark_local"] = tool
    name = "tests/unit_tests/loop/test_tirust_benchmark_local.py"
    module = types.ModuleType("proposal_local_launcher_tests")
    # Preserve the test's real checkout ROOT while executing proposed bytes.
    module.__file__ = str(ROOT / name)
    exec(compile((HERE / "proposed" / name).read_text(), module.__file__, "exec"), module.__dict__)
    return module


def focused_suite(module):
    names = [
        "test_install_records_real_cli_command_and_verifiable_source_receipt",
        "test_failed_cli_install_retains_failure_without_receipt",
        "test_missing_cli_retains_exact_intent_and_failure_without_receipt",
        "test_cli_exit_zero_without_success_marker_has_no_receipt",
        "test_source_change_during_cli_install_has_no_receipt",
    ]
    suite = unittest.TestSuite(module.TestLocalMeasurement(name) for name in names)
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(module.TestLocalInstallEntrypoint))
    return suite


def run_handoff(tool):
    path = HERE / "candidate-timing-handoff-4/test_handoff.py"
    spec = importlib.util.spec_from_file_location("proposal_timing_handoff4_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_digest = module.digest
    replacements = {
        ROOT / name: HERE / "proposed" / name
        for name in ("tools/tirust_benchmark_local.py", "tests/unit_tests/loop/test_tirust_benchmark_local.py")
    }
    # Only the two explicitly proposed source replacements are mapped. All
    # canonical inputs, historical packet files and other tools are read as-is.
    module.digest = lambda path: original_digest(replacements.get(Path(path), path))
    return unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))


def main():
    baseline = load_tests(load_tool(ROOT / "tools/tirust_benchmark_local.py"))
    output = io.StringIO()
    with contextlib.redirect_stderr(output):
        negative = unittest.TextTestRunner(stream=output, verbosity=2).run(unittest.TestSuite([
            baseline.TestLocalMeasurement("test_install_records_real_cli_command_and_verifiable_source_receipt")]))
    if negative.testsRun != 1 or len(negative.failures) != 1 or negative.errors:
        print(output.getvalue())
        raise SystemExit("Baseline must fail the new exact-argv regression test exactly once")
    print("Baseline negative control: existing launcher rejected by exact-argv test (expected).", flush=True)
    tool = load_tool(HERE / "proposed/tools/tirust_benchmark_local.py")
    result = unittest.TextTestRunner(verbosity=2).run(focused_suite(load_tests(tool)))
    assessment = {"candidate_tests_run": result.testsRun, "candidate_failures": len(result.failures),
                  "candidate_errors": len(result.errors), "candidate_skipped": len(result.skipped),
                  "baseline_regression_detected": True,
                  "native_installation_or_compilation_or_amplitude": False}
    if "--handoff" in sys.argv:
        handoff = run_handoff(tool)
        assessment["handoff_tests_run"] = handoff.testsRun
        assessment["handoff_failures"] = len(handoff.failures)
        assessment["handoff_errors"] = len(handoff.errors)
        assessment["handoff_skipped"] = len(handoff.skipped)
    else:
        handoff = None
    print(json.dumps(assessment, sort_keys=True), flush=True)
    if not result.wasSuccessful() or (handoff and not handoff.wasSuccessful()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
