"""Qualify two version-type guards in a private module; no production edits."""
import difflib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
WORK = ROOT / '.tirust-build/strict-metadata-prototype'
SOURCE = ROOT / 'madgraph/various/tirust_install.py'
TESTS = ROOT / 'tests/unit_tests/various/test_tirust_install.py'
SOURCE_SHA = '352052637afbd82ab84f27508eda1bdfd9abc6115409adc23f8262e94c38d370'
TEST_SHA = 'e9c14988c4617aac52d8028db95b4122be0c86f651b43b4e48e69c04985f3f17'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    assert Path.cwd() == ROOT
    assert sha(SOURCE) == SOURCE_SHA and sha(TESTS) == TEST_SHA
    WORK.mkdir(exist_ok=False)
    original = SOURCE.read_text()
    fixed = original
    for old, new in [
        ("if not isinstance(config, dict) or config.get('abi_version') != ABI_VERSION:",
         "if (not isinstance(config, dict) or\n"
         "            type(config.get('abi_version')) is not int or\n"
         "            config.get('abi_version') != ABI_VERSION):"),
        ("if (config.get('link_order_version') != 1 or",
         "if (type(config.get('link_order_version')) is not int or\n"
         "            config.get('link_order_version') != 1 or")]:
        assert fixed.count(old) == 1
        fixed = fixed.replace(old, new)
    private = WORK / 'tirust_install.py'
    private.write_text(fixed)
    (HERE / 'implementation.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), fixed.splitlines(True),
        fromfile='a/madgraph/various/tirust_install.py',
        tofile='b/madgraph/various/tirust_install.py')))
    sys.dont_write_bytecode = True
    module = load('strict_metadata_private_installer', private)
    tests = load('strict_metadata_private_tests', TESTS)
    import madgraph.various
    stream = io.StringIO()
    # Any unexpected real installer subprocess is a test error. Existing
    # fixtures replace tool/build/smoke execution with their recording callback.
    with mock.patch.object(tests, 'tirust_install', module), \
            mock.patch.object(madgraph.various, 'tirust_install', module), \
            mock.patch.dict(sys.modules, {'madgraph.various.tirust_install': module}), \
            mock.patch.object(subprocess, 'Popen', side_effect=AssertionError('Unexpected subprocess')):
        suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        # Explicitly adopt the integer schema for JSON's integral-float spelling.
        fixture = tests.TestTIRustInstaller()
        fixture.setUp()
        try:
            fixture.assert_incompatible_metadata_rejected([
                ('abi_version', 1.0, 'Unsupported TIRust bridge ABI'),
                ('link_order_version', 1.0, 'Missing or invalid TIRust compiler-runtime link metadata')])
        finally:
            fixture.doCleanups()
    (HERE / 'tests.txt').write_text(stream.getvalue())
    assert sha(SOURCE) == SOURCE_SHA and sha(TESTS) == TEST_SHA
    record = {'tests_run': result.testsRun, 'failures': len(result.failures),
        'errors': len(result.errors), 'successful': result.wasSuccessful(),
        'additional_integral_float_rejections': 2,
        'production_modified': False, 'compiler_or_native_execution': False,
        'scope': 'Mocked installer fixtures; strict integer schema, not a real installation',
        'source_sha256': SOURCE_SHA, 'tests_source_sha256': TEST_SHA,
        'private_source_sha256': sha(private), 'runner_sha256': sha(Path(__file__)),
        'patch_sha256': sha(HERE / 'implementation.patch'),
        'test_log_sha256': sha(HERE / 'tests.txt')}
    (HERE / 'results.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print(json.dumps(record, sort_keys=True))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == '__main__':
    main()
