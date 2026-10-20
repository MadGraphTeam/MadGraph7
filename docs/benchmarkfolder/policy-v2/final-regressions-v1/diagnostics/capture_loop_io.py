"""Diagnostic only: regenerate into a private comparison copy, never the fixtures."""
import difflib
import hashlib
import json
import pathlib
import shutil
import sys
import unittest

root = pathlib.Path.cwd()
sys.path.insert(0, str(root))
import tests.IOTests as io

out = root / '.benchmark-work/final-regression-review-v1'
original = root / 'tests/input_files/IOTestsComparison'
private = out / 'actual-comparisons'
if private.exists():
    raise SystemExit('Refuse to overwrite prior diagnostic output')
shutil.copytree(original, private)
io._hc_comparison_files = str(private)
io.IOTestManager._hc_comparison_files = str(private)
original_run = io.IOTestManager.runIOTests
calls = []
def capture(self, *args, **kwargs):
    kwargs.update(update=True, force=10, verbose=False)
    modifications = original_run(self, *args, **kwargs)
    calls.append({'test': self.id(), 'modifications': modifications})
    (out/'capture-calls.json').write_text(json.dumps(calls, indent=2)+'\n')
    return modifications
io.IOTestManager.runIOTests = capture
suite=unittest.defaultTestLoader.loadTestsFromName('tests.unit_tests.loop.test_loop_exporters')
result=unittest.TextTestRunner(verbosity=2).run(suite)
changed=[]
for path in sorted(private.rglob('*')):
    if not path.is_file() or path.name.endswith('.BackUp'):
        continue
    relative=path.relative_to(private)
    old=original/relative
    current=path.read_bytes()
    previous=old.read_bytes() if old.exists() else b''
    if previous==current:
        continue
    diff=''.join(difflib.unified_diff(previous.decode().splitlines(True),current.decode().splitlines(True),fromfile=str(relative),tofile=str(relative)))
    diffpath=out/'diffs'/relative.with_suffix(relative.suffix+'.diff')
    diffpath.parent.mkdir(parents=True,exist_ok=True)
    diffpath.write_text(diff)
    changed.append({'path':str(relative), 'kind':'changed' if old.exists() else 'new', 'old_sha256': hashlib.sha256(previous).hexdigest() if old.exists() else None, 'actual_sha256':hashlib.sha256(current).hexdigest(),'diff_lines':len(diff.splitlines())})
(out/'capture-summary.json').write_text(json.dumps({'diagnostic_only':True,'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'changed':changed},indent=2)+'\n')
print(json.dumps({'diagnostic_only':True,'changed_count':len(changed),'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures)},indent=2))
sys.exit(0 if result.wasSuccessful() else 1)
