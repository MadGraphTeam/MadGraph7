#!/usr/bin/env python3
"""Relocate this capsule and replay in isolated Python with external reads denied."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from portable_inputs import verify_payloads


HERE = Path(__file__).resolve().parent
CHILD = r'''
import collections, difflib, gzip, hashlib, itertools, json, math
import os, pathlib, re, sys, types, zlib
root = pathlib.Path(sys.argv[1]).resolve()
sys.dont_write_bytecode = True
sys.path.insert(0, str(root))
denied = []
def guard(event, args):
    if event != "open":
        return
    path, mode, flags = args
    if isinstance(path, int):
        raise PermissionError("unbound file descriptor open")
    path = pathlib.Path(os.fsdecode(path)).absolute()
    if (path.parent != root or path.is_symlink()
            or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
        denied.append(str(path))
        raise PermissionError("external or writable open forbidden during portable replay")
sys.addaudithook(guard)
# Demonstrate that the old workspace and an arbitrary external read are blocked.
for path in [sys.argv[2], "/etc/passwd"]:
    try:
        pathlib.Path(path).read_bytes()
    except PermissionError:
        pass
    else:
        raise AssertionError("read isolation did not block " + path)
assert len(denied) == 2
for name, filename in [("portable_inputs", "portable_inputs.py"), ("portable_verify", "verify.py")]:
    module = types.ModuleType(name)
    module.__file__ = str(root / filename)
    sys.modules[name] = module
    exec(compile((root / filename).read_bytes(), module.__file__, "exec"), module.__dict__)
result = module.verify(root)
assert len(denied) == 2, denied
result.update({"clean_relocation": True, "all_external_file_reads_forbidden": True,
               "intentional_denied_read_probes": 2, "unexpected_denied_reads": 0,
               "writes_forbidden_during_replay": True, "isolated_python": True})
print(json.dumps(result, sort_keys=True, indent=2))
'''


def main():
    index = verify_payloads(HERE)
    with tempfile.TemporaryDirectory(prefix="closed-qcdn3-portable-") as temporary:
        target = Path(temporary) / "capsule"
        target.mkdir()
        for name in ["index.json", *index["files"]]:
            shutil.copyfile(HERE / name, target / name)
        original = HERE.parent / "attempt1" / "analysis.json"
        result = subprocess.run([sys.executable, "-I", "-B", "-c", CHILD,
                                 str(target), str(original)], cwd=target,
                                env={"PATH": os.environ.get("PATH", "")},
                                text=True, capture_output=True, timeout=30, check=True)
        assert not result.stderr, result.stderr
        parsed = json.loads(result.stdout)
        assert parsed["passed"] is True
        print(json.dumps(parsed, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
