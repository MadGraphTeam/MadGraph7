#!/usr/bin/env python3
"""Run this actual relocated-export check, restoring its private prefix finally."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import datetime

sys.dont_write_bytecode = True
import runtime_validator as runtime

ROOT, REPORT, WORK = runtime.ROOT, runtime.REPORT, runtime.WORK


def image(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(directory.rglob("*")) if p.is_file()}


def main():
    os.sched_setaffinity(0, {190})
    original = WORK / "source/HEPTools/ninjars"
    hidden = original.with_name("ninjars-unavailable")
    installation = json.loads((REPORT / "installation.json").read_text())
    assert original.is_dir() and not hidden.exists()
    assert image(original) == installation["installed_file_sha256"]
    command = [sys.executable, "-B", str(REPORT / "runtime_validator.py"),
               "--hide-installation", "--run"]
    result = None
    try:
        with (REPORT / "relocated-runtime-controller.stdout").open("x") as out, \
                (REPORT / "relocated-runtime-controller.stderr").open("x") as err:
            result = subprocess.run(command, cwd=ROOT, env=runtime.environment(),
                                    stdout=out, stderr=err)
    finally:
        if hidden.is_dir() and not original.exists():
            hidden.rename(original)
        actual = image(original)
        restored = original.is_dir() and not hidden.exists() and actual == installation["installed_file_sha256"]
        record = {"schema": "final-pins-installation-restoration-v1",
                  "restored_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "original_path": str(original.relative_to(ROOT)),
                  "hidden_path_absent": not hidden.exists(), "restored": restored,
                  "installed_file_sha256": actual, "runtime_command": command,
                  "runtime_exit_code": result.returncode if result is not None else None}
        with (REPORT / "installation-restored.json").open("x") as stream:
            json.dump(record, stream, indent=2, sort_keys=True); stream.write("\n")
        if not restored:
            raise RuntimeError("The private installation did not restore byte-exactly")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
