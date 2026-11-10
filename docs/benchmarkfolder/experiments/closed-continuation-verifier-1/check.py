#!/usr/bin/env python3
"""Pure closed-fixture checks; no native execution or writes to the repository."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def check(root):
    root = Path(root).resolve()
    verifier = module(HERE / "verify.py", "closed_fixture_verifier")
    bindings = json.loads((HERE / "bindings.json").read_text())
    outcomes = []
    n3_detail = None
    for path, expected in bindings["tested_checkpoints"].items():
        result = verifier.verify(root, path, expected)
        if result["cases"] == ["qcd_n3_uux_zgg"]:
            n3_detail = result
        outcomes.append({key: result[key] for key in (
            "cases", "index_sha256", "payloads_verified", "lanes_replayed",
            "regression_passed", "ordinary_target_passed", "native_executions")})
    n3 = next(row for row in outcomes if row["cases"] == ["qcd_n3_uux_zgg"])
    if not n3["regression_passed"] or n3["ordinary_target_passed"]:
        raise ValueError("The retained QCD n3 target misses were not preserved")
    tirust = next(row for row in n3_detail["lanes"]
                  if row["source"] == "candidate" and row["backend"] == 9)
    if tirust["accepted_backend_precision_counts"] != {"9:DP": 87, "9:QP": 13}:
        raise ValueError("The retained QCD n3 DP87/QP13 counts were not preserved")
    if any(row["ordinary_target_passed"] or row["ordinary_target_miss_count"] != 56
           for row in n3_detail["lanes"] if row["source"] == "candidate"
           and row["backend"] in (6, 7, 8, 9)):
        raise ValueError("The retained QCD n3 lane target misses were not preserved")
    path = next(name for name in bindings["tested_checkpoints"] if "/mixed_uux_ga/" in name)
    expected = bindings["tested_checkpoints"][path]
    checkpoint = json.loads((root / path).read_text())
    owned = HERE.relative_to(root)
    with tempfile.TemporaryDirectory(prefix="closed-verifier-portable-") as temporary:
        destination = Path(temporary)
        for name in [path, *checkpoint["files"], *[str(owned / n) for n in (
                "verify.py", "bindings.json", "README.md")]]:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
        portable = module(destination / owned / "verify.py", "portable_fixture_verifier")
        result = portable.verify(destination, path, expected)
        if not result["verified"] or result["lanes_replayed"] != 6:
            raise ValueError("Indexed-only temporary replay failed")
        try:
            portable.verify(destination, path, "0" * 64)
        except ValueError as error:
            if str(error) != "Checkpoint hash mismatch":
                raise
        else:
            raise ValueError("Wrong checkpoint hash was accepted")
        point = next(name for name in checkpoint["files"]
                     if name.endswith("/points/mixed_uux_ga.dat"))
        (destination / point).unlink()
        with tempfile.TemporaryDirectory(prefix="closed-verifier-outside-") as outside:
            target = Path(outside) / "point.dat"
            target.write_bytes((root / point).read_bytes())
            (destination / point).symlink_to(target)
            try:
                portable.verify(destination, path, expected)
            except ValueError as error:
                if str(error) != "Path escapes supplied repository":
                    raise
            else:
                raise ValueError("Escaping payload symlink was accepted")
    return {"cases": outcomes, "indexed_only_temporary_replay": True,
            "wrong_checkpoint_hash_rejected": True, "escaping_symlink_rejected": True,
            "qcd_n3_target_misses_preserved": True, "native_executions": 0,
            "qcd_n3_dp87_qp13_counts_preserved": True,
            "source_sha256": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                              for name in ("verify.py", "bindings.json", "README.md", "check.py")}}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check.py REPOSITORY_ROOT")
    print(json.dumps(check(sys.argv[1]), indent=2, sort_keys=True))
