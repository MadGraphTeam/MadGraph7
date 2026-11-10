"""Pure filesystem preflight proposal; no loader, compiler or native execution.

The caller supplies independently pinned SONAME/canonical-path/hash records.
The helper only inspects an already constructed private shim directory.
"""
import hashlib
import os
from pathlib import Path
import stat


def validate_shim(directory, pinned):
    directory = Path(directory)
    if not directory.is_absolute() or directory.is_symlink() or not directory.is_dir():
        raise ValueError("Shim directory must be a literal absolute directory")
    if not pinned:
        raise ValueError("No pinned providers")
    canonical_targets = set()
    for soname, record in pinned.items():
        if not isinstance(soname, str) or soname in ("", ".", "..") or Path(soname).name != soname:
            raise ValueError("SONAME must be one literal basename")
        if set(record) != {"canonical_path", "sha256"}:
            raise ValueError("Unexpected provider metadata")
        target = Path(record["canonical_path"])
        if not target.is_absolute() or target.is_symlink() or target.resolve(strict=True) != target:
            raise ValueError("Target must be an exact absolute canonical file")
        if not target.is_file():
            raise ValueError("Provider is not a regular file")
        if target in canonical_targets:
            raise ValueError("Provider canonical paths must be unique")
        canonical_targets.add(target)
        wanted_hash = record["sha256"]
        if not isinstance(wanted_hash, str) or len(wanted_hash) != 64 or any(c not in "0123456789abcdef" for c in wanted_hash):
            raise ValueError("Invalid pinned SHA-256")
    if {entry.name for entry in directory.iterdir()} != set(pinned):
        raise ValueError("Shim entries differ from the exact declared SONAME set")
    checked = []
    for soname, record in sorted(pinned.items()):
        alias = directory / soname
        if not stat.S_ISLNK(alias.lstat().st_mode):
            raise ValueError("Every SONAME alias must be a symlink")
        literal_target = os.readlink(alias)
        if literal_target != record["canonical_path"]:
            raise ValueError("Alias target is not the exact canonical pinned provider")
        if str(alias.resolve(strict=True)) != record["canonical_path"]:
            raise ValueError("Alias resolution changed")
        actual_hash = hashlib.sha256(alias.read_bytes()).hexdigest()
        if actual_hash != record["sha256"]:
            raise ValueError("Pinned provider hash changed")
        checked.append({"soname": soname, "alias": str(alias),
                        "canonical_path": record["canonical_path"], "sha256": actual_hash})
    return {"passed": True, "provider_count": len(checked), "providers": checked}
