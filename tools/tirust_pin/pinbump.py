"""pinbump.py <mg_root> <tirust_export_dir> <new_rev> <new_patch>

Update MadGraph's TIRust pin: copy the regenerated poles-only patch, rewrite
vendor/tirust/patches/manifest.json (upstream_revision, patch_sha256, source_sha256
of every listed source file, taken from a clean `git archive <new_rev>` export),
and replace the old revision in vendor/tirust/Cargo.toml, Cargo.lock and
madgraph/various/tirust_install.py (UPSTREAM_REVISION). The old revision is read
from the manifest; Cargo.lock only carries the git rev string, replaced textually.
"""
import hashlib, json, sys
from pathlib import Path
mg, export, rev, patch = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
assert len(rev) == 40
vt = mg / "vendor/tirust"
manifest = json.loads((vt / "patches/manifest.json").read_text())
old = manifest["upstream_revision"]
(vt / "patches/0001-direct-poles-only.patch").write_bytes(patch.read_bytes())
manifest["upstream_revision"] = rev
manifest["patch_sha256"] = hashlib.sha256(patch.read_bytes()).hexdigest()
for name, digest in list(manifest["source_sha256"].items()):
    path = export / name
    if digest is None:
        assert not path.exists(), name
    else:
        manifest["source_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
(vt / "patches/manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
for rel in ("vendor/tirust/Cargo.toml", "vendor/tirust/Cargo.lock", "madgraph/various/tirust_install.py"):
    p = mg / rel
    s = p.read_text()
    assert old in s, rel
    p.write_text(s.replace(old, rev))
print("pin", old, "->", rev, "patch", manifest["patch_sha256"])
