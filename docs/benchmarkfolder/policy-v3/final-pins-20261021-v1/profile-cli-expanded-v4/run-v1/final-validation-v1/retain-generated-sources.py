"""Retain the exact source bytes already named by the 72 generated manifests."""
import hashlib
import json
from pathlib import Path

ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
RUN = HERE.parent
DEST = RUN / "generated-source-closure-v1"
assert not DEST.exists()
DEST.mkdir()
(DEST / "sources").mkdir()


def read(path):
    return json.loads(Path(path).read_text())


def bound(path):
    data = Path(path).read_bytes()
    return {"path": str(Path(path).relative_to(ROOT)), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


blobs, results, missing = {}, [], []
for filename in read(RUN / "result-list.json"):
    path = ROOT / filename
    result = read(path)
    manifest = result["generated_sources"]
    output = Path(result["executable"]["path"])
    if not output.is_absolute():
        output = ROOT / output
    output = output.parents[2]
    for name, record in manifest["files"].items():
        digest = record["sha256"]
        if digest in blobs:
            assert blobs[digest]["bytes"] == record["bytes"]
            continue
        candidates = []
        if Path(name).name in ("check_sa.f", "loop_matrix.f"):
            candidates.append(path.parent / Path(name).name)
        candidates.append(output / name)
        selected, data = None, None
        for candidate in candidates:
            if candidate.is_file():
                payload = candidate.read_bytes()
                if len(payload) == record["bytes"] and hashlib.sha256(payload).hexdigest() == digest:
                    selected, data = candidate, payload
                    break
        if selected is None:
            missing.append({"result": filename, "name": name, "expected": record})
            continue
        assert b"\x00" not in data and not data.startswith(b"\x7fELF")
        target = DEST / "sources" / (digest + ".txt")
        target.write_bytes(data)
        blobs[digest] = {**bound(target), "observed_source": str(selected.relative_to(ROOT))}
    results.append({"result": bound(path), "generated_manifest_sha256": manifest["files_sha256"], "source_entries": len(manifest["files"])})

assert not missing, missing
for filename in read(RUN / "result-list.json"):
    for record in read(ROOT / filename)["generated_sources"]["files"].values():
        assert blobs[record["sha256"]]["bytes"] == record["bytes"]

receipt = {"schema": "retained-cli-generated-source-closure-v1", "status": "passed", "scope": "Exact Fortran/include/header/makefile bytes named by each actual result generated-source manifest; content-addressed deduplication, no executable/object/module/native library", "invocation": bound(Path(__file__)), "result_list": bound(RUN / "result-list.json"), "results": results, "source_blobs": dict(sorted(blobs.items())), "unique_sources": len(blobs), "source_bytes": sum(b["bytes"] for b in blobs.values()), "source_entries": sum(r["source_entries"] for r in results), "missing": [], "numerical_calls": 0, "native_archive_bytes_read": 0}
(DEST / "closure.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps({"receipt": bound(DEST / "closure.json"), "sources": len(blobs), "bytes": receipt["source_bytes"]}))
