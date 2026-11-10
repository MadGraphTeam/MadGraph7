"""Portable text-only verification. Does not import or execute benchmark tools."""
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    index = json.loads((HERE / "index.json").read_text())
    assert {p.relative_to(HERE).as_posix() for p in HERE.rglob("*") if p.is_file()} == set(index["files"]) | {"index.json"}
    for name, checksum in index["files"].items():
        assert sha((HERE / name).read_bytes()) == checksum, name
    bundle = json.loads(gzip.decompress((HERE / "source-text.json.gz").read_bytes()))
    proposal = bundle["proposal"]
    assert sha(proposal["source-index.json"].encode()) == "ed0d91e6cc71ac555b67ec44145bc1f4cb8ccbbfb68e3ea6a3ee19821a119110"
    original_index = json.loads(proposal["source-index.json"])
    assert set(proposal) == set(original_index["files"]) | {"source-index.json"}
    for name, checksum in original_index["files"].items():
        assert sha(proposal[name].encode()) == checksum, name
    assert proposal["proposed.patch"].encode() == (HERE / "proposed.patch").read_bytes()
    receipt = json.loads((HERE / "qualification.json").read_text())
    assert sha((HERE / "qualification.json").read_bytes()) == "fbcfd030b6b49bbc62fa3a0cc9dbeb8573498cada6f1a66abde4b7bcc4ba49b0"
    assert receipt["tests_run"] == 61 and receipt["passed"] is True
    assert receipt["failures"] == receipt["errors"] == receipt["skipped"] == 0
    assert receipt["input_sha256_before"] == receipt["input_sha256_after"]
    assert receipt["test_log_sha256"] == sha((HERE / "tests.log").read_bytes())
    assert receipt["reviewed_patch_sha256"] == sha((HERE / "proposed.patch").read_bytes())
    for name, checksum in receipt["input_sha256_before"].items():
        text = proposal.get("src/" + name, bundle["unchanged_existing_tests"].get(name))
        assert text is not None and sha(text.encode()) == checksum, name
    for name, path in receipt["actual_module_paths"].items():
        assert Path(path) == Path(receipt["cwd"]) / "tools" / (name + ".py")
    preview = json.loads(proposal["actual-predecessor-card-origins-preview.json"])
    assert len(preview["cards"]) == 11 and len(preview["observations"]) == 66
    assert sum(len(value["files"]) for value in preview["cards"].values()) == 22
    assert preview["scope"] == "physical-input-ancestry-only"
    print("PASS: 15 proposal files, 6 live source bindings, 61 passing live Python tests; no native qualification")


if __name__ == "__main__":
    main()
