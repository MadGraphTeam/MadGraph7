"""Read the seven closed inputs solely from this portable package."""

import hashlib
import json
import re
import zlib
from pathlib import Path


def strict_json(blob):
    def unique(items):
        result = {}
        for name, value in items:
            if name in result:
                raise ValueError("duplicate JSON key: " + name)
            result[name] = value
        return result
    return json.loads(blob, object_pairs_hook=unique)


def local_path(root, name):
    if (not isinstance(name, str) or name in (".", "..")
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None):
        raise ValueError("portable payload must be a simple relative filename")
    root = Path(root).absolute()
    path = root / name
    if any(p.is_symlink() for p in (path, root, *root.parents)):
        raise ValueError("portable payloads and their parents must not be symlinks")
    if not path.is_file():
        raise ValueError("missing regular portable payload: " + name)
    return path


def file_record(blob):
    return {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}


def verify_payloads(root):
    index = strict_json(local_path(root, "index.json").read_bytes())
    if index.get("schema_version") != 1 or not isinstance(index.get("files"), dict):
        raise ValueError("invalid portable index")
    if set(p.name for p in Path(root).iterdir()) != set(index["files"]) | {"index.json"}:
        raise ValueError("unexpected or missing portable files")
    for name, expected in index["files"].items():
        if file_record(local_path(root, name).read_bytes()) != expected:
            raise ValueError("portable payload hash/size mismatch: " + name)
    return index


def input_map(root, pins):
    index = verify_payloads(root)
    mapping = strict_json(local_path(root, "input-map.json").read_bytes())
    if mapping.get("schema_version") != 1 or set(mapping.get("inputs", {})) != set(pins):
        raise ValueError("input map must contain exactly the seven original input names")
    seen = set()
    for name, entry in mapping["inputs"].items():
        if set(entry) != {"gzip_file", "gzip", "raw"}:
            raise ValueError("unexpected input-map fields")
        compressed = entry["gzip_file"]
        local_path(root, compressed)
        if compressed in seen or compressed not in index["files"]:
            raise ValueError("duplicate or unindexed compressed input")
        seen.add(compressed)
        if entry["gzip"] != index["files"][compressed]:
            raise ValueError("compressed input binding differs from package index")
        if (entry["raw"].get("sha256") != pins[name]
                or type(entry["raw"].get("bytes")) is not int
                or not 0 < entry["raw"]["bytes"] <= 64 * 1024 * 1024):
            raise ValueError("raw input pin or bounded length mismatch")
    return mapping


def read_input(root, name, pins):
    if name not in pins:
        raise ValueError("unregistered original input name")
    entry = input_map(root, pins)["inputs"][name]
    compressed = local_path(root, entry["gzip_file"]).read_bytes()
    if file_record(compressed) != entry["gzip"]:
        raise ValueError("compressed bytes changed during read")
    # Single gzip member, no filename/comment/time fields, deterministic mtime0.
    if compressed[:8] != b"\x1f\x8b\x08\x00\x00\x00\x00\x00":
        raise ValueError("noncanonical gzip header")
    decoder = zlib.decompressobj(31)
    raw = decoder.decompress(compressed, entry["raw"]["bytes"] + 1)
    if (not decoder.eof or decoder.unused_data or decoder.unconsumed_tail
            or file_record(raw) != entry["raw"]):
        raise ValueError("truncated, concatenated, oversized or mismatched gzip input")
    return raw
