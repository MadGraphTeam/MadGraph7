#!/usr/bin/env python3
"""Read-only verification of the immutable completed static-owner stage."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    manifest = json.loads((HERE / "SHA256SUMS.json").read_text())
    for relative, record in manifest["files"].items():
        path = HERE / relative
        assert path.is_file() and path.stat().st_size == record["bytes"], relative
        assert digest(path) == record["sha256"], relative
    report = HERE / "full-evaluation-profile-v1"
    archives = json.loads((report / "profile-archives.json").read_text())["artifacts"]
    for filename, record in archives.items():
        path = report / filename
        assert digest(path) == record["compressed_sha256"]
        result = hashlib.sha256()
        size = 0
        with gzip.open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                result.update(chunk)
                size += len(chunk)
        assert result.hexdigest() == record["decompressed_sha256"]
        assert size == record["decompressed_bytes"]
    with gzip.open(report / "perf.data.gz", "rb") as raw, \
         gzip.open(report / "perf-recovered.data.gz", "rb") as recovered:
        old_header = bytearray(raw.read(104))
        new_header = recovered.read(104)
        offset, size = struct.unpack_from("<QQ", old_header, 40)
        assert size == 0 and offset == 3336
        struct.pack_into("<Q", old_header, 48,
                         archives["perf.data.gz"]["decompressed_bytes"] - offset)
        struct.pack_into("<4Q", old_header, 72, 0, 0, 0, 0)
        assert bytes(old_header) == new_header
        assert raw.read(offset - 104) == recovered.read(offset - 104)
        types = Counter()
        while header := raw.read(8):
            assert len(header) == 8 and recovered.read(8) == header
            kind, misc, length = struct.unpack("<IHH", header)
            assert length >= 8
            payload = raw.read(length - 8)
            assert len(payload) == length - 8 and recovered.read(length - 8) == payload
            types[kind] += 1
        assert not recovered.read(1)
        assert sum(types.values()) == 6980 and types[9] == 6476 and types[4] == 1
    print(f"Verified {len(manifest['files'])} files and both lossless profile streams.")


if __name__ == "__main__":
    main()
