#!/usr/bin/env python3
"""Create an isolated cache prototype from verified production bridge sources."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(
        ".benchmark-work/experiments/master-cache-v2-reproduced/bridge"))
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    allowed = ROOT / ".benchmark-work/experiments"
    if not output.is_relative_to(allowed) or output == allowed:
        raise RuntimeError(f"Output must be a new directory under {allowed}")
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite {output}")
    base = json.loads((HERE / "base.json").read_text())
    source = ROOT / "vendor/ninjars"
    for name, expected in base["base_files"].items():
        actual = hashlib.sha256((source / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Production source no longer matches prototype base: {name}")
    shutil.copytree(source, output, ignore=shutil.ignore_patterns(
        "target", "*.a", "*.o", "*.mod", "*.log", "smoke",
        "ninjars-native-libs.txt", "ninjars-link-before.txt"))
    subprocess.run(["patch", "--batch", "-p1", "--directory", str(output),
                    "--input", str(HERE / "prototype.patch")], cwd=ROOT, check=True)
    expected = json.loads((HERE / "prototype-files.json").read_text())
    for name, digest in expected.items():
        actual = hashlib.sha256((output / name).read_bytes()).hexdigest()
        if actual != digest:
            raise RuntimeError(f"Reproduced prototype differs: {name}")
    print(output)
    print("Source reconstructed. No build or numerical evaluation was run.")


if __name__ == "__main__":
    main()
