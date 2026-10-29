"""Verify the finite offline document closure and reproduce the final PDF."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reproduced.pdf"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((HERE / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        filename = (ROOT / name).resolve()
        if not filename.is_relative_to(ROOT.resolve()) or digest(filename) != expected["sha256"] or filename.stat().st_size != expected["bytes"]:
            raise ValueError("Changed offline report input: " + name)
    if args.verify_only:
        print(json.dumps({"verified_inputs": len(manifest["files"]), "native_reads": 0}, sort_keys=True))
        return
    typst = shutil.which("typst")
    if typst is None:
        raise RuntimeError("Install Typst0.15.1 or activate the recorded dev environment")
    version = subprocess.check_output([typst, "--version"], text=True).strip()
    if version != manifest["typst_version"]:
        raise RuntimeError("Expected " + manifest["typst_version"] + "; observed " + version)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, SOURCE_DATE_EPOCH=str(manifest["source_date_epoch"]),
               TYPST_FONT_PATHS="", RAYON_NUM_THREADS="1")
    command = [typst, "compile", "--root", str(ROOT), "--ignore-system-fonts", "--ignore-embedded-fonts",
               "--font-path", str(HERE / "fonts"), "--creation-timestamp", str(manifest["source_date_epoch"]),
               str(ROOT / "benchmark.typ"), str(output)]
    subprocess.run(command, env=env, check=True)
    actual = digest(output)
    if actual != manifest["expected_pdf"]["sha256"]:
        raise ValueError("PDF bytes differ from the retained final build: " + actual)
    print(json.dumps({"verified_inputs": len(manifest["files"]), "output": str(output),
                      "sha256": actual, "byte_identical": True, "native_reads": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
