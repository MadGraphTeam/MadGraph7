"""Fetch the frozen report's optional binary inputs into ignored local paths.

The published PDF and its numerical summaries remain in the source tree. Run
this once before the separately offline, byte-identical PDF reproduction.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

HERE = Path(__file__).resolve().parent
REVISION = "cc6f6fe6dd344a0a767dcfe52341c147e39a193a"
BASE_URL = ("https://raw.githubusercontent.com/MadGraphTeam/MadGraph7/"
            + REVISION + "/doc/benchmarkfolder/")
ARTIFACTS = (
    "final-report-v1/cli-report.json.gz",
    "final-report-v1/fonts/LibertinusMath-Regular.otf",
    "final-report-v1/fonts/LibertinusMono-Regular.otf",
    "final-report-v1/fonts/LibertinusSerif-Bold.otf",
    "final-report-v1/fonts/LibertinusSerif-BoldItalic.otf",
    "final-report-v1/fonts/LibertinusSerif-Italic.otf",
    "final-report-v1/fonts/LibertinusSerif-Regular.otf",
    "final-report-v1/fonts/LibertinusSerif-Semibold.otf",
)


def matches(data, expected):
    return (len(data) == expected["bytes"]
            and hashlib.sha256(data).hexdigest() == expected["sha256"])


def fetch_artifacts(output_root, manifest):
    """Fetch only missing, hash-verified files; never overwrite changed inputs."""
    root = output_root.resolve()
    downloaded = 0
    for name in ARTIFACTS:
        expected = manifest["files"][name]
        target = (root / name).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Artifact path escapes output directory: " + name)
        if target.exists():
            if not matches(target.read_bytes(), expected):
                raise ValueError("Existing artifact differs; move it aside first: "
                                 + str(target))
            continue
        # Reading at most the expected size plus one also rejects oversized
        # responses. No response is installed before its size and hash agree.
        with urllib.request.urlopen(BASE_URL + name, timeout=60) as response:
            data = response.read(expected["bytes"] + 1)
        if not matches(data, expected):
            raise ValueError("Downloaded artifact failed checksum: " + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        downloaded += 1
    return {"verified_artifacts": len(ARTIFACTS), "downloaded": downloaded,
            "revision": REVISION, "output_root": str(root)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=HERE,
                        help="report root to populate (default: this directory)")
    args = parser.parse_args()
    manifest = json.loads((HERE / "final-report-v1/manifest.json").read_text())
    print(json.dumps(fetch_artifacts(args.output_root, manifest), sort_keys=True))


if __name__ == "__main__":
    main()
