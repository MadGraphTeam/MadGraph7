#!/usr/bin/env python3
"""Replay bundled tensor fixtures against pinned original Ninja results."""
import argparse
import hashlib
import json
import pathlib
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=("dp", "mp"), default="dp")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--executable", type=pathlib.Path)
    args = parser.parse_args()
    bridge = pathlib.Path(__file__).resolve().parent.parent
    data = bridge / "tests" / "data"
    provenance = json.loads((data / "provenance.json").read_text())
    for item in provenance["files"]:
        digest = hashlib.sha256((data / item["file"]).read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise SystemExit(f"Fixture checksum mismatch: {item['file']}")

    executable = args.executable
    if executable is None:
        command = ["cargo", "--manifest-path", str(bridge / "Cargo.toml")]
        subprocess.run(
            [command[0], "build", *command[1:], "--locked", "--example", "replay"],
            check=True,
        )
        metadata = subprocess.run(
            [command[0], "metadata", *command[1:], "--locked",
             "--no-deps", "--format-version=1"],
            check=True, text=True, capture_output=True,
        )
        target = pathlib.Path(json.loads(metadata.stdout)["target_directory"])
        executable = target / "debug" / "examples" / "replay"
    executable = executable.resolve()
    results = []
    for case in sorted((data / "cases").glob("*.json")):
        command = [str(executable), str(case), str(data / "ninja" / case.name)]
        if args.precision == "mp":
            command.append("--mp")
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode:
            raise SystemExit(
                f"Replay failed: {case.name} ({args.precision})\n"
                f"{completed.stdout}{completed.stderr}"
            )
        # Symbolica may print its licensing banner before the final JSON line.
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        results.append(result)
        print(json.dumps(result), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
