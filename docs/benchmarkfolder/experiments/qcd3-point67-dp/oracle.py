#!/usr/bin/env python3
"""Replay frozen DP tensor inputs in production DP and genuine Float256."""
import argparse
from decimal import Decimal, localcontext
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
WORK = ROOT / ".benchmark-work/experiments/qcd3-point67-dp"
LIB = ROOT / "HEPTools/ninjars/lib"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare():
    rows = json.load(gzip.open(HERE / "normal/reductions.json.gz", "rt"))
    chosen = [row for row in rows if row["loop"] in (3, 52, 71)]
    assert len(chosen) == 6
    (HERE / "oracle-inputs.json").write_text(json.dumps(chosen, indent=2)+"\n")
    with (HERE / "oracle-inputs.dat").open("w") as out:
        print(len(chosen), file=out)
        for row in chosen:
            print(row["loop"], row["direction"], row["n"], row["rank"],
                  len(row["coefficients"]), file=out)
            values = [x for pair in row["coefficients"] for x in pair]
            values += row["s_matrix_column_major"] + row["momenta_column_major"]
            values += [x for pair in row["masses_squared"] for x in pair]
            values += [row["scale_squared"]]
            print("\n".join(format(x, ".17e") for x in values), file=out)
    obj = WORK / "replay.o"
    commands = [
        ["gcc", "-O2", "-I"+str(ROOT / "vendor/ninjars"), "-c",
         str(HERE / "replay.c"), "-o", str(obj)],
        ["gfortran", str(obj),
         *shlex.split((LIB / "ninjars-link-before.txt").read_text()),
         str(LIB / "libninjars.a"),
         *shlex.split((LIB / "ninjars-native-libs.txt").read_text()),
         "-lstdc++", "-o", str(WORK / "replay")],
    ]
    with (WORK / "replay-build.log").open("w") as log:
        for command in commands:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    (HERE / "oracle-build.json").write_text(json.dumps({
        "commands": commands, "executable_sha256": digest(WORK / "replay"),
        "library_sha256": digest(LIB / "libninjars.a"),
        "source_sha256": digest(HERE / "replay.c"),
        "inputs_sha256": digest(HERE / "oracle-inputs.dat"),
        "purpose": "Frozen binary64 inputs widened exactly through40-digit binary128 transport; production Float256 reduction",
    }, indent=2)+"\n")

def run(cpu):
    metadata = json.loads((HERE / "oracle-build.json").read_text())
    assert digest(WORK / "replay") == metadata["executable_sha256"]
    assert digest(HERE / "oracle-inputs.dat") == metadata["inputs_sha256"]
    command = ["taskset", "-c", str(cpu), str(WORK / "replay")]
    env = os.environ.copy()
    env.setdefault("SYMBOLICA_LOCK_DIR", str(ROOT / ".ninjars-build/symbolica-locks"))
    with (HERE / "oracle-inputs.dat").open() as source, (WORK / "replay.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdin=source, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    raw = (WORK / "replay.log").read_text()
    answers = {}
    for line in raw.splitlines():
        words = line.split()
        if words[:1] not in (["DP"], ["MP"]):
            continue
        kind, loop, direction, status = words[:4]
        assert int(status) == 0 and len(words) == 12
        answers[(int(loop), int(direction), kind)] = words[4:]
    rows = json.loads((HERE / "oracle-inputs.json").read_text())
    report = []
    with localcontext() as ctx:
        ctx.prec = 80
        for row in rows:
            identity = row["loop"], row["direction"]
            dp = list(map(float, answers[(*identity, "DP")]))
            expected = [x for pair in row["ninjars"] for x in pair]
            assert dp[:6] == expected, (identity, "DP replay changed")
            mp = list(map(Decimal, answers[(*identity, "MP")]))
            rs_errors = [Decimal.from_float(x)-y for x, y in zip(dp[:6], mp)]
            ninja = [x for pair in row["ninja"] for x in pair]
            ninja_errors = [Decimal.from_float(x)-y for x, y in zip(ninja, mp)]
            report.append(dict(loop=identity[0], direction=identity[1],
                n=row["n"], rank=row["rank"],
                oracle_laurent_components=[str(x) for x in mp[:6]],
                ninjars_errors=[str(x) for x in rs_errors],
                ninja_errors=[str(x) for x in ninja_errors]))
    shutil.copy2(WORK / "replay.log", HERE / "oracle-output.txt")
    (HERE / "oracle-results.json").write_text(json.dumps({
        "command": command, "metadata": metadata, "dp_replay_exact": True,
        "note": "Oracle uses the same frozen DP numerator inputs; this does not replace actual QP numerator construction.",
        "rows": report,
    }, indent=2)+"\n")
    for row in report:
        print(row["loop"], row["direction"], "finite real errors",
              row["ninjars_errors"][0], row["ninja_errors"][0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--cpu", type=int, default=4)
    args = parser.parse_args()
    prepare() if args.action == "prepare" else run(args.cpu)
