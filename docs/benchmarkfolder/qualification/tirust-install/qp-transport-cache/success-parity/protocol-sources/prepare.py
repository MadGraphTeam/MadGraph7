#!/usr/bin/env python3
"""Freeze input only; never compiles or executes native code."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIXTURES = ROOT / "docs/benchmarkfolder/experiments/poles-diagnostics/tensor-cases-r2.json"
BUDGET = 268435456


def main():
    fixtures = {f["name"]: f for f in json.loads(FIXTURES.read_text())}
    lastbit = dict(name="binary128-last-bit-tadpole", n=1, rank=0, mu2=4,
                   shifts=[[0, 0, 0, 0]], mass2=[[4, 0]])
    fixtures[lastbit["name"]] = lastbit
    records, refs = [], {}

    def add(name, precision, fixture="infrared-triangle-r2", mode=0, finite=1,
            check=0, cache=1, control=0, budget=BUDGET, expectation="values",
            reference=False, allow_refusal=False):
        key = fixture, precision, mode, finite
        identifier = len(records) + 1
        if reference:
            refs[key] = identifier
        records.append(dict(id=identifier, name=name, precision=precision, fixture=fixture,
            mode=mode, finite=finite, check=check, cache=cache, control=control,
            budget=budget, expectation=expectation, reference=refs.get(key),
            allow_refusal=allow_refusal, lastbit=int(fixture==lastbit["name"])))

    for precision in (64, 128):
        for finite in (1, 0):
            for mode in range(4):
                if finite==0 and mode==3:
                    continue  # Native Mode::new rejects an empty poles-only request.
                add(f"reference-{precision}-finite{finite}-mode{mode}", precision,
                    mode=mode, finite=finite, cache=0, control=1,
                    expectation="bypass", reference=True)
    for precision in (64, 128):
        prefix = "dp" if precision==64 else "qp"
        for finite in (1, 0):
            add(f"{prefix}-finite{finite}-cold", precision, finite=finite,
                control=1 if precision==64 and finite==1 else 0, expectation=prefix+"_miss")
            add(f"{prefix}-finite{finite}-hit", precision, finite=finite, expectation=prefix+"_hit")
        if precision==128:
            # DP entries survive the preceding QP insertions. Check both
            # precision stores while each contains real retained tensors.
            add("dp-hit-with-qp-populated",64,expectation="dp_hit")
            add("qp-hit-after-dp-interleave",128,expectation="qp_hit")
        for finite in (1, 0):
            add(f"{prefix}-finite{finite}-check", precision, finite=finite, check=1, expectation="bypass")
        for finite in (1, 0):
            add(f"{prefix}-finite{finite}-disabled", precision, finite=finite, cache=0, expectation="bypass")
        for finite in (1, 0):
            add(f"{prefix}-finite{finite}-hit-after-bypass", precision, finite=finite, expectation=prefix+"_hit")
        for finite in (1, 0):
            add(f"{prefix}-finite{finite}-zero-budget", precision, finite=finite, budget=0,
                control=3 if finite==1 else 0, expectation=prefix+"_refused")
        for finite in (1, 0):
            add(f"{prefix}-finite{finite}-budget-restored", precision, finite=finite,
                control=3 if finite==1 else 0, expectation=prefix+"_miss")
        for finite in (1, 0):
            add(f"{prefix}-finite{finite}-after-clear", precision, finite=finite,
                control=1 if finite==1 else 0, expectation=prefix+"_miss")
        for finite in (1, 0):
            add(f"{prefix}-finite{finite}-after-finalize", precision, finite=finite,
                control=2 if finite==1 else 0, expectation=prefix+"_miss")
    for fixture in ("massive-bubble-r2", "massive-triangle-r2"):
        for precision in (64, 128):
            for finite in (1, 0):
                refusal = fixture=="massive-triangle-r2" and precision==64
                add(f"{fixture}-{precision}-finite{finite}", precision, fixture=fixture,
                    finite=finite, cache=0, control=1, expectation="bypass",
                    reference=True, allow_refusal=refusal)
    add("last-bit-cold", 128, fixture=lastbit["name"], control=1, expectation="qp_miss", reference=True)
    add("last-bit-hit", 128, fixture=lastbit["name"], expectation="qp_hit")
    add("last-bit-check", 128, fixture=lastbit["name"], check=1, expectation="bypass")
    add("last-bit-disabled", 128, fixture=lastbit["name"], cache=0, expectation="bypass")
    add("last-bit-after-clear", 128, fixture=lastbit["name"], control=1, expectation="qp_miss")

    lines = [str(len(records))]
    for r in records:
        f = fixtures[r["fixture"]]
        lines.append(" ".join(str(x) for x in (r["id"], r["precision"], f["n"], f["rank"],
            r["mode"], r["finite"], r["check"], r["cache"], r["control"], r["budget"], r["lastbit"])))
        lines.append(format(float(f["mu2"]), ".17e"))
        lines.extend(" ".join(format(float(x), ".17e") for x in row) for row in f["shifts"])
        lines.extend(" ".join(format(float(x), ".17e") for x in row) for row in f["mass2"])
    raw = ("\n".join(lines)+"\n").encode()
    manifest = dict(scope="Raw public Fortran mtirust coefficient bytes, statuses and cache/lifecycle counters; no generated-amplitude, IEEE-environment or performance qualification",
        source_fixture_path=str(FIXTURES.relative_to(ROOT)),
        source_fixture_sha256=hashlib.sha256(FIXTURES.read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(raw).hexdigest(),
        fixture_precision="Decimal fixture values are read into binary64 and promoted for QP, except the exact binary128 tadpole override.",
        output_encoding="Native-order 64-bit integer words in hexadecimal; complex values contain real then imaginary parts.",
        expected_calls=len(records), fixtures={name: fixtures[name] for name in sorted({r['fixture'] for r in records})},
        actions=records)
    (HERE / "fixtures.dat").write_bytes(raw)
    (HERE / "specification.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"Prepared {len(records)} calls; no native code executed.")


if __name__ == "__main__":
    main()
