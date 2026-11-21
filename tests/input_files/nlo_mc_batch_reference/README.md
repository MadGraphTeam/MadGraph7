# Native width-one MC@NLO controls

These fixtures bind the 19 November 2026 development source and the locally checked-out main in `source.json`. Each job used an ungrouped on-shell Born subprocess, the `loop_sm-no_b_mass` model, 6.5 TeV proton beams, NNPDF31_nlo_as_0118 member 0, and classic Ninja 1.2.1. The model card sets the stable top width to zero in the top cases. Compiler and external-library hashes identify the common strict O3, non-fast-math installation. The controls use the physical Fortran real amplitudes within the generated MG7 driver. They provide correctness evidence only.

The DY and gluon-initiated top fixtures contain results for each native MINT channel and mode. The six stored numbers from each `res_*.dat` are the absolute integral, its error, the signed integral, its error, the iteration count and the point count. The LHE initialization and SHA-256 of ordered event bodies bind the generator output. Quark-initiated top is an additional recoil and history control. Different processes with explicit incident partons represent different cross sections.

The `cards/<case>/` directories contain the model and loop cards used by each run and the applied seed-42 run card embedded in its LHE output. The JSON files distinguish the pre-launch run-card hash from the hash of the formatter-normalized LHE copy.

The two principal fixtures were generated from the tracked Git archive. A separate export added the diagnostic decision ledger and physical-provider trace, recorded the same seed-42 run, then replayed its complete journals in a fresh directory. The checks require the pristine, recorded and replayed outputs to agree on every per-channel scientific value and every event body. The smaller examples in the fixtures preserve the first genuine physical input and return of each job. Their sixteen-digit hexadecimal words encode the binary64 bit pattern. Complete traces and journals remain in the macrotask reference directory under the hashes recorded here.

The maintained workspace commands below reproduce the controls from their bound sources and dependencies. Run the generator with the repository's supported Python and GNU toolchain. The helper defaults to the declared compiler-bound core IDs. A fresh output directory is required for an independent execution.

```bash
cd scratch/production_batching/reference
python freeze.py
for process in dy tt qqtt; do
    python run_fixture.py "$process" generate
    python run_fixture.py "$process" compile
    python run_fixture.py "$process" run
done
python freeze_trace.py
for process in dy tt; do
    python run_fixture.py --traced "$process" generate
    python run_fixture.py --traced "$process" compile
    python run_fixture.py --traced "$process" run
    python replay_fixture.py "$process"
done
```

With a new generator revision, preserve the original archive, source manifest, run cards, thresholds and histories as the independent comparison. The native gate reads actual job traces from `madevent_mintMC`. For the recorded control use `python tests/acceptance_tests/mc_batch_trace.py --width-one-control --expected-writes 12 <job traces>` from a feature checkout. A multi-point run uses the same command with the width-one flag removed.
