This append-only correction retains the first attempted run and its loader log.
The original fixture required libquadmath even though its ELF dependencies do
not load that library. Its complete dependency closure is exactly libgfortran,
libm, libgcc_s, libc and the recorded ELF interpreter. The corrected check
requires exact identities for all five, with no subset allowance.

The controller imports the frozen original controller and keeps its original
build binding. Only the first loader check's expected map, subset flag and log
destination change; all seven ELF closures are independently derived and bound.
The six amplitude executables still require their original complete runtime
map, including libquadmath. No numerical source, object, executable, model,
card, case, scientific validator, command, deadline or failure expectation is
changed. The original failure and all protected inputs remain hash-bound.

Run from the repository root after numerical-slot admission:

    source tools/tirust_env.sh
    python -B .tirust-build/qp-mass-general-implementation/combined-qualification/loader-correction/control.py verify
    python -B .tirust-build/qp-mass-general-implementation/combined-qualification/loader-correction/control.py run

The result is written here. Original scientific output directories are still
empty and retain their frozen names so the unchanged validators can consume
them. No retry is automatic. The original run.json remains a failed attempt.
