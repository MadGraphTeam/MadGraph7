# Stock MadGraph `check profile` preparation

Status: prepared and independently audited; no CLI, compiler, SDK or numerical
program has been executed from this capsule. Numerical execution requires the
coordinator's GO after the other compilation/timing windows close.

The source archive is exactly MadGraph commit
`cafb35d7987436661c3346e39c724d329846a1f2`, which includes the explicit-pole
profiling filter fix. If the final TIRust integration or compiler policy changes,
prepare a fresh namespace from that revision before collecting timings. Never
relabel this source snapshot as a later implementation.

The actual stock commands are in `n1-8.mg5` and `n1-9.mg5`. They select
`loop_sm`, `u u~ > z [virt=QCD]`, backend 8 or 9 separately, full finite plus
UV/IR pole mode 0, ten stability points and the same explicit Python seed.
`n2-*.mg5` is an optional, unexecuted QCD four-quark extension.

The command is `check profile`, rather than a standalone `profile` command.
Only the first reduction library in a selection list is timed; separate
commands are necessary to compare two backends. The stock timer chooses its
own timing batch lengths, targets about 30 seconds and has a minimum of 50
points. The argument `10` controls the subsequent stability analysis, not the
timing batch length. Exact point counts, drivers, cards, flags and runtime
identities are preserved in stock `profile_timing_metadata.json`.

The stock timing setup uses fixed `CTModeRun=1`. Its timing stream has no
per-point return codes or precision counters. These results must be labeled
stock fixed-DP timing, with the external stability analysis reported separately.
They do not qualify native-DP success for every timed point or automatic
full-amplitude rescue performance. `FULL - SKIP` is an estimated reduction time.
The Fortran timing point generator has its own state, so a Python seed alone
does not freeze identical timing points across the two commands. Shared-host
measurements are descriptive, not the paired local acceptance campaign.

After coordinator GO, use the pinned Python:

```
/nix/store/4jddwqz8nx9v4p4wzvhz1kfg88gnjcv8-python3-3.13.12-env/bin/python3 -B .tirust-build/madgraph-cli-profile-2/run_cli.py --case n1 --backend 8
```

The successful terminal receipt identifies its `saved_export`. The second call
may clone this saved export for efficient reuse of model and subprocess builds:

```
/nix/store/4jddwqz8nx9v4p4wzvhz1kfg88gnjcv8-python3-3.13.12-env/bin/python3 -B .tirust-build/madgraph-cli-profile-2/run_cli.py --case n1 --backend 9 --reuse-export <saved_export_from_backend_8>
```

Both outputs remain independent. Internal export symlinks are rebound to the
clone. Reuse explicitly rejects a changed TIRust prefix or changed DSO because
existing external library/include/make paths must not retain an earlier
installation. Use a fresh export when switching prefixes. `--tirust-prefix`
selects an available installation for a fresh run; all runtime identities are
recorded. Nothing updates the repository's saved configuration.

The runner uses the existing owned-process/deadline helpers, the numerical
mutex, a nonexclusive allowed CPU, a 3,600-second total cap with 60 seconds
reserved for cleanup, and 32-GiB address-space/2-GiB file-size limits. It retains
all output on failure and checks shared source, configuration and libraries
before/after. An exit status of zero alone is insufficient: it requires a fresh
profile log and matching stock timing metadata before reporting completion.

A source-only independent audit verified all 2,348 frozen files and identified
the changed-prefix reuse trap; the runner now rejects that combination.
