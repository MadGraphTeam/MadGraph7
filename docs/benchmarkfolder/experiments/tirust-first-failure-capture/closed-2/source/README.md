# Private TIRust first-failure status capture

This source packet prepares a diagnostic for the unresolved causes identified
in `raw-dp-pilot-5-diagnosis-1`. Production, published evidence, the original
software flags, and COLLIER remain unchanged. Before separate native admission,
only pure tests, source inspection, archive/ELF parsing and read-only readiness
checks are permitted. The packet does not claim numerical truth or acceptance.

The capture copies the complete qualified source3 masters, preserving their
model, Cards resources, aliases, numerical objects, libraries and runtime.
It replaces only private copies of the driver, `TIR_interface` and
`TIRUST_interface`, and adds the Fortran/C observer. Original helicity/numerator
construction and contraction remain in the existing numerical objects.
No reduced model fixture or replacement library is used.

The driver retains initialization and the original three-probe history through
Zgg point 25 or ZZgg point 2. It does not clear software or IEEE flags. Full
tensor inputs are captured only for these registered primary requests:

| Case ID | Point | Groups | N | Rank | Probes |
|---|---:|---|---:|---:|---|
| 1, Zgg | 25 | 59, 61 | 5 | 4 | Original, reversed, signed z rotation |
| 2, ZZgg | 2 | 408, 422 | 6 | 5 | Original, reversed, signed z rotation |

This produces at most twelve unique input files. ZZgg bootstrap point 2 also
retains stage/status/error observations for the selected groups, without
additional full tensor requests. A duplicate primary key is an explicit
diagnostic failure rather than an overwritten or silently replaced input.
Missing early-guard requests remain missing and are not reconstructed.

`instrument.py` inserts callbacks around these existing Fortran stages:
10 entered; 11 rank rejected; 12 unsupported; 13 initialization failed;
14 cache-budget setup failed; 15 momentum-conservation guard rejected;
20 supplied invariant/tensor input ready; 21 tensor status returned;
22 nonfinite raw tensor rejected; 30 adapter succeeded; 40 adapter returned.
The final group event retains the existing stability flag after contraction or
the existing failed-result path. Entry callbacks read only initialized PL and
mass arrays; later unused intermediate buffers are not inspected.

The C observer wraps the unresolved `ml_tirust_evaluate_dp` import from static
`mtirust.o`, forwards exactly once, then copies status and thread-local
`LAST_ERROR` immediately. Its 4096-byte error copy records required capacity and
truncation. Fortran-local error text has a 512-byte buffer and unknown required
length/truncation. The observer serializes integer bit patterns, performs no
floating-point formatting/classification, reads tensor output only on C status
zero, and restores the bridge call's errno. Software flags and hardware IEEE
flags are not reset. A stage with no C-boundary event remains distinguishable.

`replay.c` validates the complete manifest and strict captured bit records before
making bridge calls. Each captured request gets a fresh context, one evaluation,
and one repeat in the same context. There are at most 24 standalone evaluations;
this is a diagnostic comparison, not an automatic rescue. Every failure remains
in the journal. Fresh-context success cannot refute a failure that depended on
the original structural workspace history.

The private build uses the pinned compiler, original link inputs, C11 compile
mode, and GNU `--wrap`. Readiness checks the original archive's unresolved
symbol and relocation. Preparation must additionally verify the final wrapper
definition, real import and cross-reference from `libtirust.a(mtirust.o)`.
Target loader checks and sampled actual runtime maps must match the original
runtime closure. Host tools get a clean environment; target LD settings are
applied only by the owned exec helper. Full qualified private-copy, card-search,
card-write-intent, process ownership/cleanup and protected-source checks reuse
the sealed source5 helpers.

All stages run from the MadGraph checkout. Planning is inert:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/tirust-first-failure-capture/source-proposal-1/controller.py plan
```

Read-only readiness API: import `controller.py` and call `readonly_inputs()`.
It checks the source inventory, all inherited preparations, both actual source
transformations, physical resources, compilers, original objects and static ABI
without creating work, invoking a compiler/loader, or evaluating an amplitude.
The guarded command below additionally denies mutations, children, loaders and
lock acquisition; choose a new receipt path for every invocation:

```sh
python3 -B docs/benchmarkfolder/experiments/tirust-first-failure-capture/source-proposal-1/verify_readiness.py --output .tirust-build/tirust-first-failure-readiness-1/proof.json
```

After root reviews readiness and admits each phase, the concrete commands are:

```sh
python3 -B docs/benchmarkfolder/experiments/tirust-first-failure-capture/source-proposal-1/controller.py prepare --execute
python3 -B docs/benchmarkfolder/experiments/tirust-first-failure-capture/source-proposal-1/controller.py capture --execute
python3 -B docs/benchmarkfolder/experiments/tirust-first-failure-capture/source-proposal-1/controller.py replay --execute
```

The fresh ignored namespace is `.tirust-build/tirust-first-failure-capture-1`.
Each capture has a 180-second inclusive cap; the complete standalone replay has
a 60-second cap. Native phases use the shared numerical lock, nonexclusive CPU
12 and a 4 GiB limit. No retry or fallback rebuild is performed. Setup, capture
and replay receipts retain command output, owned cleanup, maps, hashes, missing
requests and failure states. Protocol completion does not imply tensor success
or numerical acceptance. No native stage has been executed by preparing this
source packet.
