# Current kinematic trace: ordinary parity and retained stress failures

The ordinary stage passed four exact observation comparisons between unmodified,
trace-disabled and trace-enabled variants. Its 12 workers made 24 outer amplitude
entries, with 24 verified READY/DONE runtime-map boundaries. This establishes
observational parity for the selected ordinary points and fixed modes.

The stress stage **failed** its six numerical contribution checks and exited 1.
All 18 workers completed their protocols and 36 outer entries, and the seven
observation files matched byte for byte within each variant triple. Nevertheless,
every measured stress contribution set a failure flag: DP `(1,0)`, QP `(0,1)`.
Bootstrap flags were `(0,0)`. Equal failed outputs are preserved as failures and
do not qualify the stress points. All 36 stress READY/DONE maps were verified.

This packet retains both closed stages without retries: raw values and bit words,
driver states, traces, stdout, maps, specifications, exact card/point inputs,
owned-command and cleanup receipts, loader admissions, root reviews and pure
decoded analyses. The closed ordinary and stress seals are respectively
`56d4249509bbda4d…` and `df0987b3f0c20dd2…`. Source guards `325eb320…` and
`9c81ae01…` remain unchanged, including their historical pre-admission wording.

The adjacent [build packet](../current-kinematic-trace-build-qualification-1/README.md),
sealed as `89697913341ecc5fe765fb023cc09bced783eb3eedb85436afe1ca7d4dc6c033`,
is an explicit dependency. It contains the source/derived trace packets, build
receipts, pinned compiler and provider identities. This runtime packet does not
duplicate that entire packet. Literal cards and points selected by worker
specifications are copied from its sealed input inventory and checked against
the per-worker hashes. No current external worker card or binary is needed to
replay this evidence.

The unchanged pure decoder `df0c1113…` interprets integer words under the recorded
GNU x86_64 binary64/binary128, low-word-first ABI. Ordinary analysis `5842fea9…`
and stress analysis `66fab0dd…` preserve their source and trace pins.
Exact rational calculations on captured words are observations, not corrections
to thresholds, masses, inputs or amplitudes. A successful tensor status does not
override the host contribution-failure flags. The failed tensor buffers are not
used to assert accuracy.

There is **no independent accuracy, automatic stability or performance
qualification** here. Fixed-mode return codes 110/120 remain unchecked, and
timings are incidental protocol data. Historical absolute paths identify
provenance only. External ELF/library/object bytes are omitted; portable replay
does not rehash or load them. Only explicitly listed closed command, worker and
analysis payloads are selected; no execution directory is discovered or globbed.

Run from the repository directory:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/current-kinematic-trace-runtime-qualification-1/verify.py
```

The verifier reads only this packet and its declared build sibling. It verifies
stored/decompressed identities, closed worker/card/map/source bindings, ordinary
parity and stress failure gates, then replays the sealed decoder's pure in-memory
functions against the retained trace bytes. It executes no retained native
controller and launches no subprocess. Keep both sibling packets together when
relocating them for replay.
