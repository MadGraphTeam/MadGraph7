# Static qualification and bounded private diagnostic replay proposal

This packet statically qualifies the exact, unloaded V3 artifacts and proposes
one diagnostic replay. It contains no new tensor evaluation or runtime mapping.
The closed build index is `47a9283d…`; its full pin and all artifact identities
are checked by the source and input bindings. The private DSO is
`73abcfb34d01afa09c45f1acb612da0a83d05c8961ec07b3cf2f65a02e94c721`;
the driver is `8b530ead988e77df43b39d83d6a743e1ddad7de26aa77bf51ffbf0ba3a17a71c`.
Neither has been loaded or executed during this static review.

Readelf inspected sixteen hash-bound x86-64 ELF objects: the driver, the
private DSO and the original fourteen runtime providers. The proposed reachable
closure is the driver plus fourteen providers, preserving thirteen original
provider path/hash identities and replacing only EB3 with the private DSO.
The four bridge dependencies and GNU version requirements match EB3 exactly.
All 809 strong dynamic imports are satisfied within this static set. Its
thirteen `ml_tirust_*` definitions belong solely to the private DSO: twelve
unchanged public APIs plus `ml_tirust_private_dp_diagnostics_v1`. No public
Rust, Symbolica or GMP definitions are exported. The link map binds the unique
private bridge and retained original link inputs; the C replay calls the bridge
directly and does not perform Fortran amplitude contraction.

The driver uses the original glibc 2.40 interpreter. The private DSO retains
the same glibc 2.42 RUNPATH as EB3, so ambient resolution is **not** admitted.
The proposal derives exactly fourteen SONAME/NEEDED-name aliases pointing to
the preserved provider files and the private DSO. Only the target process
receives that directory in `LD_LIBRARY_PATH`; Python and taskset start with
the existing helper's clean environment. A separate invocation of the pinned
original loader with `--inhibit-cache --library-path ALIASES --list DRIVER`
must resolve exactly the fourteen expected paths before the driver starts.
Mandatory sampled `/proc` maps must then authenticate the driver and all
fourteen providers, including inode/device and byte hashes. EB3 coexistence or
any unregistered mapped ELF is a retained failure. Static compatibility does
not substitute for those future runtime observations.

The existing diagnostic driver and parser are unchanged. Inputs are exactly
the six retained Zgg requests at point 25, methods 1/2/3 and groups 59/61,
with five propagators and rank four. Each gets its own fresh context and two
DP evaluations, twelve calls maximum. All six source files and copied files
are checked by hash; the exact ordered manifest is checked before execution
and afterward, rejecting symlink or hardlink replacements. Both poles and
the finite term remain enabled, cache remains zero, and the captured stability
flags remain unchanged. Missing ZZgg captures are excluded.

The driver copies `LAST_ERROR` before the metadata getter. The getter leaves
that error untouched; neither it nor the C driver reads failed tensor values.
The frozen parser requires thirty lifecycle records, twelve original status-1
refusals, matching error bytes, fresh complete DP snapshots and no failed
output serialization. Changed status, truncated metadata, incomplete output,
runtime mismatch or timeout is retained as failure without retry. This is a
metadata diagnostic, not a numerical accuracy or performance qualification.

`replay.py` reuses the existing immutable pilot command, exec gate, cleanup
owner and runtime-map verifier. Its only function adaptations are nonexclusive
CPU 12 to 16 and the expected private bridge basename. The shared numerical
lock is acquired nonblockingly before creating the exclusive replay directory;
a busy lock causes no loader or numerical launch. EW continuation has priority.
The admitted scope would be one process with at most twelve tensor calls,
4 GiB address space, single-thread settings, and sixty seconds including loader
checks and child cleanup. Synchronous pre/post hashing makes this an acceptance
deadline rather than a hard supervisor wall cap. Sampled maps are observations,
not a pre-entry mapping barrier or proof against transient mappings. No hardware
floating-point-exception or full backend tracing claim is made.

All source, input, artifact, provider, alias and manifest guards are checked
before launch and after terminal. Complete logs, loader output, sampled maps,
cleanup receipts, raw diagnostic bytes and a terminal receipt are retained in
the fresh `.tirust-build/tirust-private-diagnostic-replay-1` directory, including
failures. Production sources, original captures, installed libraries, reference
repositories, thresholds, masses, precision policies and numerical code stay
unchanged. The private Symbolica lock directory is confined to that new output.

Pure review commands, from the required checkout with its sourced environment:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/tirust-first-failure-capture/diagnostics-replay-proposal-1/replay.py
python3 -B docs/benchmarkfolder/experiments/tirust-first-failure-capture/diagnostics-replay-proposal-1/test_guards.py
```

Only after separate root review and admission, with EW no longer occupying
the shared numerical interval, the proposed single invocation is:

```sh
python3 -B docs/benchmarkfolder/experiments/tirust-first-failure-capture/diagnostics-replay-proposal-1/replay.py --execute
```

No native replay has been admitted or started by this packet.
