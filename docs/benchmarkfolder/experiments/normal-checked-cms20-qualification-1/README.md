# Private checked CMS20 qualification

This capsule retains one ordinary bootstrap and one CMS20 evaluation of the
private Fortran invariant candidate, with MadLoop's automatic stability checks
and genuine QP rescue. It also preserves the earlier failed controller attempt.
The experiment uses the saved complex-mass `e+ e- > mu+ mu- [virt=QED]` export,
backend 9, cache disabled, all poles and finite terms, `CTModeRun=-1`, and
`FORCE_STABILITY_CHECK(.TRUE.)`. The original cards, thresholds, scientific
libraries, precision transport and scientific candidate remain unchanged.

The successful worker made exactly two THRES requests, two SLOOP entries, six
passes and 156 native reductions. The ordinary bootstrap accepted DP modes 1/2
with return 219. CMS20 retained six failed DP reductions in groups 8, 12 and 24
across modes 1/2. MadLoop's existing failure decision and reset led to genuine
QP modes 4/5; all 52 QP reductions succeeded, the final software flags were clear,
and MadLoop returned 323. Failed DP tensor outputs are not used for comparisons.
Input traces, statuses, immediate errors and the complete accepted QP epoch are
retained.

Every accepted group passed source-bound geometry, producer/environment,
stability, completeness and omission checks. The 26 primary QP groups matched
the earlier matched-input TIRust experiment in native input and canonical
numerator words. Final Born, finite, single-pole and double-pole words, for both
split orders, equal that saved TIRust result. Agreement with the matched NinjaRS
control is within `2.014e-16`; separate universal single/double pole consistency
errors are `5.232e-16` and `2.679e-16`. The ordinary control difference is below
`3.669e-15`. These are distinct checks, not a proof of all finite amplitudes.

The first worker stopped after its ordinary bootstrap because its Python gate
looked for `scientific_source_index_sha256` in the link receipt; the field belongs
to the closed build summary. That run remains failed and never requested CMS20.
Its 52 native records and READY-to-BYE closure are preserved. The RUN-only
successor authenticates the original summary and receipt without rewriting
either. A complete retrospective replay of the saved bootstrap preceded the
second worker. No executable was rebuilt for the correction.

## Portable verification

Run with Python's standard library; no Nix environment, compiler, shared library,
installed MadGraph tree, native executable, network or private build directory is
needed:

```sh
python3 -B docs/benchmarkfolder/experiments/normal-checked-cms20-qualification-1/verify.py
```

The five focused replay and tamper tests also run entirely from the retained
packets, once this directory is sealed:

```sh
python3 -B -m unittest discover -s docs/benchmarkfolder/experiments/normal-checked-cms20-qualification-1 -p test_worker_checks.py -v
```

Keep these five indexed ancestor directories alongside this directory when
copying the capsule elsewhere:

| Required ancestor | SHA-256 of `evidence-index.json` |
|---|---|
| `kinematic-invariant-amplitude-qualification-1` | `c04a73c86f6aa3db59e6d5cafd2fb9edf92dd2dbd6d3e418cab462ebbb5327d4` |
| `matched-native-input-qualification-1` | `eacced6cc887ee586d15c4ceade8ac1f089fec0a36b84708f12086d05ba2f8a0` |
| `kinematic-legacy-reference-adjudication-1` | `248bd61e987ac97c81e5397489eb4322ecfbf96cf85a42c84607aef2c013500e` |
| `cms-universal-qed-pole-audit-1` | `1565ac55fe6aeeefbd4f660455c422c4b954737a434d27138cbe8433d9534b77` |
| `cms-universal-qed-single-pole-audit-1` | `99603d41d662e90f071b8052eb4303bcba8af451c5c96654febc276e620ad11e` |

The verifier authenticates the indexed stored and decompressed bytes, source
seals, closed command/log inventories, source inverses and donor ancestry. It
then replays the retained original scientific comparator using read-only byte
callbacks: lifecycle, input geometry, failure/rescue sequence, native-input and
numerator joins, full coefficient controls and separate pole formulas. Absolute
historical paths are provenance labels; they cannot fall back to live files.
Historical binaries are deliberately absent. The verifier does not re-observe
the original live binary hashes, process absence, mutex state or hardware
environment; their closed audit receipts remain clearly identified evidence.

## Immutable experiment identities

| Record | SHA-256 |
|---|---|
| Scientific observation source | `7dab8a04296a44e6eef1cf017bee751491220b77804ebe2a34faa328f85cb2c9` |
| Original controller | `fd482474221715397a0dff7b47244ef25a07f077e79b4b8a47c5e3beca398c03` |
| RUN-only successor | `88cc660d0deecc3b8f87eba23bc10deed729a048cb4bddfcbe3ca4265f8830da` |
| Closed build | `3071f3887244a34ebdf134e3867484cae6ddec9a6623309d3668d8817d26f09a` |
| Original link receipt | `f172832d12ef1a568467fe44bd3ec2c80f8f9717c628e1ab689a3773a57e0145` |
| Failed first run | `754078acb9700ffa10b9c040bc6c0de86784e4da66adaaef0e8fdcb8441985af` |
| Successful second run | `046d83232bd40235f2401341c0806fb9446599b321da23e72b663b9430d68aa1` |
| Original checked comparison | `8ff34fa6a4648ebb84cab4b2cb8a0a1d2f3e25591c42b55aa8f398b308a27465` |
| Independent final root review | `226918bd2b26c740d493473aa12384fd9a5859da04c5e14a03e17b23ec954ff5` |

This qualifies only the private CMS20 checked fixture with cache disabled. It
does not promote the implementation to production, qualify the generic exporter,
other points/processes, enabled caching, all reuse/rotation paths, hardware
floating-point traps, or runtime performance. The stage and native-owner receipts
retain conservative numerical-qualification fields; the separately replayed
comparison carries the narrower `host_stability_qualified_this_fixture=true`.
No native action or Git operation is part of portable verification.
