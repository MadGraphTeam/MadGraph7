# Closed legacy Ninja/NinjaRS reference probe

Both reference workers completed their protocols, but **neither provides a
valid numerical reference** for the CMS point20 amplitude. The retained run
exits 1 because the eligibility gate fails; every coefficient comparison is
blocked. This packet preserves the failed probe without promoting its output
buffers, the private candidate, or a production change.

Two fresh fixed-QP workers ran in order NinjaRS (8), then Ninja (6), each with
a point1 bootstrap and point20 measurement for
`e+ e- > mu+ mu- [virt=QED]`. Four outer amplitude calls and four READY/DONE
runtime-map captures completed. The usage records show 104 completed native
adapter calls: 26 loop groups for each phase of each worker. All observed
calls used mode4 and reported QP interface storage width128/digits113; no DP
native adapter call was observed. These records establish the API precision
route, not every library's internal working precision.

- NinjaRS returned status1 for five point20 box groups: 3, 4, 8, 12 and 24.
  The measurement also set the software QP failure flag.
- Ninja returned status0 for all observed adapter calls, but its final finite,
  single-pole and double-pole words were nonfinite. The measurement set the
  host software DP failure flag despite the absence of any native DP call.
  That flag must not be interpreted as evidence of a DP reduction.

Both bootstrap post-call software flags are clear. The whole-worker gates
nevertheless reject both workers, so **no bootstrap numerical comparison was
attempted either**. Raw coefficient words and decimal output text are retained;
verification classifies nonfinite exponent fields and failure status without
converting failed coefficients into numerical values. It performs no invalid
coefficient arithmetic.

## Source, build and input scope

The source seal is
`0a0dba9baf12ff5a988e4d459141a48140f285a396658ee10f7f09099848dbd8`
(29 payloads). A diagnostic driver and adapter entry/exit logging extend the
earlier original build. The shared legacy kinematic policy remains in use;
this is not a matched-input adjudication of the candidate's certified
spacelike policy. Native usage traces record route, group and return status,
not complete kinematic input matrices. No scalar cutoff was changed.

The closed build
`08a5b485954e20dc25fcffc2c26edddd73520b982e9daa380e1d5eca06709f22`
contains 138 payloads and 64 owned commands, including one loader query.
Only the driver and `CT_interface` objects were recompiled; 205 objects,
seven existing modules and four libraries were preserved. One executable
uses fifteen recorded runtime providers. The build itself made no amplitude
or tensor call and rebuilt no scientific library.

The closed run
`b94dca1a42b7acd2c835ce6b1bf04fe9036b0d64755ec1348240235a76b06462`
contains 43 payloads and four owned commands. The worker commands finish
successfully while the controller records the failed scientific gate. Both
workers were reaped; historical terminal/independent reviews record all six
owned identities absent, the shared lock free and protected inputs unchanged.
No retry or extra evaluation belongs to this probe. Elapsed times are
accountability records, not performance measurements.

All source, cards, point files, commands, stdout, maps, raw status records,
failure buffers and closed reviews are retained through explicit sealed file
lists. Original native binaries are not stored. Their hashes and provider
checks are historical identities; portable verification does not re-open
external libraries or inspect today's live processes. The earlier 269-source
inventory remains historical; no fresh reparse of all original files is claimed.
The root review records independent rehashing of 802 external source, object,
library, provider and executable files at its review boundary; the portable
verifier authenticates that receipt and does not repeat those external reads.

The source proposal and contemporaneous reviews remain verbatim. Their plans
and outstanding-item descriptions are historical. This probe by itself makes
no finite or pole accuracy conclusion, checked-stability qualification or
performance claim, and does not supersede separate universal-pole audits.

## Portable verification

Keep `../kinematic-invariant-amplitude-qualification-1` beside this directory,
with seal
`c04a73c86f6aa3db59e6d5cafd2fb9edf92dd2dbd6d3e418cab462ebbb5327d4`.
Its own broader verifier needs its separately declared dependencies.

From the repository directory:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/kinematic-legacy-reference-adjudication-1/verify.py
```

The standard-library-only verifier authenticates the packet and public parent,
joins source/build/module/object/library/provider records, reparses usage,
driver state and raw word records, and checks cards, worker protocols and
runtime maps. The comparison replay keeps coefficient decoding forbidden for
both ineligible workers. FIFO protocol messages survive as parsed result
records rather than a raw FIFO transcript; the raw state/bit/output files are
retained separately.

`capture.py` documents the explicit closed-source/receipt selection and
deterministic gzip retention. No retained native controller, compiler, loader,
amplitude program or external binary path is executed by verification.
