# Compiled non9 dispatch and input parity

This portable follow-up records one closed BUILD and one closed recording-unit
RUN for the unapplied QP-mass source2 patch. All126 public/private helper pairs
(252 calls across seven actual helper families) and two chooser-only controls
passed. This qualifies compiled dispatch and synthetic input/output parity.
No real reducer or physical amplitude was evaluated.

| Recording target | DP input signature | QP input signature | Requested-mode detail |
|---|---:|---:|---|
| CutTools target1 |84 |0 |56 of those84 calls request CTMODE4/5; this is its unchanged CTLOOP argument signature, not a real-reducer precision result |
| Ninja target6 |28 |28 |DP signature in modes1/2; QP momenta with legacy DP squared masses in modes4/5 |
| COLLIER target7 |28 |0 |DP-only recording detector and reduction stub; no real initialization or cache qualification |
| NinjaRS target8 |28 |56 |Includes explicitly controlled requested9/actual8 selection |

Requested CTMODE counts are1:56,2:56,4:70,5:70. The unit uses the actual chooser
for196 helper calls and an explicitly controlled selector for56. All6,425 raw
records match the frozen integer dyadic oracle. All507 observed IEEE-state
records contain zero rounding-code, flag and halting bits. The finite MP mass
poison and literal source guard are diagnostics; zero observed flags do not
prove instruction absence or qualify trap freedom.

BUILD used six original-flag Fortran compiles and one link, with21 prelink
queries, a loader listing and a symbol query:30 commands, all returned zero.
The14 actual helper definitions,23 recorded LOAD bindings and five-provider
system/compiler subset were checked. Both stages closed without survivors under
180s total caps including60s cleanup,4GiB address/256MiB file limits, core dumps0,
nonexclusive CPU12 and the shared mutex. Runtime evidence is the recorded BUILD
listing/provider identities and unchanged RUN ELF; no per-RUN maps were added.

The exact source2 index is `4de119d2…`; BUILD terminal is `fb947cbd…`; RUN terminal
is `c0bd0cce…`; the238,802-byte raw journal is `39ec847e…`. Full hashes and exact
commands are retained. Source1 `e5ad27c3…` was rejected during pre-native review
because a successful Popen followed by a raising ownership constructor could
skip cleanup. It never ran. The minimal ancestor subset, rejection receipt and
source2 correction are preserved; this is not a native-failure claim.

`source/` contains exact source2, including dormant historical controllers;
do not execute them from this capsule. `parent-subset/` contains only six exact
text members and the authenticated index of the already published mass parent,
enough to reproduce fourteen helper bodies and the chooser. It does not repeat
that campaign. Native executables, objects and compiler modules are excluded;
sixteen identities remain in the index. Larger text uses deterministic gzip.

Run with a standard-library Python3 from any directory:

```
python3 -B replay.py --index-sha256 <SHA256-of-index.json>
python3 -B test_replay.py
python3 -O -B test_replay.py
```

Replay authenticates every owned byte, nested source/member identity, closed
stage and command log. It reruns the frozen parser/oracle against the actual
journal and checks recorded symbols, input signatures and requested modes.
Semantic replay has no filesystem access beyond preloaded contained text and
blocks process, native-library, network, signal and mutation APIs. Tests include
relocation, payload/claim/native-identity tampering, symlink rejection and guarded
effect probes. No historical provider or binary is opened or executed.

Physical `UseQPIntegrandForTIRust=False`, complex-mass EW, loop-induced physical
qualification, independent finite truth, Hua-Sheng's local performance acceptance
and production promotion remain open. Existing mass and NinjaRS reports are
unchanged.
