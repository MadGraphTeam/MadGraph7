# Recurrence And AmpliCol Architecture Validation

This read-only validation revisits
`RECURRENCE_AMPLICOL_ARCHITECTURE_AUDIT.md` after the first executable
topology-replay runtime and the runtime-selector dispatch correction. It
compares the current branch with original AmpliCol's mode-1 and mode-2
recurrences.

## Verdict

Topology replay now implements AmpliCol-like local source-ancestry sharing.
It does not store or execute an independent recurrence for every complete
helicity assignment.

For built-in-SM `d d~ > Z g`, the retained schedule contains:

| Quantity | Count |
|---|---:|
| Source variants | 9 |
| Non-source currents/finalizations | 22 |
| Total currents | 31 |
| Contributions | 34 |
| Closure destinations | 12 |
| Full graph passes for the helicity sum | 1 |

The nine source variants are the two quark, two antiquark, three vector-boson,
and two gluon states. Complete helicity assignments occur only at amplitude
destinations. Internal current identity contains only its local source-state
ancestry. This closes the topology-replay helicity-identity gap identified by
the original audit.

Dynamic LC color state is also part of current identity and is updated by
authenticated transition witnesses. The corresponding topology-replay gap is
closed.

## Runtime Gap

The approximately 14 microsecond per-point `Zg` recurrence cost is not caused
by twelve independent helicity recurrences. The runtime executes one compact
graph pass per replay target and reduces all twelve live helicity destinations
afterward.

The remaining owners are generic execution plumbing:

- each homogeneous packet gathers component-major currents into lane-major
  prepared-kernel inputs and scatters results back;
- active current and amplitude workspaces are cleared for every tile;
- even an identity replay copies source values and external momenta through
  replay scratch;
- momentum linear forms are evaluated once per current rather than once per
  unique form;
- prepared-backend dispatch performs lookup and contract checks per packet.

The schedule is already grouped by `(stage, role, prepared-kernel)` and does
not invoke the backend once per point, contribution, current, or helicity.
Further optimization must therefore be driven by packet, lane, gather,
scatter, finalization, and momentum-form counters.

## Remaining Architecture Gap

All-flow union is not yet an honest AmpliCol mode-2 analogue. Its source
construction still materializes numerical source states and its runtime loops
over physical helicities. Before making a mode-2 performance claim, it must:

1. use one runtime full-state source domain per external leg;
2. remove numerical helicity from source and internal-current identity;
3. demonstrate schedule-size invariance as retained helicity coverage changes;
4. execute one selected runtime helicity without retaining a second
   helicity-expanded graph.

## Comparator Correction

At the shared validation point, recurrence and eager agree with the preserved
AmpliCol oracle for `Z` and `Zg`. The temporary complete compiled-JIT
comparator has a left-chiral selector/scaling defect and must not be used as a
recurrence physics oracle until fixed. No recurrence coupling compensation is
justified by those compiled-artifact mismatches.

## Required Next Evidence

- graph passes, replay targets, source fills/copies, backend calls and lanes;
- contributions, attachments, finalizations, and closures by stage/kernel;
- gather/scatter bytes and timings, unique momentum forms, and warmed
  allocations;
- built-in/UFO-SM component parity for representative and replayed flows;
- reflected pure-gluon, same-flavour, and three-open-line closure canaries;
- the `q q~ > Z + n g` scaling ladder through `n=6`;
- all-flow-union schedule-size invariance before runtime qualification.

This validation does not declare the recurrence feature complete. It confirms
that topology replay has the intended compact AmpliCol-like state identity and
identifies all-flow union and runtime packet efficiency as the remaining
architectural work.
