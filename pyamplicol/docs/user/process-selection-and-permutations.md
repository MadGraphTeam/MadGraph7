---
title: "Process Selection and Permutations"
nav_order: 2
parent: "Configuration"
---
<!-- SPDX-License-Identifier: 0BSD -->

# Process Selection and Permutations

pyAmpliCol separates two related operations:

- **generation requests** decide which process representatives an artifact
  contains;
- **runtime selection** chooses one representative and the public particle
  order in which the caller wants to use it.

This lets one evaluator serve equivalent incoming and outgoing orderings
without asking users to memorize generated alias names.

## A representative is not an output directory

An artifact may contain one process or many. Each retained concrete process
has a stable ID, such as `p_p_to_z_j_j_4`; the artifact itself is still loaded
from one directory:

```console
pyamplicol inspect artifacts/pp_zjj
pyamplicol inspect artifacts/pp_zjj --process p_p_to_z_j_j_4
```

The same process can be selected with a readable expression:

```console
pyamplicol inspect artifacts/pp_zjj --process 'd d~ > g z g'
```

Stable process IDs are selectors, not paths below `artifacts/pp_zjj`.

## How a selector is resolved

Rusticol applies a deterministic precedence order:

1. stable process ID;
2. explicit alias ID;
3. exact stored process expression;
4. exact explicit-alias expression;
5. one unique permutation-equivalent expression.

Case and whitespace in a process expression are normalized. At the final step,
the incoming and outgoing particle multisets are matched independently.

For example, an artifact may store the representative ordering
`d d~ > Z g g`, while the caller requests:

```text
d d~ > g z g
```

The outgoing multisets are equal, so the request selects the same evaluator and
exposes the requested public ordering. No pre-generated alias is required.

If more than one stored representative could match, selection fails and lists
the candidate stable IDs. pyAmpliCol never silently picks an ambiguous process.

## The side-preserving rule

Permutations are allowed **within** each side of `>`:

```text
d d~ > Z g g   ->   d~ d > g Z g       allowed
```

A leg may never cross the process boundary:

```text
d d~ > Z g g   ->   d Z > d~ g g       not equivalent
```

Repeated identical particles are matched by a deterministic first-unused
rule. This makes the mapping reproducible even when several `g` legs are
indistinguishable by particle name.

## What is remapped

The selected ordering is not cosmetic. One representative-to-public
permutation is applied centrally to:

- input momenta and external-particle PDGs;
- external-particle metadata and momentum slots;
- helicity vectors, IDs, representatives, and selectors;
- LC color-flow words, IDs, replay labels, and selectors;
- reduction groups and resolved-output metadata;
- bundled validation kinematics;
- compiled, eager, recurrence, and on-the-fly f64 execution;
- Python exact execution when retained expressions support it.

Consequently, all public inputs and outputs describe the ordering written in
the selector:

```python
from pyamplicol import Runtime

runtime = Runtime.load("artifacts/pp_zjj", process="d d~ > g z g")
print(runtime.physics.process)  # d d~ > g Z g
print([particle.name for particle in runtime.physics.external_particles])
```

The process key continues to identify the representative evaluator. Generic
permutation metadata records the active public mapping; explicit aliases also
retain their alias identity.

## Kinematics follow the requested order

Momenta always have shape:

```text
[point][external particle][E, px, py, pz]
```

The external-particle dimension follows `runtime.physics.external_particles`,
which in turn follows the selected expression. For `d d~ > g z g`, a custom
five-particle point is therefore ordered as:

```json
[
  ["250.0", "0", "0", "250.0"],
  ["250.0", "0", "0", "-250.0"],
  ["204.406", "204.406", "0", "0"],
  ["91.188", "0", "0", "0"],
  ["204.406", "-204.406", "0", "0"]
]
```

The first two rows are the incoming `d` and `d~`; the remaining rows are the
outgoing `g`, `z`, and `g` in exactly that order.

Generated standalone drivers accept the file with `--kinematics`:

```console
python artifacts/pp_zjj/API/python/check_standalone.py \
  --process 'd d~ > g z g' \
  --kinematics my_sample_point.json \
  --precision 80
```

The file may contain one point as `[external][4]` or a singleton batch as
`[[external][4]]`. Components may be JSON numbers or decimal strings. The
Python exact path preserves decimal strings without an intermediate f64
conversion; native drivers parse the same representation as f64. Booleans,
non-finite values, multiple points, and incorrect shapes are rejected.

When `--kinematics` is omitted, the bundled representative validation point is
automatically reordered to the requested public expression.

## Helicity and color selectors follow the requested order

Always obtain stable IDs after selecting the desired process ordering:

```python
runtime = Runtime.load("artifacts/pp_zjj", process="d d~ > g z g")

helicity = runtime.physics.helicity_ids[0]
flow = runtime.physics.color_flow_ids[0]
value = runtime.evaluate(
    momenta,
    helicities=[helicity],
    color_flows=[flow],
)
```

A helicity ID encodes values in the public external order. An LC flow word and
its stable ID likewise use public leg labels. Reduction metadata points to the
remapped public IDs, so `evaluate_resolved()` remains consistent with
`evaluate()` after any permitted permutation.

Color-flow selection is available only at LC. NLC and full-color artifacts
have a contracted output axis and accept helicity selectors only. See
[Runtime and Selectors](runtime-and-selectors.md).

## Generation-time multiprocess expansion

One request may expand through multiparticle labels:

```toml
[process]
entries = [{ expression = "p p > Z j j" }]
flavor_scheme = 2
max_quark_lines = 2

[process.multiparticles]
p = ["d", "d~", "g"]
j = ["d", "d~", "g"]
```

For the shipped serialized Standard Model this request produces 19 ordered
candidates. Incoming/outgoing permutation equivalence collapses them into eight
representative classes. Seven representatives have tree-level amplitudes and
are stored. The remaining class, `g g > Z g g`, is loop-induced and is
reported as unsupported at tree level rather than emitted as a zero process.

Those seven evaluators cover all 18 ordered tree-level channels, including
public orderings not stored as explicit aliases.

## The model-generic `all` label

Every model supplies `all` as its declaration-ordered valid physical external
states. It includes propagating, physical-spin particles and excludes ghosts,
Goldstones, non-propagating records, and auxiliary states.

User definitions merge over model defaults, so defining `p` or `j` leaves
`all` available. Defining `all` explicitly replaces the default.

Inspect a bounded built-in expansion before generating it:

```console
pyamplicol model processes "p p > all all" \
  --model built-in-sm \
  --flavor-scheme 1 \
  --max-quark-lines 0
```

`all` products can expand combinatorially, especially for large UFO models.
Use a narrower custom label for routine production work.

## Explicit process sets and aliases in Python

When stable names are useful, construct a `ProcessSet`:

```python
from pyamplicol import ProcessRequest, ProcessSet

processes = ProcessSet(
    requests=(
        ProcessRequest.parse("u u~ > Z g", name="uubar_Zg"),
        ProcessRequest.parse("u u~ > Z g g", name="uubar_Zgg"),
    )
)
```

Explicit `ProcessAlias` records remain available for named public interfaces,
but ordinary callers do not need to enumerate every incoming or outgoing
ordering. Runtime inference uses the same side-preserving mapping.

## Failure guide

| Message or symptom | Meaning | Action |
| --- | --- | --- |
| Several candidate stable IDs are listed | The expression is permutation-equivalent to more than one stored representative | Select the intended stable ID explicitly |
| No process matches | The artifact does not contain the requested particle multisets, or a leg crossed `>` | Run `pyamplicol inspect ARTIFACT` and correct the request |
| Kinematics shape mismatch | The point does not have one four-vector per public external particle | Follow `runtime.physics.external_particles` or the `--process` order |
| Unknown helicity or flow ID | The ID came from another ordering or process | Read IDs after loading the requested process ordering |
| Color-flow selector rejected | The artifact is NLC/full, not LC | Select helicities only, or generate an LC artifact |

## See also

- [Models and Processes](models-and-processes.md)
- [Runtime and Selectors](runtime-and-selectors.md)
- [Native APIs](native-apis.md)
- [Artifacts and Portability](artifacts-and-portability.md)
