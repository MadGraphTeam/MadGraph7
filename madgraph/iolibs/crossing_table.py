################################################################################
#
# Copyright (c) 2026 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this
# distribution.
#
################################################################################
"""Per matrix element crossing tables.

A crossed subprocess is evaluated by a BASE matrix element through an extended
flavor index

    FLAV_IDX = K * NFLAV + FLAV

where FLAV is the base flavor index (1-based in fortran, 0-based in C++; its
meaning -- physical flavor row or coupling class -- is the backend's) and K a
row of a table generated for that matrix element. Row 0 is the identity, so a
plain flavor index keeps its meaning and ``FLAV_IDX % NFLAV`` is always the
base flavor.

A row is a full slot permutation. It replaces the former ``I*(NEXTERNAL+1)+J``
code, which could only swap incoming leg 1 with final leg I and incoming leg 2
with final leg J: no final<->final reordering, no beam swap, J meaningless for
1->N, and a code space mostly made of inapplicable or unrequested crossings.
The table holds only the permutations its consumers need.

Conventions (0-based slots), for the row that serves a dependent (crossed)
process whose legs come in the dependent's own order ("input slots"):

* ``D[k]``  -- the base slot receiving input slot k;
* ``B[b]``  -- the input slot whose momentum lands in base slot b (``B = D^-1``,
  what APPLY_CROSSING_TABLE and the C++ gather use: ``P(b) = P_IN(B(b))``);
* ``SD[k]`` -- -1 when input slot k sits on the other side of the
  initial/final line than base slot ``D[k]`` (the leg is crossed), else +1;
* ``SB[b] = SD[B[b]]`` -- the same sign indexed by base slot (the NSF flag of
  base slot b, and the sign of the tau helicity map).

The crossed signed PDG of input slot k, for a base row ``pdg``, is
``pdg[D[k]]`` charge-conjugated when ``SD[k] == -1``. Both directions are
stored because the consumers need both: the permutation is generally NOT an
involution (e.g. u u~ > g g off g g > u u~ is the 4-cycle D = [3,2,0,1]).
"""

from __future__ import absolute_import

import itertools
import logging

logger = logging.getLogger('madgraph.crossing_table')

#: Safety net on the bijection enumeration of one crossed process. Every real
#: process stays far below it (the number of bijections is the product of the
#: factorials of the identical-label multiplicities).
MAX_BIJECTIONS = 200000


class CrossingPerm(object):
    """One row of a crossing table (see the module docstring)."""

    __slots__ = ('D', 'B', 'SD', 'SB')

    def __init__(self, D, ninitial):
        D = tuple(int(d) for d in D)
        n = len(D)
        if sorted(D) != list(range(n)):
            raise ValueError('%s is not a permutation' % (D,))
        B = [0] * n
        for k, b in enumerate(D):
            B[b] = k
        self.D = D
        self.B = tuple(B)
        self.SD = tuple(-1 if ((k < ninitial) != (D[k] < ninitial)) else 1
                        for k in range(n))
        self.SB = tuple(self.SD[B[b]] for b in range(n))

    @classmethod
    def identity(cls, nexternal, ninitial):
        return cls(range(nexternal), ninitial)

    def is_identity(self):
        return self.D == tuple(range(len(self.D)))

    def __eq__(self, other):
        return isinstance(other, CrossingPerm) and self.D == other.D

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash(self.D)

    def __repr__(self):
        return 'CrossingPerm(D=%s)' % (list(self.D),)

    def crossed(self, row, anti):
        """The signed PDGs of the dependent process for the base row `row`
        (base slot order), `anti` giving the charge conjugate of a PDG."""
        return tuple(row[self.D[k]] if self.SD[k] == 1 else anti(row[self.D[k]])
                     for k in range(len(self.D)))

    def base_row(self, crossed_row, anti):
        """Inverse of crossed(): the base row a dependent row comes from."""
        n = len(self.D)
        row = [None] * n
        for k in range(n):
            row[self.D[k]] = crossed_row[k] if self.SD[k] == 1 \
                else anti(crossed_row[k])
        return tuple(row)


class CrossingAssignment(object):
    """A dependent (crossed) physical row and the (row K, base flavor) that
    evaluates it. `flav` is 0-based in the backend's flavor convention."""

    __slots__ = ('K', 'flav', 'pdgs')

    def __init__(self, K, flav, pdgs):
        self.K = K
        self.flav = flav
        self.pdgs = tuple(pdgs)

    def index(self, nflav, one_based=False):
        """The extended flavor index of this assignment."""
        return self.K * nflav + self.flav + (1 if one_based else 0)

    def __repr__(self):
        return 'CrossingAssignment(K=%d, flav=%d, pdgs=%s)' % (
            self.K, self.flav, list(self.pdgs))


class CrossingRecord(object):
    """The crossed subprocesses a recorded process (merge_crossing='record')
    resolves to: every physical row, each served by exactly one assignment."""

    __slots__ = ('process', 'labels', 'assignments', 'truncated', 'unserved')

    def __init__(self, process, labels):
        self.process = process
        self.labels = tuple(labels)
        self.assignments = []
        # the permutation enumeration hit MAX_BIJECTIONS: rows may be missing
        self.truncated = False
        # the target rows (build_table) no permutation serves
        self.unserved = []

    def complete(self):
        """Whether the record may be trusted to serve every physical row of
        its process: it got a row, the search for them was not cut, and no
        row it was asked for is left unserved."""
        return bool(self.assignments) and not self.truncated and \
            not self.unserved

    def ids(self, nflav, one_based=False):
        """The distinct extended flavor indices serving this record, in the
        order they first occur."""
        seen, out = set(), []
        for a in self.assignments:
            idx = a.index(nflav, one_based)
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
        return out


class CrossingTable(object):
    """The rows of one matrix element, plus the records they serve."""

    def __init__(self, nexternal, ninitial):
        self.nexternal = nexternal
        self.ninitial = ninitial
        self.rows = [CrossingPerm.identity(nexternal, ninitial)]
        self._index = {self.rows[0].D: 0}
        self.records = []
        # rows that name no crossing (placeholders keeping a numbering): the
        # identity permutation, reported invalid
        self.invalid = set()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, K):
        return self.rows[K]

    def __iter__(self):
        return iter(self.rows)

    def copy(self):
        """An independent table with the same rows (same K) and records."""
        other = CrossingTable(self.nexternal, self.ninitial)
        other.rows = list(self.rows)
        other._index = dict(self._index)
        other.invalid = set(self.invalid)
        other.records = list(self.records)
        return other

    def add(self, perm):
        """Row index of `perm` (a CrossingPerm or a D sequence), appended when
        new."""
        if not isinstance(perm, CrossingPerm):
            perm = CrossingPerm(perm, self.ninitial)
        K = self._index.get(perm.D)
        if K is None:
            K = len(self.rows)
            self.rows.append(perm)
            self._index[perm.D] = K
        return K

    def add_invalid(self):
        """Append a placeholder row that names no crossing; returns its K."""
        K = len(self.rows)
        self.rows.append(CrossingPerm.identity(self.nexternal, self.ninitial))
        self.invalid.add(K)
        return K

    def find(self, perm):
        """Row index of `perm`, or None."""
        D = perm.D if isinstance(perm, CrossingPerm) else tuple(perm)
        return self._index.get(D)

    def valid(self, K):
        return 0 <= K < len(self.rows) and K not in self.invalid

    def has_crossing(self):
        return len(self.rows) > 1

    def complete(self):
        return all(r.complete() for r in self.records)

    def recorded_rows(self):
        """Sorted row indices K > 0 that serve at least one record."""
        return sorted(set(a.K for r in self.records for a in r.assignments
                          if a.K > 0))

    def assignments(self):
        """Every assignment of every record, in record order."""
        return [a for r in self.records for a in r.assignments]

    def demo_ids(self, nflav, one_based=False):
        """One extended flavor index per recorded crossed process -- its first
        physical row -- the beam-swapped partner of one already listed being
        skipped: what a check_sa crossing demo shows."""
        ids, seen = [], set()
        for record in self.records:
            if not record.assignments:
                continue
            first = record.assignments[0]
            if first.K == 0:
                continue
            key = physical_key(first.pdgs, self.ninitial)
            mirror = key
            if self.ninitial == 2:
                mirror = physical_key((first.pdgs[1], first.pdgs[0]) +
                                      tuple(first.pdgs[2:]), self.ninitial)
            if key in seen or mirror in seen:
                continue
            seen.add(key)
            seen.add(mirror)
            index = first.index(nflav, one_based)
            if index not in ids:
                ids.append(index)
        return ids

    # --------------------------------------------------------------------
    # per row data shared by every backend
    # --------------------------------------------------------------------
    def spincol(self, spincol_part):
        """Initial-state spin*colour average of the dependent process served
        by each row: the product of the per-leg spin*colour (conjugation
        invariant) of the base legs the row puts in the initial slots."""
        out = []
        for K, row in enumerate(self.rows):
            if K in self.invalid:
                out.append(0)
                continue
            factor = 1
            for k in range(self.ninitial):
                factor *= spincol_part[row.D[k]]
            out.append(factor)
        return out

    def flat(self, attr, offset=0):
        """The `attr` ('D', 'B', 'SD' or 'SB') of every row, flattened
        K*NEXTERNAL + slot, with `offset` added (1 for fortran slot numbers,
        only meaningful for D/B)."""
        return [v + offset for row in self.rows for v in getattr(row, attr)]


# ------------------------------------------------------------------------------
# helpers on labels
# ------------------------------------------------------------------------------
def make_anti(model):
    """PDG -> PDG of the antiparticle, following the model (a self-conjugate
    particle stays itself); unknown codes fall back to the naive sign flip."""
    particle_dict = model.get('particle_dict') if model else {}
    cache = {}

    def anti(pdg):
        try:
            return cache[pdg]
        except KeyError:
            pass
        try:
            value = particle_dict[pdg].get_anti_pdg_code()
        except (KeyError, AttributeError, TypeError):
            value = -pdg
        cache[pdg] = value
        return value
    return anti


def make_leg_matches(model):
    """(label, pdg) -> does the physical/merged `pdg` instantiate `label`?

    Equal ids always match; otherwise one of the two may be a merged label
    (81 = _quark, ...) covering the other's flavor, with the same sign."""
    merged = (model.get('merged_particles') or {}) if model else {}

    def matches(label, pdg):
        if label == pdg:
            return True
        if (label > 0) != (pdg > 0):
            return False
        a, b = abs(label), abs(pdg)
        return (a in merged and b in merged[a]) or \
               (b in merged and a in merged[b])
    return matches


def iter_bijections(dep_labels, base_labels, ninitial, anti, fixed=(),
                    seed=None, limit=MAX_BIJECTIONS, compatible=None,
                    state=None):
    """Yield every label-consistent D (tuple, dependent slot -> base slot).

    Input slot k may take base slot b when the base leg's label, conjugated if
    the two sit on different sides, is compatible with the dependent label
    (``compatible(dep_label, base_label_as_seen)``, equality by default).
    `fixed` lists the slots that must map onto themselves on the same side
    (the leaves of a decay block: a crossing never splits a resonance). The
    `seed` permutation, when label-consistent, is yielded first. Past `limit`
    permutations the enumeration stops, and ``state['truncated']`` is set
    when a `state` dict is given.
    """
    n = len(dep_labels)
    if compatible is None:
        def compatible(a, b):
            return a == b
    fixed = set(fixed)
    options = []
    for k in range(n):
        if k in fixed:
            options.append([k])
            continue
        opts = []
        for b in range(n):
            if b in fixed:
                continue
            seen_as = base_labels[b] if (k < ninitial) == (b < ninitial) \
                else anti(base_labels[b])
            if compatible(dep_labels[k], seen_as):
                opts.append(b)
        options.append(opts)

    produced = 0
    if seed is not None:
        seed = tuple(seed)
        if len(seed) == n and sorted(seed) == list(range(n)) and \
                all(seed[k] in options[k] for k in range(n)):
            produced += 1
            yield seed
    else:
        seed = None

    used = [False] * n
    current = [0] * n

    def rec(k):
        if k == n:
            yield tuple(current)
            return
        for b in options[k]:
            if used[b]:
                continue
            used[b] = True
            current[k] = b
            for d in rec(k + 1):
                yield d
            used[b] = False

    for D in rec(0):
        if D == seed:
            continue
        produced += 1
        if produced > limit:
            logger.warning('Crossing table: more than %d label-consistent '
                           'permutations, the enumeration is truncated.'
                           % limit)
            if state is not None:
                state['truncated'] = True
            return
        yield D


def physical_key(pdgs, ninitial):
    """Key identifying a physical partonic process: the initial legs in order,
    the final legs as a multiset."""
    return (tuple(pdgs[:ninitial]), tuple(sorted(pdgs[ninitial:])))


def pairing_from_record(base_perm, crossed_perm):
    """The diagram pairing MultiProcess.cross_amplitude uses for a recorded
    crossing, as D (0-based): base leg org_perm[i] becomes crossed leg
    new_perm[i] (the legs sharing sorted outgoing-id position i)."""
    D = [None] * len(crossed_perm)
    for org, new in zip(base_perm, crossed_perm):
        D[new - 1] = org - 1
    if None in D:
        return None
    return tuple(D)


# ------------------------------------------------------------------------------
# builders
# ------------------------------------------------------------------------------
def applicable_perms(nexternal, ninitial, fixed=()):
    """Every ordered choice of the base legs that start in the initial state:
    the permutations a standalone user may ask for without a record
    (``--crossing_table=all``). The identity comes first.

    Every other leg keeps its slot, except that a base initial leg sent to the
    final state takes the final slot vacated by the leg that replaces it --
    the slot of the leg it is swapped with (base initial leg i goes where the
    leg now in slot i came from), or, when that leg was itself initial (a beam
    swap combined with a crossing), the one vacated slot left. So the table
    holds every transposition pair the former ``I*(NEXTERNAL+1)+J`` code could
    name, plus the beam swaps it could not. Decay-block leaves (`fixed`) never
    move: a crossing never splits a resonance."""
    fixed = set(fixed)
    if any(k < ninitial for k in fixed):
        raise ValueError('an initial leg cannot be a decay-block leaf')
    movable = [b for b in range(nexternal) if b not in fixed]
    out = []
    for initial in itertools.permutations(movable, ninitial):
        D = list(range(nexternal))
        D[:ninitial] = initial
        leaving = [i for i in range(ninitial) if i not in initial]
        # transposition first: slot initial[i] takes base leg i back
        rest = []
        for i in range(ninitial):
            if initial[i] < ninitial:
                continue
            if i in leaving:
                D[initial[i]] = i
                leaving.remove(i)
            else:
                rest.append(initial[i])
        assert len(rest) == len(leaving)
        for slot, leg in zip(rest, leaving):
            D[slot] = leg
        out.append(tuple(D))
    identity = tuple(range(nexternal))
    out.sort(key=lambda D: (D != identity, D))
    return out


def build_table(nexternal, ninitial, base_labels, base_entries, records,
                model, fixed=(), all_applicable=False):
    """Crossing table of a folding output.

    base_labels  -- the base legs' ids (merged labels allowed), base order;
    base_entries -- every physical base row as (flav, pdg tuple), flav being
                    the 0-based flavor index that evaluates it in the backend's
                    convention (several rows may share a flav: a coupling
                    class);
    records      -- [(process, dep_labels, seed_D)] the recorded crossed
                    processes, their leg ids in their own order (decays
                    expanded to leaves) and the diagram pairing (or None); a
                    fourth element, when not None, lists the TARGET rows of
                    the record -- see below;
    fixed        -- slots that no crossing moves (decay-block leaves);
    all_applicable -- also add every applicable_perms() row.

    Every physical row of every record is served by exactly one assignment.
    Rows already in the table are preferred, then the recorded pairing, then
    the other label-consistent permutations in lexicographic order.

    A record with target rows is served exactly those, each in its own slot
    order (solve_row): a consumer that passes the momenta of a crossed
    process in a given order -- the mg7 subprocess entries, whose channels
    are the crossed process's own -- must get a row feeding its slots as they
    come, not merely one serving the same physical process (the two identical
    final antiquarks of q~ q~ > w+ q~ q~ swapped: the base then fills the
    amp2 of diagrams the crossed process does not have in that order). A
    target no permutation serves is listed in the record's `unserved`."""
    anti = make_anti(model)
    leg_matches = make_leg_matches(model)
    table = CrossingTable(nexternal, ninitial)
    if all_applicable:
        for D in applicable_perms(nexternal, ninitial, fixed):
            table.add(D)

    def compatible(dep_label, base_label):
        return leg_matches(dep_label, base_label) or \
            leg_matches(base_label, dep_label)

    for entry in records:
        process, dep_labels, seed = entry[:3]
        targets = entry[3] if len(entry) > 3 else None
        record = CrossingRecord(process, dep_labels)
        table.records.append(record)
        if len(dep_labels) != nexternal:
            continue
        if targets is not None:
            for target in targets:
                prefer = [row.D for K, row in enumerate(table.rows)
                          if K not in table.invalid]
                if seed is not None:
                    prefer.append(tuple(seed))
                hit = solve_row(target, base_labels, base_entries, ninitial,
                                model, fixed=fixed, prefer=prefer)
                if hit is None:
                    record.unserved.append(tuple(target))
                    continue
                perm, flav = hit
                K = table.find(perm)
                if K is None:
                    K = table.add(perm)
                record.assignments.append(CrossingAssignment(K, flav, target))
            continue
        state = {}
        candidates = list(iter_bijections(dep_labels, base_labels, ninitial,
                                          anti, fixed=fixed, seed=seed,
                                          compatible=compatible, state=state))
        record.truncated = bool(state.get('truncated'))
        known = [D for D in candidates if table.find(D) is not None]
        rest = [D for D in candidates if table.find(D) is None]
        taken = set()
        for D in known + rest:
            perm = CrossingPerm(D, ninitial)
            K = table.find(perm)
            for flav, row in base_entries:
                xrow = perm.crossed(row, anti)
                if not all(leg_matches(dep_labels[k], xrow[k])
                           for k in range(nexternal)):
                    continue
                key = physical_key(xrow, ninitial)
                if key in taken:
                    continue
                taken.add(key)
                if K is None:
                    K = table.add(perm)
                record.assignments.append(CrossingAssignment(K, flav, xrow))
    return table


def solve_row(target, base_labels, base_entries, ninitial, model, fixed=(),
              prefer=(), accept=None):
    """(perm, flav) such that ``perm.crossed(row) == target`` EXACTLY for a
    base row of flavor `flav`, or None.

    Used by the madevent routing, where the dependent module passes its
    momenta in its own slot order, so the match is slot by slot (not merely
    the same physical process). `prefer` lists D tuples tried first (rows the
    base table already has); `accept(perm)` may veto a permutation (the
    routing asks for a genuine crossing: one that moves a leg across)."""
    anti = make_anti(model)
    leg_matches = make_leg_matches(model)
    target = tuple(target)
    tried = set()
    # the base row a permutation would have to start from, looked up rather
    # than every base row crossed and compared (first flavor wins)
    flav_of = {}
    for flav, row in base_entries:
        flav_of.setdefault(tuple(row), flav)

    def solve(D):
        perm = CrossingPerm(D, ninitial)
        if accept is not None and not accept(perm):
            return None
        flav = flav_of.get(perm.base_row(target, anti))
        return None if flav is None else (perm, flav)

    for D in prefer:
        D = tuple(D)
        tried.add(D)
        hit = solve(D)
        if hit is not None:
            return hit

    def compatible(dep_pdg, base_label):
        # the target carries physical PDGs, the base labels may be merged
        return dep_pdg == base_label or leg_matches(base_label, dep_pdg)
    for D in iter_bijections(target, base_labels, ninitial, anti,
                             fixed=fixed, compatible=compatible):
        if D in tried:
            continue
        hit = solve(D)
        if hit is not None:
            return hit
    return None
