"""Pure-Python reference and archive audit for FKSSectorLearning=29.

Independent oracle for the tiled S-event maps (no NumPy/SciPy). Probe banks
use nested lists b[N][16][F] and r[N][S][16][F]; they already contain the
probe's 1/16 weight. Maps are lists maps[S][16] of zero-based target tiles.
The tile loss of one orbit is sum over nodes and canonical flavours of
max(-(B(node) + sum_s R_s(map_s(node))), 0).

The audit replays the native10 epoch archive (sector_learning_online.bin) and
the frozen SCL10 table in mint_grids: static gauges, capacity/support policy,
every phase and assignment candidate trace, the strict TRAIN-loss selection,
the half-weight immediate-previous replay bank and the S-only grid proxy.
Assignment optimality is checked with a dual (potential) certificate. Loss
comparisons allow round-off, since the producer may sum in another order.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import math
import struct

MAGIC = 73110620
COEFFICIENT = .5
GEOMETRIES = ((2, 2, 1), (4, 2, 1), (2, 4, 1), (4, 4, 1))
SHAPES = ((1, 1), (2, 2), (4, 2), (2, 4), (4, 4))
RTOL, ATOL = 2.e-11, 2.e-10


# ---------------------------------------------------------------- arrays

class Array3:
    """Fortran-ordered 3D array: element (i,j,k) at i+n1*(j+n2*k)."""

    def __init__(self, shape, data=None, fill=0.):
        self.shape = tuple(int(n) for n in shape)
        size = self.shape[0] * self.shape[1] * self.shape[2]
        self.data = [fill] * size if data is None else list(data)
        if len(self.data) != size:
            raise ValueError('Array3 data size differs from its shape')

    def index(self, i, j, k):
        return i + self.shape[0] * (j + self.shape[1] * k)

    def __getitem__(self, ijk):
        return self.data[self.index(*ijk)]

    def __setitem__(self, ijk, value):
        self.data[self.index(*ijk)] = value


def _finite(values):
    return all(math.isfinite(v) for v in values)


def _ordered_sum(values):
    """Left-to-right float sum, matching the Fortran accumulation order.

    The built-in sum() is compensated from Python 3.12 on, which can differ
    from the Fortran by an ulp and flip strict tie decisions.
    """
    total = 0.
    for value in values:
        total += value
    return total


def _close(a, b, reason):
    a, b = _flat(a), _flat(b)
    if len(a) != len(b) or any(abs(x - y) > ATOL + RTOL * abs(y) for x, y in zip(a, b)):
        raise ValueError(reason)


def _flat(value):
    if isinstance(value, Array3):
        return value.data
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_flat(item) if isinstance(item, (list, tuple, Array3)) else [item])
        return out
    return [value]


def _tolerance(*values):
    return RTOL * max(1., *(abs(v) for v in values))


def arrays(b, r):
    """Validate b[N][16][F], r[N][S][16][F]; return float copies."""
    b = [[[float(x) for x in node] for node in orbit] for orbit in b]
    r = [[[[float(x) for x in node] for node in sector] for sector in orbit] for orbit in r]
    if len(r) != len(b):
        raise ValueError('Bank orbit counts differ')
    if b:
        nf, ns = len(b[0][0]), len(r[0])
        if nf < 1 or ns < 1:
            raise ValueError('Bank needs flavours and sectors')
        for orbit_b, orbit_r in zip(b, r):
            if (len(orbit_b) != 16 or any(len(node) != nf for node in orbit_b) or len(orbit_r) != ns or
                    any(len(sector) != 16 or any(len(node) != nf for node in sector) for sector in orbit_r)):
                raise ValueError('Require b[N][16][F],r[N][S][16][F]')
            if not _finite(_flat(orbit_b)) or not _finite(_flat(orbit_r)):
                raise ValueError('Nonfinite bank')
    return b, r


def dims(b, r):
    return len(b), (len(r[0]) if r else 0), (len(b[0][0]) if b else 0)


# ------------------------------------------------------------ tile maps

def permutation(maps, nodes=16):
    maps = [[int(x) for x in row] for row in maps]
    if any(sorted(row) != list(range(nodes)) for row in maps):
        raise ValueError('Require a zero-based complete bijection per sector')
    return maps


def geometry(shape):
    shape = tuple(int(x) for x in shape)
    if shape not in GEOMETRIES:
        raise ValueError('Require one of 221,421,241,441 deployed geometries')
    return shape


def lift(coarse, shape):
    """Lift whole coarse-cell transport to the fine 4x4 tiles."""
    gx, gy, _ = geometry(shape)
    coarse = permutation(coarse, gx * gy)
    hx, hy = 4 // gx, 4 // gy
    result = []
    for row in coarse:
        fine = []
        for node in range(16):
            x, y = divmod(node, 4)
            tx, ty = divmod(row[gy * (x // hx) + y // hy], gy)
            fine.append(4 * (tx * hx + x % hx) + ty * hy + y % hy)
        result.append(fine)
    return result


def extract(fine, shape):
    fine = permutation(fine)
    gx, gy, _ = geometry(shape)
    hx, hy = 4 // gx, 4 // gy
    coarse = []
    for row in fine:
        cells = [0] * (gx * gy)
        for x in range(gx):
            for y in range(gy):
                tx, ty = divmod(row[4 * x * hx + y * hy], 4)
                cells[gy * x + y] = gy * (tx // hx) + ty // hy
        coarse.append(cells)
    permutation(coarse, gx * gy)
    if lift(coarse, shape) != fine:
        raise ValueError('Fine map is not representable in this coarse geometry')
    return coarse


def phase_map(code):
    if isinstance(code, bool) or not isinstance(code, int) or not 0 <= code < 16:
        raise ValueError('Invalid fine phase code')
    return [4 * ((d // 4 + code // 4) % 4) + (d % 4 + code % 4) % 4 for d in range(16)]


def legal_phase_codes(shape):
    gx, gy, _ = geometry(shape)
    return [c for c in range(16) if (gx * (c // 4)) % 4 == 0 and (gy * (c % 4)) % 4 == 0]


# ---------------------------------------------------------------- losses

def _base(b, r, maps, skip=None):
    """B plus every sector except skip at its mapped tile, sector order."""
    out = []
    for orbit_b, orbit_r in zip(b, r):
        rows = []
        for node in range(16):
            values = list(orbit_b[node])
            for s, sector in enumerate(orbit_r):
                if s != skip:
                    tile = sector[maps[s][node]]
                    for f in range(len(values)):
                        values[f] += tile[f]
            rows.append(values)
        out.append(rows)
    return out


def losses(b, r, maps):
    b, r = arrays(b, r)
    maps = permutation(maps)
    if b and len(maps) != len(r[0]):
        raise ValueError('Map sector count differs')
    return [_ordered_sum(max(-v, 0.) for node in orbit for v in node) for orbit in _base(b, r, maps)]


def objective(b, r, maps):
    return _ordered_sum(losses(b, r, maps))


def _sector_objectives(b, r, maps, sector, columns):
    """Objective for each candidate map column of one sector."""
    base = _base(b, r, maps, skip=sector)
    results = []
    for column in columns:
        answer = 0.
        for orbit_base, orbit_r in zip(base, r):
            tiles = orbit_r[sector]
            for node in range(16):
                tile = tiles[column[node]]
                for f, v in enumerate(orbit_base[node]):
                    answer += max(-(v + tile[f]), 0.)
        results.append(answer)
    return results


def grid_proxy(b, r, maps):
    """S-only grid feed per orbit and node: 16*sum_F |B + sum_s R_s(map)|."""
    maps = permutation(maps)
    return [[_ordered_sum(16. * abs(v) for v in node) for node in orbit] for orbit in _base(b, r, maps)]


# ------------------------------------------------- assignment with certificate

def assignment(cost):
    """Minimum-cost square assignment (shortest augmenting paths).

    Returns column per row and the optimum value. The dual potentials are
    checked as an optimality certificate: reduced costs are nonnegative and
    vanish on the returned assignment, up to round-off.
    """
    n = len(cost)
    if n < 1 or any(len(row) != n for row in cost) or not _finite(_flat(cost)):
        raise ValueError('Invalid square assignment costs')
    inf = float('inf')
    u, v, owner, way = [0.] * (n + 1), [0.] * (n + 1), [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n + 1):
        owner[0], j0 = i, 0
        minimum, used = [inf] * (n + 1), [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = owner[j0], inf, 0
            for j in range(1, n + 1):
                if not used[j]:
                    reduced = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if reduced < minimum[j]:
                        minimum[j], way[j] = reduced, j0
                    if minimum[j] < delta:
                        delta, j1 = minimum[j], j
            for j in range(n + 1):
                if used[j]:
                    u[owner[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if owner[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            owner[j0] = owner[j1]
            j0 = j1
    column = [0] * n
    for j in range(1, n + 1):
        column[owner[j] - 1] = j - 1
    value = _ordered_sum(cost[i][column[i]] for i in range(n))
    tolerance = 1e-9 * max(1., max(abs(x) for x in _flat(cost)))
    for i in range(n):
        for j in range(n):
            reduced = cost[i][j] - u[i + 1] - v[j + 1]
            if reduced < -tolerance or (j == column[i] and abs(reduced) > tolerance):
                raise ValueError('Assignment optimality certificate failed')
    return column, value


def assignment_cost(b, r, current, sector, shape):
    gx, gy, _ = geometry(shape)
    hx, hy, k = 4 // gx, 4 // gy, gx * gy
    base = _base(b, r, current, skip=sector)
    cost = [[0.] * k for _ in range(k)]
    for cx in range(hx):
        for cy in range(hy):
            nodes = [4 * ((c // gy) * hx + cx) + (c % gy) * hy + cy for c in range(k)]
            for orbit_base, orbit_r in zip(base, r):
                tiles = orbit_r[sector]
                for i, destination in enumerate(nodes):
                    row = orbit_base[destination]
                    for j, source in enumerate(nodes):
                        tile = tiles[source]
                        for f, v in enumerate(row):
                            cost[i][j] += max(0., -(v + tile[f]))
    return cost


# ---------------------------------------------------------------- fitting

@dataclass
class Fit:
    maps: list
    geometry: tuple
    before: float
    after: float


def optimize(b, r, current, shape):
    """Three assignment passes; accept a sector change only on strict decrease."""
    shape = geometry(shape)
    current = permutation(current)
    extract(current, shape)
    before = value = objective(b, r, current)
    for _ in range(3):
        changed = False
        for sector in range(len(current)):
            column, _ = assignment(assignment_cost(b, r, current, sector, shape))
            trial = [list(row) for row in current]
            trial[sector] = lift([column], shape)[0]
            candidate = objective(b, r, trial)
            if candidate < value:
                current, value, changed = trial, candidate, True
        if not changed:
            break
    return Fit(current, shape, before, value)


def replay_trace(b, r, current, shape, trace):
    """Check a recorded tile-assignment trace by assignment VALUE (ties allowed)."""
    current, shape = permutation(current), geometry(shape)
    extract(current, shape)
    ns = len(current)
    if any(len(row) != 18 for row in trace) or not trace or len(trace) % ns or len(trace) > 3 * ns:
        raise ValueError('Incomplete/invalid three-pass tile trace')
    before = old = objective(b, r, current)
    changed, accepted, largest = False, 0, 0.
    for index, row in enumerate(trace):
        step, sector = row[0], row[1]
        if (step, sector) != (1 + index // ns, index % ns):
            raise ValueError('Tile trace changed visit order')
        if index and sector == 0:
            if not changed:
                raise ValueError('Tile trace continued after an unchanged pass')
            changed = False
        cost = assignment_cost(b, r, current, sector, shape)
        _, optimum = assignment(cost)
        trial = [list(m) for m in current]
        trial[sector] = list(row[2:])
        q = extract([trial[sector]], shape)[0]
        selected = _ordered_sum(cost[i][q[i]] for i in range(len(q)))
        tolerance = _tolerance(optimum, selected)
        if selected > optimum + tolerance:
            raise ValueError('Recorded assignment is not optimal in value')
        largest = max(largest, abs(selected - optimum))
        if trial != current:
            new = objective(b, r, trial)
            if new > old + _tolerance(old, new):
                raise ValueError('Tile map changed without a loss decrease')
            current, old, changed, accepted = trial, new, True, accepted + 1
    if len(trace) < 3 * ns and changed:
        raise ValueError('Tile fit stopped before a complete unchanged pass')
    return dict(passed=True, before=before, after=old, maps=current,
                steps=len(trace), accepted_steps=accepted, maximum_assignment_optimum_error=largest)


def sobol_words(ids):
    ids = [int(i) for i in ids]
    if not ids or min(ids) < 1 or max(ids) > 2**30 or len(set(ids)) != len(ids):
        raise ValueError('Require distinct positive outer FKS IDs within 30 bits')
    words = []
    for sector in ids:
        gray = (sector - 1) ^ ((sector - 1) >> 1)
        word, direction = [0, 0], [2**29, 2**29]
        for j in range(30):
            if gray & (1 << j):
                word = [word[0] ^ direction[0], word[1] ^ direction[1]]
            direction = [direction[0] >> 1, direction[1] ^ (direction[1] >> 1)]
        words.append(word)
    return words


def gauge(ids):
    """Static quarter gauge: Sobol shifts if all are quarter-aligned, else 0."""
    words = sobol_words(ids)
    aligned = all(w % 2**28 == 0 for word in words for w in word)
    q = [[w // 2**28 for w in word] if aligned else [0, 0] for word in words]
    return dict(quarter_aligned=aligned, words=words, q=q, convention=1)


def checked_q(q, sectors=None):
    q = [[int(x) for x in row] for row in q]
    if not q or any(len(row) != 2 or min(row) < 0 or max(row) > 3 for row in q) or (
            sectors is not None and len(q) != sectors):
        raise ValueError('Require exact quarter gauge q[S][2] in 0..3')
    return q


def zero_seed(q):
    codes = [4 * ((-x) % 4) + (-y) % 4 for x, y in checked_q(q)]
    return [phase_map(c) for c in codes], codes


def zero_seed_legal(q, shape):
    gx, gy, _ = geometry(shape)
    return all((gx * x) % 4 == 0 and (gy * y) % 4 == 0 for x, y in checked_q(q))


def phase_fit(b, r, shape, initial_codes):
    shape = geometry(shape)
    legal = legal_phase_codes(shape)
    codes = [int(c) for c in initial_codes]
    if any(c not in legal for c in codes):
        raise ValueError('Initial phases violate deployed coarse geometry')
    maps = [phase_map(c) for c in codes]
    for _ in range(3):
        changed = False
        for s in range(len(maps)):
            values = _sector_objectives(b, r, maps, s, [phase_map(c) for c in legal])
            best = min(values)
            if values[legal.index(codes[s])] > best:
                codes[s] = legal[values.index(best)]
                maps[s] = phase_map(codes[s])
                changed = True
        if not changed:
            break
    extract(maps, shape)
    return maps


@dataclass
class MultiFit:
    maps: list
    before: float
    after: float
    winner: int
    geometry: tuple


def multistart(b, r, current, q, shape):
    """Current, identity-phase and gauge-zero-phase starts; strict TRAIN minimum."""
    shape, current = geometry(shape), permutation(current)
    q = checked_q(q, len(current))
    extract(current, shape)
    if not b:
        raise ValueError('Fit needs observed TRAIN orbits')
    before = objective(b, r, current)
    candidates = [optimize(b, r, current, shape)]
    starts = [[0] * len(current)]
    if zero_seed_legal(q, shape):
        starts.append(zero_seed(q)[1])
    for codes in starts:
        candidates.append(optimize(b, r, phase_fit(b, r, shape, codes), shape))
    winner, best, maps = 0, before, current
    for index, fit in enumerate(candidates, 1):
        if fit.after < best:
            winner, best, maps = index, fit.after, fit.maps
    return MultiFit(maps, before, best, winner, shape)


# ----------------------------------------------------------- epoch replay

@dataclass
class ReplayBank:
    b: list
    r: list
    current_n: int
    previous_n: int
    previous_scale: float
    replayed: bool


def bank(b, r, previous_b=None, previous_r=None):
    """Current bank first, then the previous bank scaled by .5*N/N_previous.

    Canonical flavours are zero-padded to the larger flavour count. Without
    history (or with an empty side) the current bank is returned unchanged.
    """
    b, r = arrays(b, r)
    n = len(b)
    if previous_b is None or previous_r is None:
        if previous_b is not None or previous_r is not None:
            raise ValueError('Incomplete previous bank')
        return ReplayBank(b, r, n, 0, 0., False)
    pb, pr = arrays(previous_b, previous_r)
    pn = len(pb)
    if n and pn and len(r[0]) != len(pr[0]):
        raise ValueError('Previous sector count differs')
    if n == 0 or pn == 0:
        return ReplayBank(b, r, n, pn, 0., False)
    nf = max(len(b[0][0]), len(pb[0][0]))
    scale = COEFFICIENT * n / pn

    def pad(node, factor=1.):
        return [factor * x for x in node] + [0.] * (nf - len(node))
    fit_b = [[pad(node) for node in orbit] for orbit in b]
    fit_b += [[pad(node, scale) for node in orbit] for orbit in pb]
    fit_r = [[[pad(node) for node in sector] for sector in orbit] for orbit in r]
    fit_r += [[[pad(node, scale) for node in sector] for sector in orbit] for orbit in pr]
    if not _finite(_flat(fit_b)) or not _finite(_flat(fit_r)):
        raise ValueError('Nonfinite scaled replay bank')
    return ReplayBank(fit_b, fit_r, n, pn, scale, True)


def bank_objective(value, maps):
    return objective(value.b, value.r, maps) if value.current_n else 0.


def compatible(current, previous):
    if previous is None:
        return
    for key in ('slot', 'ns', 'nlegs', 'alias_gauge', 'ids', 'providers', 'groups', 'pdgs', 'q'):
        if current[key] != previous[key]:
            raise ValueError('Replay static metadata differs: ' + key)


def immediate_previous(records, index, slot):
    """Only the immediately preceding epoch; an empty epoch drops history."""
    if index == 0:
        return None
    if not 0 < index < len(records):
        raise ValueError('Invalid replay epoch index')
    matches = [entry for entry in records[index - 1]['entries'] if entry['slot'] == slot]
    if len(matches) > 1:
        raise ValueError('Duplicate previous slot')
    return matches[0] if matches else None


def current_geometry(current_n, oldshape, epoch):
    """Capacity growth and support use CURRENT-epoch orbit counts only."""
    oldshape = tuple(oldshape)
    if (oldshape not in ((1, 1, 1),) + GEOMETRIES or not isinstance(epoch, int) or epoch < 1 or
            not isinstance(current_n, int) or current_n < 0):
        raise ValueError('Invalid current support/geometry/epoch')
    size = oldshape[0] * oldshape[1]
    shape = oldshape
    if current_n >= 64 and epoch >= 3:
        shape = (4, 4, 1)
    elif current_n >= 32 and epoch >= 2 and size < 8:
        shape = (4, 2, 1)
    elif current_n >= 16 and size < 4:
        shape = (2, 2, 1)
    minimum = {1: 10**30, 4: 16, 8: 32, 16: 64}[shape[0] * shape[1]]
    return shape, current_n >= minimum


def fit_epoch(b, r, previous_b, previous_r, current, q, oldshape, epoch):
    current, oldshape = permutation(current), tuple(oldshape)
    q = checked_q(q, len(current))
    for sectors in (r, previous_r):
        if sectors and len(sectors[0]) != len(current):
            raise ValueError('Current map/sector count differs')
    if oldshape == (1, 1, 1):
        if any(row != list(range(16)) for row in current):
            raise ValueError('Untrained map must be identity')
    else:
        extract(current, oldshape)
    value = bank(b, r, previous_b, previous_r)
    shape, supported = current_geometry(value.current_n, oldshape, epoch)
    before = bank_objective(value, current)
    if not supported:
        return dict(maps=current, geometry=oldshape, fit=None, alternatives=[], bank=value,
                    before=before, after=before)
    chosen = multistart(value.b, value.r, current, q, shape)
    alternatives = [chosen]
    if shape[0] * shape[1] == 8 and oldshape[0] * oldshape[1] < 8:
        other = multistart(value.b, value.r, current, q, (2, 4, 1))
        alternatives.append(other)
        if other.after < chosen.after:
            chosen = other
    return dict(maps=chosen.maps, geometry=chosen.geometry, fit=chosen, alternatives=alternatives,
                bank=value, before=before, after=chosen.after)


def reassemble(entries, shape, map_key='after'):
    """S-only proxy and hits from probe rows, in global sample order."""
    if map_key not in ('before', 'after') or len(shape) != 3 or min(shape) < 1 or shape[0] < 2:
        raise ValueError('Invalid map selector/accumulator shape')
    n, ndim, nchannels = map(int, shape)
    acc, hits, ordered, seen = Array3(shape), Array3((n - 1, ndim, nchannels), fill=0), [], set()
    for entry in entries:
        nt = int(entry['nt'])
        if not nt:
            continue
        cells, channels, ids = entry['cells'], entry['channels'], entry['sample_ids']
        if len(cells) != nt or len(channels) != nt or len(ids) != nt or any(
                len(c) != 16 or any(len(node) != ndim for node in c) for c in cells):
            raise ValueError('Invalid original cell/channel/sample shapes')
        if (any(not 1 <= x < n for c in cells for node in c for x in node) or
                any(not 0 <= c < nchannels for c in channels) or min(ids) < 1):
            raise ValueError('Original cell/channel/sample out of range')
        proxy = grid_proxy(entry['b'], entry['r'], entry[map_key])
        for i, sample in enumerate(ids):
            if sample in seen:
                raise ValueError('Repeated global sample ID across slots')
            seen.add(sample)
            ordered.append((sample, cells[i], channels[i], proxy[i]))
    for _, cells, channel, proxy in sorted(ordered, key=lambda row: row[0]):
        for node in range(16):
            for dim in range(ndim):
                cell = cells[node][dim]
                acc[cell, dim, channel] += proxy[node]
                hits[cell - 1, dim, channel] += 1
    return acc, hits


# ------------------------------------------------------- native10 reader

class _Reader:
    def __init__(self, payload, endian):
        self.payload, self.endian, self.offset = payload, endian, 0

    def take(self, count, kind):
        size = struct.calcsize(kind)
        if count < 0 or count * size > len(self.payload) - self.offset:
            raise ValueError('Truncated or dimension-corrupt epoch archive')
        values = struct.unpack_from('%s%d%s' % (self.endian, count, kind), self.payload, self.offset)
        self.offset += count * size
        return list(values)

    def ints(self, count):
        return self.take(count, 'i')

    def doubles(self, count):
        return self.take(count, 'd')


def _grid(value):
    n, d, c = value.shape
    for k in range(c):
        for j in range(d):
            column = [value[i, j, k] for i in range(n)]
            if (not _finite(column) or column[0] != 0. or column[-1] != 1. or
                    any(b <= a for a, b in zip(column, column[1:]))):
                raise ValueError('Invalid adaptive grid')


def _maps(value, shape):
    permutation(value)
    if tuple(shape) not in SHAPES:
        raise ValueError('Invalid map capacity')
    if tuple(shape) == (1, 1):
        if any(row != list(range(16)) for row in value):
            raise ValueError('Untrained capacity must be identity')
    else:
        extract(value, (shape[0], shape[1], 1))


def _metadata(entry):
    if (len(set(entry['ids'])) != entry['ns'] or min(entry['ids']) < 1 or
            min(entry['providers']) < 1 or min(entry['groups']) < 1):
        raise ValueError('Invalid static sector metadata')
    keys = [(tuple(pdgs), provider) for pdgs, provider in zip(entry['pdgs'], entry['providers'])]
    expected = {key: i + 1 for i, key in enumerate(sorted(set(keys)))}
    if any(group != expected[key] for group, key in zip(entry['groups'], keys)):
        raise ValueError('Canonical groups differ from provider/ordered-PDG keys')


def _matrix(values, rows, columns):
    """Fortran (columns-fastest-in-memory) block of shape (columns,rows) -> rows of columns."""
    return [values[k * columns:(k + 1) * columns] for k in range(rows)]


def _attempt(reader, ns):
    out = dict(zip(('kind', 'status', 'reuse', 'nphase', 'ntile'), reader.ints(5)))
    if (out['status'] not in (0, 1, 2, 3) or out['kind'] not in (0, 1, 2, 3) or
            min(out['reuse'], out['nphase'], out['ntile']) < 0 or max(out['nphase'], out['ntile']) > 3 * ns):
        raise ValueError('Invalid fit candidate header')
    out['losses'] = reader.doubles(3)
    if not _finite(out['losses']) or min(out['losses']) < 0:
        raise ValueError('Invalid fit candidate losses')
    if out['status'] == 0:
        if out['reuse'] or out['nphase'] or out['ntile'] or any(out['losses']):
            raise ValueError('Not-run candidate lacks zero sentinels')
        return out
    for key in ('initial_map', 'phase_map', 'final_map'):
        out[key] = permutation([[x - 1 for x in row] for row in _matrix(reader.ints(16 * ns), ns, 16)])
    for key, n in (('phase_trace', out['nphase']), ('tile_trace', out['ntile'])):
        out[key] = [[row[0]] + [x - 1 for x in row[1:]] for row in _matrix(reader.ints(18 * n), n, 18)]
    return out


def _fit_extension(reader, ns):
    names = ('fit_status', 'alias_gauge', 'orientation', 'primarygx', 'primarygy', 'primary_selected',
             'altgx', 'altgy', 'alt_selected')
    out = dict(zip(names, reader.ints(9)))
    if out['fit_status'] not in (0, 1, 2) or out['alias_gauge'] not in (0, 1) or out['orientation'] not in (0, 1, 2):
        raise ValueError('Invalid fit status/gauge/orientation')
    out['q'] = checked_q(_matrix(reader.ints(2 * ns), ns, 2), ns)
    out['primary'] = [_attempt(reader, ns) for _ in range(3)]
    out['alternative'] = [_attempt(reader, ns) for _ in range(3)]
    return out


def read_archive(path):
    payload = Path(path).read_bytes()
    if payload[:4] == struct.pack('<i', MAGIC):
        endian = '<'
    elif payload[:4] == struct.pack('>i', MAGIC):
        endian = '>'
    else:
        raise ValueError('Invalid epoch archive magic/native int32 representation')
    reader, records, footer = _Reader(payload, endian), [], False
    names = ('version', 'mode', 'epoch', 'npre', 'npost', 'ndim', 'nchans', 'nentries',
             'ordinary', 'probes', 'reused', 'rejections', 'restores')
    counters = ('ordinary', 'probes', 'reused', 'rejections', 'restores')
    while reader.offset < len(payload):
        magic = reader.ints(1)[0]
        if magic == -MAGIC:
            footer = True
            break
        if magic != MAGIC:
            raise ValueError('Corrupt epoch record marker')
        row = dict(zip(names, reader.ints(13)))
        if (row['version'] != 10 or row['mode'] != 29 or row['epoch'] < 1 or
                not 4 <= row['npre'] <= 10000 or not 4 <= row['npost'] <= 10000 or
                row['npre'] % 4 or row['npost'] % 4 or not 3 <= row['ndim'] <= 100 or
                not 1 <= row['nchans'] <= 10000 or not 1 <= row['nentries'] <= 10000 or
                min(row[k] for k in counters) < 0 or row['rejections'] > row['probes'] or
                row['reused'] != 16 * (row['probes'] - row['rejections']) or row['restores'] != row['probes']):
            raise ValueError('Unsupported epoch header/counts')
        row['cpu'] = reader.doubles(4)
        if not _finite(row['cpu']) or min(row['cpu']) < 0.:
            raise ValueError('Invalid cumulative CPU counters')
        d, c, n = row['ndim'], row['nchans'], row['npre']
        row['grid_before'] = Array3((n + 1, d, c), reader.doubles((n + 1) * d * c))
        row['grid_after'] = Array3((row['npost'] + 1, d, c), reader.doubles((row['npost'] + 1) * d * c))
        _grid(row['grid_before'])
        _grid(row['grid_after'])
        row['acc'] = {k: Array3((n + 1, d, c), reader.doubles((n + 1) * d * c))
                      for k in ('base', 'total', 'ordinary', 'proxy')}
        row['hits'] = {k: Array3((n, d, c), reader.ints(n * d * c)) for k in ('base', 'total', 'ordinary', 'proxy')}
        if (any(not _finite(a.data) or min(a.data) < 0. for a in row['acc'].values()) or
                any(min(a.data) < 0 for a in row['hits'].values())):
            raise ValueError('Nonfinite/negative grid accumulation')
        entries = []
        for _ in range(row['nentries']):
            entry = dict(zip(('slot', 'nf', 'ns', 'nt', 'nlegs', 'oldgx', 'oldgy', 'gx', 'gy', 'total_samples',
                              'nsteps'), reader.ints(11)))
            nf, ns, nt = entry['nf'], entry['ns'], entry['nt']
            if (entry['slot'] < 1 or not 1 <= ns <= 10000 or not 3 <= entry['nlegs'] <= 10000 or
                    min(nf, nt, entry['total_samples']) < 0 or nf > 100000 or
                    (nt > 0 and nf < 1) or nt > row['probes']):
                raise ValueError('Invalid slot dimensions/support')
            if entry['nsteps'] != 0:
                raise ValueError('Slot must have no legacy base trace')
            entry['ids'], entry['providers'], entry['groups'] = (reader.ints(ns) for _ in range(3))
            pdgs = reader.ints(ns * entry['nlegs'])
            entry['pdgs'] = [[pdgs[s + ns * leg] for leg in range(entry['nlegs'])] for s in range(ns)]
            entry['before'] = [[x - 1 for x in row16] for row16 in _matrix(reader.ints(16 * ns), ns, 16)]
            entry['after'] = [[x - 1 for x in row16] for row16 in _matrix(reader.ints(16 * ns), ns, 16)]
            _metadata(entry)
            _maps(entry['before'], (entry['oldgx'], entry['oldgy']))
            _maps(entry['after'], (entry['gx'], entry['gy']))
            entry['diagnostics'] = reader.doubles(4)
            if not _finite(entry['diagnostics']) or min(entry['diagnostics']) < 0.:
                raise ValueError('Nonfinite/negative fit diagnostics')
            entry.update(_fit_extension(reader, ns))
            if nt:
                fb, fr = reader.doubles(nf * 16 * nt), reader.doubles(nf * 16 * ns * nt)
                entry['b'] = [[[fb[f + nf * (node + 16 * o)] for f in range(nf)] for node in range(16)]
                              for o in range(nt)]
                entry['r'] = [[[[fr[f + nf * (node + 16 * (s + ns * o))] for f in range(nf)] for node in range(16)]
                               for s in range(ns)] for o in range(nt)]
                arrays(entry['b'], entry['r'])
                cells = reader.ints(d * 16 * nt)
                entry['cells'] = [[[cells[x + d * (node + 16 * o)] for x in range(d)] for node in range(16)]
                                  for o in range(nt)]
                entry['channels'] = [x - 1 for x in reader.ints(nt)]
                entry['sample_ids'] = reader.ints(nt)
                ids = entry['sample_ids']
                if (any(not 1 <= x <= n for cell in entry['cells'] for node in cell for x in node) or
                        any(not 0 <= x < c for x in entry['channels']) or
                        any(b <= a for a, b in zip(ids, ids[1:])) or min(ids) < 64 or
                        any(x % 16 for x in ids) or max(ids) > row['ordinary']):
                    raise ValueError('Invalid original probe cells/channels/sample IDs')
            else:
                entry['b'], entry['r'], entry['cells'], entry['channels'], entry['sample_ids'] = [], [], [], [], []
            entries.append(entry)
        if len({e['slot'] for e in entries}) != len(entries):
            raise ValueError('Duplicate static slot')
        row['entries'] = entries
        records.append(row)
    if not footer or reader.offset != len(payload) or not records:
        raise ValueError('Missing end marker, trailing bytes or empty archive')
    return dict(version=10, mode=records[0]['mode'], byte_order='little' if endian == '<' else 'big',
                records=records, path=str(Path(path).resolve()), sha256=hashlib.sha256(payload).hexdigest())


def read_table(path):
    """Read the frozen SCL10 map table and grid stored in mint_grids."""
    tokens = [line.split() for line in Path(path).read_text().splitlines()]
    positions = [i for i, line in enumerate(tokens) if line and line[0].startswith('SCL')]
    if not positions or any(tokens[i][0] != 'SCL10' for i in positions):
        raise ValueError('Missing or mixed saved SCL protocol')
    if any(line and line[0] != 'SCL10' for line in tokens[positions[0]:]):
        raise ValueError('Trailing nonblank record within/after SCL10 table')
    lines = [line for line in tokens[positions[0]:] if line]
    if lines[0][:2] != ['SCL10', 'H'] or len(lines[0]) != 20:
        raise ValueError('Missing/malformed SCL10 header')
    names = ('version', 'mode', 'nentries', 'nint', 'ndim', 'nchans', 'epoch', 'ordinary', 'probes', 'reused',
             'rejections', 'restores', 'warmup', 'stride', 'gauge_version', 'phase_passes', 'tile_passes',
             'selection_version')
    result = dict(zip(names, map(int, lines[0][2:])))
    if (result['version'] != 10 or result['mode'] != 29 or
            min(result[k] for k in ('nentries', 'nint', 'ndim', 'nchans', 'epoch')) < 1 or
            result['ndim'] < 3 or result['nint'] % 4 or
            min(result[k] for k in ('ordinary', 'probes', 'reused', 'rejections', 'restores')) < 0 or
            result['rejections'] > result['probes'] or
            result['reused'] != 16 * (result['probes'] - result['rejections']) or
            result['restores'] != result['probes'] or (result['warmup'], result['stride']) != (64, 16)):
        raise ValueError('Unsupported SCL10 header/counts')
    if tuple(result[k] for k in ('gauge_version', 'phase_passes', 'tile_passes', 'selection_version')) != (1, 3, 3, 1):
        raise ValueError('Wrong gauge/solver/selection versions')
    if len(lines) < 2 or lines[1][:2] != ['SCL10', 'C'] or len(lines[1]) != 7:
        raise ValueError('Missing/malformed SCL10 CPU row')
    cpu = [float(x) for x in lines[1][2:]]
    if not _finite(cpu) or not 0. < cpu[0] <= 1. or min(cpu) < 0.:
        raise ValueError('Invalid SCL10 CPU budget/counters')
    result['budget'], result['cpu'] = cpu[0], cpu[1:]
    entries, grids, current = {}, {}, None
    for row in lines[2:]:
        if row[:2] == ['SCL10', 'L']:
            if len(row) != 10:
                raise ValueError('Malformed SCL10 slot row')
            slot, ns, nlegs, nf, gx, gy, total, alias_gauge = map(int, row[2:])
            if slot in entries or min(slot, ns, nlegs) < 1 or nlegs < 3 or min(nf, total) < 0 or (gx, gy) not in SHAPES:
                raise ValueError('Invalid SCL10 slot metadata')
            if alias_gauge not in (0, 1):
                raise ValueError('Invalid saved gauge flag')
            current = dict(alias_gauge=alias_gauge, slot=slot, ns=ns, nlegs=nlegs, nf=nf, gx=gx, gy=gy,
                           total_samples=total, rows=[], maps=[])
            entries[slot] = current
        elif row[:2] == ['SCL10', 'R']:
            if current is None or len(row) != 9 + current['nlegs']:
                raise ValueError('Malformed SCL10 sector row')
            slot, index, sector, provider, group = map(int, row[2:7])
            if (slot != current['slot'] or index != len(current['rows']) + 1 or min(sector, provider, group) < 1 or
                    len(current['maps']) != len(current['rows'])):
                raise ValueError('Invalid SCL10 static row order')
            current['rows'].append(dict(id=sector, provider=provider, group=group,
                                        pdgs=[int(x) for x in row[7:-2]], q=[int(x) for x in row[-2:]]))
        elif row[:2] == ['SCL10', 'P']:
            if current is None or len(row) != 20:
                raise ValueError('Malformed SCL10 permutation row')
            slot, index = map(int, row[2:4])
            if (slot != current['slot'] or index != len(current['maps']) + 1 or
                    len(current['rows']) != len(current['maps']) + 1):
                raise ValueError('Invalid SCL10 permutation order')
            current['maps'].append([int(x) - 1 for x in row[4:]])
        elif row[:2] == ['SCL10', 'G']:
            channel, dimension, nint = map(int, row[2:5])
            key = (channel, dimension)
            if (key in grids or not 1 <= channel <= result['nchans'] or not 1 <= dimension <= result['ndim'] or
                    nint != result['nint'] or len(row) != nint + 6):
                raise ValueError('Malformed/duplicate SCL10 grid row')
            grids[key] = [float(x) for x in row[5:]]
        else:
            raise ValueError('Unknown SCL10 record')
    if len(entries) != result['nentries'] or len(grids) != result['ndim'] * result['nchans']:
        raise ValueError('Incomplete SCL10 table/grid')
    for entry in entries.values():
        if len(entry['rows']) != entry['ns'] or len(entry['maps']) != entry['ns']:
            raise ValueError('Incomplete SCL10 layout/maps')
        _maps(entry['maps'], (entry['gx'], entry['gy']))
        meta = dict(entry, ids=[r['id'] for r in entry['rows']], providers=[r['provider'] for r in entry['rows']],
                    groups=[r['group'] for r in entry['rows']], pdgs=[r['pdgs'] for r in entry['rows']])
        _metadata(meta)
        expected = gauge(meta['ids'])
        q = [r['q'] for r in entry['rows']]
        if entry['alias_gauge'] != int(expected['quarter_aligned']) or q != expected['q']:
            raise ValueError('Saved static gauge disagrees with exact outer-ID words')
        entry['q'] = q
    grid = Array3((result['nint'] + 1, result['ndim'], result['nchans']))
    for (c, d), values in grids.items():
        for i, value in enumerate(values):
            grid[i, d - 1, c - 1] = value
    _grid(grid)
    native = [[float(x) for x in row[1:]] for row in tokens if row and row[0] == 'AVE' and len(row) == result['ndim'] + 1]
    if len(native) != (result['nint'] + 1) * result['nchans'] or any(
            native[c * (result['nint'] + 1) + i][d] != grid[i, d, c]
            for c in range(result['nchans']) for i in range(result['nint'] + 1) for d in range(result['ndim'])):
        raise ValueError('SCL10 grid differs from actual native MINT grid')
    result.update(entries=entries, grid=grid, path=str(Path(path).resolve()),
                  sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
    return result


# ---------------------------------------------------------------- audit

def _phase_replay(b, r, start, shape, trace, final):
    maps = permutation(start)
    extract(maps, shape)
    legal, ns = legal_phase_codes(shape), len(maps)
    if any(len(row) != 18 for row in trace) or not trace or len(trace) % ns or len(trace) > 3 * ns:
        raise ValueError('Incomplete phase trace')
    for row in maps:
        if row[0] not in legal or row != phase_map(row[0]):
            raise ValueError('Phase start is not a legal cyclic translation')
    changed, accepted, largest = False, 0, 0.
    for i, row in enumerate(trace):
        step, s = row[0], row[1]
        if (step, s) != (1 + i // ns, i % ns):
            raise ValueError('Phase trace visit order differs')
        if i and s == 0:
            if not changed:
                raise ValueError('Phase trace continued after unchanged pass')
            changed = False
        values = _sector_objectives(b, r, maps, s, [phase_map(c) for c in legal])
        current, minimum = maps[s][0], min(values)
        selected = list(row[2:])
        code = selected[0]
        if code not in legal or selected != phase_map(code):
            raise ValueError('Recorded phase is not a legal coarse translation')
        old, chosen = values[legal.index(current)], values[legal.index(code)]
        tolerance = _tolerance(minimum, chosen, old)
        largest = max(largest, chosen - minimum)
        if chosen - minimum > tolerance:
            raise ValueError('Phase choice is materially suboptimal')
        if code != current:
            if chosen > old + tolerance:
                raise ValueError('Phase changed without a loss decrease')
            maps[s], changed, accepted = selected, True, accepted + 1
        elif old - minimum > tolerance:
            raise ValueError('Phase retained a materially improvable code')
    if len(trace) < 3 * ns and changed:
        raise ValueError('Phase trace stopped before unchanged pass')
    if maps != final:
        raise ValueError('Phase trace does not close')
    return dict(passed=True, steps=len(trace), accepted_steps=accepted, maximum_optimum_gap=largest)


def _audit_attempts(e, shape, attempts, selected):
    b, r, ns = e['b'], e['r'], e['ns']
    before = objective(b, r, e['before'])
    identity = [list(range(16)) for _ in range(ns)]
    initial_maps = [e['before'], identity, zero_seed(e['q'])[0]]
    zero_legal = zero_seed_legal(e['q'], shape)
    reports, prior = [], []
    for i, a in enumerate(attempts, 1):
        if i == 3 and not zero_legal:
            if a['status'] != 0:
                raise ValueError('Coarse-illegal exact-zero candidate was run')
            reports.append(dict(kind=i, passed=True, status=0, reason='coarse_illegal_zero_seed'))
            prior.append(a)
            continue
        if a['status'] == 0 or a['kind'] != i:
            raise ValueError('Required fit candidate missing/wrong kind')
        if a['initial_map'] != initial_maps[i - 1]:
            raise ValueError('Fit candidate initial map differs')
        for key in ('initial_map', 'phase_map', 'final_map'):
            extract(a[key], shape)
        measured = [objective(b, r, a[key]) for key in ('initial_map', 'phase_map', 'final_map')]
        _close(a['losses'], measured, 'Fit candidate recorded losses differ')
        if a['losses'][1] > a['losses'][0] + ATOL or a['losses'][2] > a['losses'][1] + ATOL:
            raise ValueError('Fit candidate increased TRAIN loss')
        if a['status'] == 3:
            if i != 3 or a['reuse'] != 2 or a['nphase'] or a['ntile']:
                raise ValueError('Malformed whole phase-start reuse')
            source = prior[1]
            if any(a[key] != source[key] for key in ('initial_map', 'phase_map', 'final_map', 'losses')):
                raise ValueError('Phase-start reuse does not equal referenced fit')
            report = dict(kind=i, passed=True, status=3, reuse=2)
        else:
            if i == 1:
                if a['nphase'] or a['status'] != 1 or a['reuse']:
                    raise ValueError('Current-map candidate has phase/reuse metadata')
                if a['phase_map'] != a['initial_map']:
                    raise ValueError('Current-map candidate unexpectedly ran phase fit')
                phase_report = dict(passed=True, steps=0)
            else:
                if i == 3 and a['initial_map'] == prior[1]['initial_map']:
                    raise ValueError('Duplicate phase-stage start was recomputed')
                phase_report = _phase_replay(b, r, a['initial_map'], shape, a['phase_trace'], a['phase_map'])
            matches = [j + 1 for j, p in enumerate(prior) if p['status'] and p['phase_map'] == a['phase_map']]
            if matches:
                if a['status'] != 2 or a['reuse'] != min(matches) or a['ntile']:
                    raise ValueError('Identical assignment input not reused with earliest reference')
                source = prior[a['reuse'] - 1]
                if a['final_map'] != source['final_map'] or a['losses'][2] != source['losses'][2]:
                    raise ValueError('Assignment reuse differs from its reference')
                tile_report = dict(passed=True, steps=0, reuse=a['reuse'])
            else:
                if a['status'] != 1 or a['reuse']:
                    raise ValueError('Candidate claims nonexistent assignment reuse')
                tile_report = replay_trace(b, r, a['phase_map'], shape, a['tile_trace'])
                if tile_report.pop('maps') != a['final_map']:
                    raise ValueError('Tile trace does not close')
            report = dict(kind=i, passed=True, status=a['status'], reuse=a['reuse'], phase=phase_report, tile=tile_report)
        prior.append(a)
        reports.append(report)
    # Replay the producer's selection with its own archived scalars: strict
    # minimum, earliest wins ties. The unchanged map's loss is the current-map
    # candidate's start loss, already checked against `before` above, so the
    # decision does not depend on this module's summation order.
    winner, best, chosen = 0, attempts[0]['losses'][0], e['before']
    for i, a in enumerate(attempts, 1):
        if a['status'] and a['losses'][2] < best:
            winner, best, chosen = i, a['losses'][2], a['final_map']
    if selected != winner:
        raise ValueError('Strict TRAIN-min candidate/tie priority differs')
    return dict(passed=True, selected=winner, loss_after=best, maps=chosen, attempts=reports)


def _audit_fit(e, epoch):
    expected = gauge(e['ids'])
    if e['alias_gauge'] != int(expected['quarter_aligned']) or e['q'] != expected['q']:
        raise ValueError('Native static gauge differs from exact outer IDs')
    k = e['oldgx'] * e['oldgy']
    (gx, gy, _), supported = current_geometry(e['nt'], (e['oldgx'], e['oldgy'], 1), epoch)
    if not supported:
        status = 0 if e['nt'] == 0 else 1
        if (e['fit_status'] != status or
                any(e[n] for n in ('orientation', 'primarygx', 'primarygy', 'primary_selected', 'altgx', 'altgy',
                                   'alt_selected')) or
                any(a['status'] for a in e['primary'] + e['alternative'])):
            raise ValueError('No-fit slot has computed candidate metadata')
        if e['after'] != e['before'] or (e['gx'], e['gy']) != (e['oldgx'], e['oldgy']):
            raise ValueError('Unsupported slot changed its map or capacity')
        return dict(passed=True, fit_status=status, unchanged_fallback=True, geometry=[e['gx'], e['gy'], 1])
    if e['fit_status'] != 2 or (e['primarygx'], e['primarygy']) != (gx, gy):
        raise ValueError('Primary geometry/support policy differs')
    primary = _audit_attempts(e, (gx, gy, 1), e['primary'], e['primary_selected'])
    chosen, orientation, other = primary, 1, None
    if gx * gy == 8 and k < 8:
        if (e['altgx'], e['altgy']) != (2, 4):
            raise ValueError('Missing alternate 8-tile geometry')
        other = _audit_attempts(e, (2, 4, 1), e['alternative'], e['alt_selected'])
        _close(e['diagnostics'][2:], [primary['loss_after'], other['loss_after']], 'Orientation loss diagnostics differ')
        if other['loss_after'] < primary['loss_after']:
            chosen, gx, gy, orientation = other, 2, 4, 2
    elif (e['altgx'] or e['altgy'] or e['alt_selected'] or any(a['status'] for a in e['alternative']) or
          any(e['diagnostics'][2:])):
        raise ValueError('Unexpected alternative geometry fit')
    if e['orientation'] != orientation or (e['gx'], e['gy']) != (gx, gy):
        raise ValueError('Strict orientation selection/tie policy differs')
    if e['after'] != chosen['maps']:
        raise ValueError('Selected fit does not close to the deployed map')
    strip = lambda report: {key: value for key, value in report.items() if key != 'maps'}
    return dict(passed=True, fit_status=2, geometry=[gx, gy, 1], orientation=orientation, primary=strip(primary),
                alternative=strip(other) if other else None)


def audit_archive(path, table_path, *, expected_budget=None):
    """Replay every epoch of sector_learning_online.bin against mint_grids."""
    archive, table = read_archive(path), read_table(table_path)
    if expected_budget is not None and table['budget'] != expected_budget:
        raise ValueError('SCL10 budget differs from the runtime card')
    previous, slots, reports = None, {}, []
    totals = dict(ordinary=0, probes=0, reused=0, rejections=0, restores=0)
    for index, record in enumerate(archive['records']):
        if record['epoch'] != len(reports) + 1:
            raise ValueError('Epoch sequence differs')
        if previous is not None:
            if record['grid_before'].data != previous['grid_after'].data:
                raise ValueError('Grid changed outside an iteration boundary')
            if any(a < b for a, b in zip(record['cpu'], previous['cpu'])):
                raise ValueError('Cumulative CPU counters decreased')
        delta = {key: record[key] - totals[key] for key in totals}
        if min(delta.values()) < 0:
            raise ValueError('Cumulative point counters decreased')
        if sum(e['nt'] for e in record['entries']) != delta['probes'] - delta['rejections']:
            raise ValueError('Orbit coverage differs from probe/rejection counts')
        acc, hits = record['acc'], record['hits']
        _close(acc['total'], [x + y + z for x, y, z in zip(acc['base'].data, acc['ordinary'].data, acc['proxy'].data)],
               'Float grid accounting differs')
        if hits['total'].data != [x + y + z for x, y, z in zip(hits['base'].data, hits['ordinary'].data, hits['proxy'].data)]:
            raise ValueError('Hit accounting differs')
        n, d, c = hits['ordinary'].shape
        # Each ordinary point is recorded once per dimension, in its own channel.
        for j in range(d):
            if sum(hits['ordinary'][i, j, k] for i in range(n) for k in range(c)) != delta['ordinary']:
                raise ValueError('Ordinary point count differs from grid hits')
            if sum(hits['proxy'][i, j, k] for i in range(n) for k in range(c)) != delta['reused']:
                raise ValueError('Reused-node total differs from grid hits')
        if previous is not None and {e['slot'] for e in record['entries']} != set(slots):
            raise ValueError('Registered slot coverage changed or omitted a slot')
        entry_reports = []
        for e in record['entries']:
            prior = slots.get(e['slot'])
            if prior is not None:
                for key in ('ids', 'providers', 'groups', 'pdgs', 'q', 'alias_gauge'):
                    if e[key] != prior[key]:
                        raise ValueError('Static layout changed across epochs')
                if e['before'] != prior['after'] or (e['oldgx'], e['oldgy']) != (prior['gx'], prior['gy']):
                    raise ValueError('Deployed map changed inside an epoch boundary')
            elif e['before'] != [list(range(16))] * e['ns'] or (e['oldgx'], e['oldgy']) != (1, 1):
                raise ValueError('New slot must start at identity')
            if e['total_samples'] != (prior['total_samples'] if prior else 0) + e['nt']:
                raise ValueError('Per-slot sample accounting differs')
            if e['nt'] and min(e['sample_ids']) <= totals['ordinary']:
                raise ValueError('Fit reuses observations from a previous epoch')
            history = immediate_previous(archive['records'], index, e['slot'])
            compatible(e, history)
            fit_bank = bank(e['b'], e['r'], history['b'] if history else None, history['r'] if history else None)
            before, after = bank_objective(fit_bank, e['before']), bank_objective(fit_bank, e['after'])
            _close(e['diagnostics'][:2], [before, after], 'Reported mixed current/prior TRAIN loss differs')
            # The fit includes history; support and proxy use CURRENT probes only.
            replay = _audit_fit(dict(e, b=fit_bank.b, r=fit_bank.r), record['epoch'])
            stride = record['npre'] // 4
            for cells in e['cells']:
                first = cells[0]
                for node, cell in enumerate(cells):
                    if cell[:-3] != first[:-3] or cell[-1] != first[-1]:
                        raise ValueError('Probe changed Born/phi cell within the 441 orbit')
                    if (cell[-3] != first[-3] + (node // 4) * stride or cell[-2] != first[-2] + (node % 4) * stride):
                        raise ValueError('Raw441 node order/cell strata differ')
                if not 1 <= first[-3] <= stride or not 1 <= first[-2] <= stride:
                    raise ValueError('Raw441 base cell outside its stratum')
            entry_reports.append(dict(
                slot=e['slot'], training_orbits=e['nt'], fit_current_orbits=fit_bank.current_n,
                fit_previous_orbits=fit_bank.previous_n, fit_previous_scale=fit_bank.previous_scale,
                fit_history_replayed=fit_bank.replayed, fit_orbits=len(fit_bank.b),
                old_geometry=[e['oldgx'], e['oldgy'], 1], geometry=[e['gx'], e['gy'], 1],
                loss_before=before, loss_after=after, trace_audit=replay, alias_gauge=e['alias_gauge'],
                quarter_codes=e['q']))
            slots[e['slot']] = e
        expected_acc, expected_hits = reassemble(record['entries'], acc['proxy'].shape, 'after')
        _close(expected_acc, acc['proxy'], 'Reused S-only grid proxy differs')
        if expected_hits.data != hits['proxy'].data:
            raise ValueError('Proxy grid hits differ')
        reports.append(dict(epoch=record['epoch'], ordinary_calls=delta['ordinary'], probes=delta['probes'],
                            reused_nodes=delta['reused'], probe_rejections=delta['rejections'], slots=entry_reports))
        totals, previous = {key: record[key] for key in totals}, record
    final = archive['records'][-1]
    for key in ('epoch', 'ordinary', 'probes', 'reused', 'rejections', 'restores', 'nentries'):
        if table[key] != final[key]:
            raise ValueError('Final SCL10 counters differ from native epochs')
    if table['grid'].data != final['grid_after'].data or set(table['entries']) != set(slots):
        raise ValueError('Final grid/layout differs')
    _close(table['cpu'][:3], final['cpu'][:3], 'Final CPU counters differ')
    if table['cpu'][3] < final['cpu'][3]:
        raise ValueError('Final archive CPU counter decreased')
    for slot, e in slots.items():
        saved = table['entries'][slot]
        if saved['alias_gauge'] != e['alias_gauge'] or saved['q'] != e['q']:
            raise ValueError('Final SCL10 static gauge differs')
        if ((saved['gx'], saved['gy'], saved['total_samples'], saved['nf']) != (e['gx'], e['gy'], e['total_samples'], e['nf'])
                or saved['maps'] != e['after']):
            raise ValueError('Final SCL10 map/capacity differs')
        for i, metadata in enumerate(saved['rows']):
            if metadata != dict(id=e['ids'][i], provider=e['providers'][i], group=e['groups'][i],
                                pdgs=e['pdgs'][i], q=e['q'][i]):
                raise ValueError('Final static metadata differs')
    ratio = sum(table['cpu'][1:]) / table['cpu'][0] if table['cpu'][0] > 0 else None
    return dict(passed=True, native_version=10, mode=archive['mode'], replay_coefficient=COEFFICIENT,
                archive_sha256=archive['sha256'], mint_grids_sha256=table['sha256'], epoch_count=len(reports),
                epochs=reports, counters=totals, cpu_cumulative=table['cpu'],
                extra_over_ordinary_integrand_cpu=ratio, budget_target=table['budget'])
