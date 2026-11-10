"""Cut-qualified deterministic physical inputs, frozen only at admitted prepare.

Uses the existing massive RAMBO map without claiming integration weights or a
uniform massive phase-space density. A local RNG and rejection on input cuts
make prefixes reproducible without touching the active campaign fixtures.
"""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[4]
HELPER = ROOT / 'tools/tirust_benchmark_points.py'
HELPER_SHA256 = '9f95a9a3475a18597ca182f7d42d7e021933bc01458fcbf3484ad63b5655e46c'


def helper():
    source = HELPER.read_bytes()
    if hashlib.sha256(source).hexdigest() != HELPER_SHA256:
        raise ValueError('Pinned massive RAMBO helper changed')
    spec = importlib.util.spec_from_file_location('appendix_c_pinned_rambo', HELPER)
    module = importlib.util.module_from_spec(spec)
    # Execute exactly the bytes just checked, never an ambient cached pyc or
    # a second read after the integrity check.
    exec(compile(source, str(HELPER), 'exec'), module.__dict__)
    return module


def passes_cuts(momenta, pt_min=50.0, delta_r_min=.5):
    if any(not math.isfinite(x) or x < 0 for x in (pt_min, delta_r_min)):
        raise ValueError('Cut values must be finite and nonnegative')
    outgoing = momenta[2:]
    if len(outgoing) < 2:
        raise ValueError('Pilot requires at least two final-state particles')

    def eta(vector):
        transverse = math.hypot(vector[1], vector[2])
        return math.asinh(vector[3] / transverse) if transverse else math.copysign(math.inf, vector[3])

    for i, left in enumerate(outgoing):
        if len(left) != 4 or any(not math.isfinite(x) for x in left):
            return False
        if not math.hypot(left[1], left[2]) > pt_min:
            return False
        for right in outgoing[i + 1:]:
            if len(right) != 4 or any(not math.isfinite(x) for x in right):
                return False
            if math.hypot(right[1], right[2]) == 0:
                return False
            dphi = abs(math.remainder(math.atan2(left[2], left[1]) - math.atan2(right[2], right[1]), 2 * math.pi))
            if not math.hypot(eta(left) - eta(right), dphi) > delta_r_min:
                return False
    return True


def generate_cut_points(*, masses, incoming_masses, seed, count=100,
                        energy=1000.0, max_draws=100000, pt_min=50.0, delta_r_min=.5):
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError('Positive integer count required')
    if isinstance(max_draws, bool) or not isinstance(max_draws, int) or max_draws < count:
        raise ValueError('Draw cap must be an integer at least as large as count')
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError('Explicit integer seed required')
    h = helper()
    masses, incoming_masses = h._masses(masses), h._incoming_masses(incoming_masses)
    if len(masses) < 2 or not math.isfinite(energy) or energy <= max(sum(masses), sum(incoming_masses)):
        raise ValueError('Physical 2-to-n energy and at least two outgoing legs required')
    rng, points = random.Random(seed), []
    for draw in range(1, max_draws + 1):
        final = h._rambo(masses, energy, rng)
        point = h._point('stability-%06d' % len(points), 'ordinary-cut-qualified', masses,
                         energy, final, incoming_masses)
        if not passes_cuts(point['momenta'], pt_min, delta_r_min):
            continue
        point['draw'] = draw
        points.append(point)
        if len(points) == count:
            h.validate_points(points, masses, incoming_masses=incoming_masses)
            return {'points': points, 'draw_count': draw, 'rejected_draws': draw - count}
    raise ValueError('Cut-qualified point draw cap exhausted; no partial sample admitted')


def freeze_cut_points(prefix, *, masses, incoming_masses, card_sha256, seed,
                      count=100, energy=1000.0):
    """Create two new files. Existing files are never reused or overwritten."""
    if not isinstance(card_sha256, str) or len(card_sha256) != 64 or any(c not in '0123456789abcdef' for c in card_sha256):
        raise ValueError('Exact resolved physical card SHA256 required')
    prefix = Path(prefix)
    data_path, metadata_path = prefix.with_suffix('.dat'), prefix.with_suffix('.json')
    if data_path.exists() or metadata_path.exists():
        raise FileExistsError('Fresh point namespace required')
    sample = generate_cut_points(masses=masses, incoming_masses=incoming_masses, seed=seed,
                                 count=count, energy=energy)
    text = str(count) + '\n' + ''.join(' '.join('%.17e' % x for x in vector) + '\n'
                                        for p in sample['points'] for vector in p['momenta'])
    record = {'schema': 'madloop-appendix-c-cut-points-v1',
              'inputs': {'masses': list(masses), 'incoming_masses': list(incoming_masses),
                         'seed': seed, 'count': count, 'energy': energy,
                         'pt_min_GeV': 50.0, 'delta_R_min': .5,
                         'strict_cut_boundary': True, 'resolved_physical_card_sha256': card_sha256},
              'generator': {'path': 'tools/tirust_benchmark_points.py', 'sha256': HELPER_SHA256,
                            'sampling': 'massless RAMBO map, common massive rescaling, input-cut rejection; no massive-density claim'},
              'sha256': hashlib.sha256(text.encode()).hexdigest(), **sample}
    prefix.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects a racing invocation. A partial failed pair
    # remains visible and must be preserved in the attempt evidence.
    with data_path.open('x') as stream: stream.write(text)
    with metadata_path.open('x') as stream: stream.write(json.dumps(record, indent=2) + '\n')
    return record
