"""Reproducible physical momenta for the MadLoop reduction benchmark.

Four-vectors are stored as ``[energy, px, py, pz]`` with positive incoming
energies. The first two legs are incoming particles, massless by default. These samples
are numerical validation inputs, not integration events: no phase-space
weight or sampling-density claim is attached to them.
"""

import math
from pathlib import Path
import random


DEFAULT_SEED = 20261109


def _masses(values):
    masses = [float(value) for value in values]
    if not masses or any(not math.isfinite(mass) or mass < 0 for mass in masses):
        raise ValueError("External masses must be a nonempty list of finite nonnegative values")
    return masses


def _incoming_masses(values):
    masses = _masses(values)
    if len(masses) != 2:
        raise ValueError("Exactly two incoming masses are required")
    return masses


def _dot3(left, right):
    return math.fsum(a * b for a, b in zip(left, right))


def _invariant(vector):
    return vector[0] ** 2 - _dot3(vector[1:], vector[1:])


def _boost(vector, velocity):
    """Boost a rest-frame vector into a frame moving with ``velocity``."""
    beta2 = _dot3(velocity, velocity)
    if beta2 == 0:
        return list(vector)
    if beta2 >= 1:
        raise ValueError("A Lorentz boost must be subluminal")
    gamma = 1.0 / math.sqrt(1.0 - beta2)
    bp = _dot3(velocity, vector[1:])
    factor = gamma * vector[0] + gamma * gamma / (1.0 + gamma) * bp
    return [gamma * (vector[0] + bp)] + [
        vector[axis + 1] + factor * velocity[axis] for axis in range(3)
    ]


def _massive_rescale(spatial, masses, energy):
    """Solve sum(sqrt(m_i^2 + xi^2 |p_i|^2)) = energy by bisection."""
    if energy <= math.fsum(masses):
        raise ValueError("The centre-of-mass energy must exceed the final-state mass sum")
    norms = [_dot3(vector, vector) for vector in spatial]
    if not any(norms):
        raise ValueError("Cannot rescale an all-zero momentum configuration")

    def total(scale):
        return math.fsum(math.sqrt(mass * mass + scale * scale * norm)
                         for mass, norm in zip(masses, norms))

    low, high = 0.0, 1.0
    while total(high) < energy:
        high *= 2.0
    # Bisection avoids a poorly conditioned Newton derivative near threshold.
    for _ in range(90):
        middle = (low + high) * 0.5
        if total(middle) < energy:
            low = middle
        else:
            high = middle
    scale = (low + high) * 0.5
    return [[math.sqrt(mass * mass + scale * scale * norm)] +
            [scale * component for component in vector]
            for mass, norm, vector in zip(masses, norms, spatial)]


def _rambo(masses, energy, rng):
    """Massless RAMBO conformal map followed by a common massive rescaling."""
    if len(masses) < 2:
        raise ValueError("RAMBO requires at least two final-state particles")
    vectors = []
    for _ in masses:
        cosine = 2.0 * rng.random() - 1.0
        sine = math.sqrt(max(0.0, 1.0 - cosine * cosine))
        phi = 2.0 * math.pi * rng.random()
        # 1-random() lies in (0,1], so log(0) cannot occur.
        magnitude = -math.log(1.0 - rng.random()) - math.log(1.0 - rng.random())
        vectors.append([magnitude, magnitude * sine * math.cos(phi),
                        magnitude * sine * math.sin(phi), magnitude * cosine])
    total = [math.fsum(vector[axis] for vector in vectors) for axis in range(4)]
    if _invariant(total) <= 0:
        raise ValueError("Degenerate RAMBO draw; choose another seed")
    rest = [_boost(vector, [-component / total[0] for component in total[1:]])
            for vector in vectors]
    spatial = [vector[1:] for vector in rest]
    # Remove rounding residual before scaling, without selecting a special leg.
    mean = [math.fsum(vector[axis] for vector in spatial) / len(spatial)
            for axis in range(3)]
    spatial = [[vector[axis] - mean[axis] for axis in range(3)] for vector in spatial]
    return _massive_rescale(spatial, masses, energy)


def _point(identifier, kind, masses, energy, final, incoming_masses):
    first, second = incoming_masses
    if energy <= first + second:
        raise ValueError("The centre-of-mass energy must exceed the incoming mass sum")
    e1 = 0.5 * (energy + (first - second) * (first + second) / energy)
    e2 = energy - e1
    # Factored Kallen function avoids subtracting nearly equal squares at threshold.
    momentum = math.sqrt((energy - first - second) * (energy + first + second) *
                         (energy - first + second) * (energy + first - second)) / (2 * energy)
    return {"id": identifier, "kind": kind, "sqrt_s": energy,
            "masses": list(masses), "incoming_masses": list(incoming_masses), "momenta":
            [[e1, 0.0, 0.0, momentum], [e2, 0.0, 0.0, -momentum]] + final}


def generate_points(masses, count=100, energy=1000.0, seed=DEFAULT_SEED, incoming_masses=(0.0, 0.0)):
    """Return JSON-serializable deterministic 2 -> n phase-space points.

    For n >= 2, ``energy`` is the centre-of-mass energy in GeV. For n = 1,
    the invariant incoming energy must equal the final-state mass; ``energy``
    is therefore ignored and the sample varies longitudinal rapidity in
    [-2, 2]. The first 2 -> 1 sample is the unboosted point. A local RNG keeps
    MadGraph's global random sequence unchanged.
    """
    masses = _masses(masses)
    incoming_masses = _incoming_masses(incoming_masses)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("The point count must be a positive integer")
    energy = float(energy)
    if not math.isfinite(energy) or energy <= 0:
        raise ValueError("The centre-of-mass energy must be finite and positive")
    rng = random.Random(seed)
    points = []
    if len(masses) == 1:
        mass = masses[0]
        if mass <= 0:
            raise ValueError("An on-shell 2 -> 1 final state must have positive mass")
        for index in range(count):
            rapidity = 0.0 if index == 0 else rng.uniform(-2.0, 2.0)
            point = _point("ordinary-%04d" % index, "ordinary", masses, mass,
                           [[mass, 0.0, 0.0, 0.0]], incoming_masses)
            for vector in point["momenta"]:
                plus = (vector[0] + vector[3]) * math.exp(rapidity) * 0.5
                minus = (vector[0] - vector[3]) * math.exp(-rapidity) * 0.5
                vector[0], vector[3] = plus + minus, plus - minus
            point["rapidity"] = rapidity
            points.append(point)
    else:
        if energy <= math.fsum(masses):
            raise ValueError("The centre-of-mass energy must exceed the final-state mass sum")
        for index in range(count):
            points.append(_point("ordinary-%04d" % index, "ordinary", masses, energy,
                                 _rambo(masses, energy, rng), incoming_masses))
    validate_points(points, masses, incoming_masses=incoming_masses)
    return points


def generate_stress_points(masses, energy=1000.0, seed=DEFAULT_SEED, incoming_masses=(0.0, 0.0)):
    """Return separate near-threshold, soft/collinear and small-Gram fixtures.

    Threshold requires a nonzero incoming or final-state mass sum. A soft emission
    requires at least three final legs and a massless leg. Small-Gram points
    approach the beam for two-body final states and a plane containing the
    beam for higher multiplicities. These approach singular regions without
    introducing exactly zero energies or exactly singular kinematics.
    Complex internal masses belong to model parameter cards, not momenta.
    """
    masses = _masses(masses)
    incoming_masses = _incoming_masses(incoming_masses)
    energy = float(energy)
    if not math.isfinite(energy) or energy <= 0:
        raise ValueError("The centre-of-mass energy must be finite and positive")
    if len(masses) == 1:
        if masses[0] <= 0:
            raise ValueError("An on-shell 2 -> 1 final state must have positive mass")
        if masses[0] <= math.fsum(incoming_masses):
            raise ValueError("The on-shell final-state mass must exceed the incoming mass sum")
        return []
    if energy <= math.fsum(masses):
        raise ValueError("The centre-of-mass energy must exceed the final-state mass sum")
    if energy <= math.fsum(incoming_masses):
        raise ValueError("The centre-of-mass energy must exceed the incoming mass sum")
    rng = random.Random(seed)
    points = []
    mass_sum = math.fsum(masses)
    threshold = max(mass_sum, math.fsum(incoming_masses))
    if threshold > 0:
        threshold_energy = threshold * (1.0 + 1e-7)
        points.append(_point("stress-threshold", "threshold", masses, threshold_energy,
                             _rambo(masses, threshold_energy, rng), incoming_masses))

    if len(masses) >= 3 and 0.0 in masses:
        emitted = masses.index(0.0)
        # Keep recoil above threshold, including when energy is already close.
        soft_energy = min(energy * 1e-7, (energy * energy - mass_sum * mass_sum) / (4 * energy))
        angle = 1e-6
        soft = [soft_energy, soft_energy * math.sin(angle), 0.0, soft_energy * math.cos(angle)]
        recoil_energy = energy - soft_energy
        recoil_mass = math.sqrt(energy * energy - 2 * energy * soft_energy)
        rest_masses = masses[:emitted] + masses[emitted + 1:]
        final = [_boost(vector, [-component / recoil_energy for component in soft[1:]])
                 for vector in _rambo(rest_masses, recoil_mass, rng)]
        final.insert(emitted, soft)
        point = _point("stress-soft-collinear", "soft-collinear", masses, energy, final, incoming_masses)
        point["soft_leg"] = emitted + 2
        point["soft_energy_fraction"] = soft_energy / energy
        points.append(point)

    if len(masses) == 2:
        angle = 1e-6
        direction = [math.sin(angle), 0.0, math.cos(angle)]
        spatial = [direction, [-component for component in direction]]
    else:
        final = _rambo(masses, energy, rng)
        spatial = [[vector[1], vector[2] * 1e-7, vector[3]] for vector in final]
    points.append(_point("stress-small-gram", "small-gram", masses, energy,
                         _massive_rescale(spatial, masses, energy), incoming_masses))
    validate_points(points, masses, incoming_masses=incoming_masses)
    return points


def validate_points(points, masses=None, tolerance=1e-10, incoming_masses=None):
    """Check physical inputs; raise ``ValueError`` on the first bad point.

    Momentum conservation uses a relative energy tolerance, and on-shell
    checks use squared energy. All outgoing and incoming energies must be
    positive, with two collinear incoming legs directed along opposite beam
    axes in their centre-of-mass frame. Old point records without incoming
    mass metadata retain the massless convention.
    """
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Validation tolerance must be finite and positive")
    expected_masses = None if masses is None else _masses(masses)
    expected_incoming = None if incoming_masses is None else _incoming_masses(incoming_masses)
    identifiers = set()
    for index, point in enumerate(points):
        identifier = point.get("id", str(index))
        if identifier in identifiers:
            raise ValueError("Duplicate point identifier: %s" % identifier)
        identifiers.add(identifier)
        actual_masses = _masses(point.get("masses", expected_masses or []))
        if expected_masses is not None and actual_masses != expected_masses:
            raise ValueError("Point %s has inconsistent external masses" % identifier)
        actual_incoming = _incoming_masses(point.get("incoming_masses", expected_incoming or [0.0, 0.0]))
        if expected_incoming is not None and actual_incoming != expected_incoming:
            raise ValueError("Point %s has inconsistent incoming masses" % identifier)
        vectors = point.get("momenta", [])
        if len(vectors) != 2 + len(actual_masses):
            raise ValueError("Point %s has the wrong number of momenta" % identifier)
        if any(len(vector) != 4 or any(not math.isfinite(value) for value in vector)
               for vector in vectors):
            raise ValueError("Point %s has malformed or nonfinite momenta" % identifier)
        if any(vector[0] <= 0 for vector in vectors):
            raise ValueError("Point %s has a nonpositive energy" % identifier)
        total = [math.fsum(vector[axis] for vector in vectors[:2]) for axis in range(4)]
        scale = max(1.0, total[0])
        for axis in range(4):
            residual = total[axis] - math.fsum(vector[axis] for vector in vectors[2:])
            if abs(residual) > tolerance * scale:
                raise ValueError("Point %s violates four-momentum conservation" % identifier)
        for mass, vector in zip(actual_incoming + actual_masses, vectors):
            if abs(_invariant(vector) - mass * mass) > tolerance * max(1.0, vector[0] ** 2, mass * mass):
                raise ValueError("Point %s has an off-shell momentum" % identifier)
        if any(abs(vector[axis]) > tolerance * scale for vector in vectors[:2] for axis in (1, 2)):
            raise ValueError("Point %s has transverse incoming momentum" % identifier)
        if vectors[0][3] / vectors[0][0] <= vectors[1][3] / vectors[1][0]:
            raise ValueError("Point %s has incorrectly oriented beams" % identifier)
        sqrt_s = point.get("sqrt_s", float("nan"))
        if not math.isfinite(sqrt_s) or sqrt_s <= 0 or abs(_invariant(total) - sqrt_s ** 2) > tolerance * scale ** 2:
            raise ValueError("Point %s has inconsistent centre-of-mass energy" % identifier)
    return True


def write_points(path, points):
    """Write count followed by E,px,py,pz rows in round-trippable precision."""
    points = list(points)
    validate_points(points)
    lines = [str(len(points))]
    lines.extend(" ".join("%.17e" % component for component in vector)
                 for point in points for vector in point["momenta"])
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")
