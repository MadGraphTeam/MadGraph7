"""PRIVATE pure symbolic routing draft; no numerical policy or native calls.

Coefficient vectors multiply physical external momenta P(:, leg).  They are
never simplified by momentum conservation inside the routing transformation.
Saved generated-source extraction and native trace validation remain pending.
"""
from dataclasses import dataclass


def require(condition, message):
    if not condition:
        raise ValueError(message)


def add(vectors, width):
    vectors = tuple(tuple(v) for v in vectors)
    require(all(len(v) == width for v in vectors), "coefficient width mismatch")
    require(all(type(x) is int for v in vectors for x in v), "noninteger coefficient")
    return tuple(sum(v[k] for v in vectors) for k in range(width))


def negate(vector):
    return tuple(-x for x in vector)


def subtract(left, right):
    return add((left, negate(right)), len(left))


@dataclass(frozen=True)
class ExternalProvenance:
    leg: int
    initial: bool
    model_mass: str
    offshell: bool | None = None
    onshell_decay_flag: bool | None = None
    repair_mass_expression: str | None = None
    # onshell_decay_flag is bookkeeping, NOT a physical-shell predicate.


def corners_from_partition(supports, external):
    """For already-verified ordered HELAS corner supports, preserve signs.

    get_struct_external_leg_ids uses sets, so reject duplicated/missing leaves
    before passing supports here; this helper also checks across corners.
    """
    supports = tuple(tuple(s) for s in supports)
    external = tuple(external)
    width = len(external)
    require(all(type(e.initial) is bool for e in external), "unknown external state")
    require(tuple(e.leg for e in external) == tuple(range(1, width + 1)),
            "external leg IDs must be contiguous and ordered")
    leaves = [leg for support in supports for leg in support]
    require(sorted(leaves) == list(range(1, width + 1)),
            "corner supports must partition external legs exactly once")
    return tuple(tuple((1 if e.initial else -1) if e.leg in support else 0
                       for e in external) for support in supports)


def route(corners, raw_mass_expressions, mode):
    """Mirror optimized exporter mass rotation and TIRust shifts exactly.

    q[0] is the conservation coefficient vector, NOT an exact zero.  Masses
    are expression labels only; this helper squares/evaluates no real value.
    """
    corners = tuple(tuple(c) for c in corners)
    n = len(corners)
    require(1 <= n <= 6, "draft limited to one through six corners")
    require(type(mode) is int and 1 <= mode <= 6, "invalid check mode")
    width = len(corners[0])
    add(corners, width)
    masses = tuple(raw_mass_expressions)
    require(len(masses) == n, "mass count mismatch")
    masses = masses[-1:] + masses[:-1]
    mass_origins = (n,) + tuple(range(1, n))
    reversed_order = mode in (2, 5) and n > 2
    if reversed_order:
        corners = tuple(negate(c) for c in reversed(corners))
        masses = masses[:1] + tuple(reversed(masses[1:]))
        mass_origins = mass_origins[:1] + tuple(reversed(mass_origins[1:]))
    residual = add(corners, width)
    shifts = (residual,) + tuple(add(corners[:i], width) for i in range(1, n))
    return {"corners": corners, "shifts": shifts,
            "conservation_residual": residual,
            "squared_mass_source_expressions": masses,
            "squared_mass_source_indices": mass_origins,
            "squaring_precision": "DP in optimized generated LOOP routine",
            "check_reversal_applied": reversed_order}


def conservation_equivalence(left, right, conservation):
    """Return integer k for left = right + k*conservation, or None.

    A result is a symbolic conditional equivalence; it does not assert that
    actual floating-point momenta sum to zero or are on shell.
    """
    delta = subtract(left, right)
    require(len(conservation) == len(delta), "coefficient width mismatch")
    require(all(type(c) is int for c in conservation), "noninteger conservation coefficient")
    nonzero = [i for i, x in enumerate(conservation) if x]
    if not nonzero:
        return 0 if not any(delta) else None
    i = nonzero[0]
    q, rem = divmod(delta[i], conservation[i])
    if rem or any(d != q * c for d, c in zip(delta, conservation)):
        return None
    return q


def reduction_identifier(has_born, group_loops, group_id_one_based, amplitude_number):
    """Mirror emitted LOOPNUM identity, preserving distinction from groups."""
    return group_id_one_based if has_born and group_loops else amplitude_number
