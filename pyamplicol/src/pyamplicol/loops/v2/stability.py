# SPDX-License-Identifier: 0BSD
"""Stability estimates returned with an evaluation (REQUIREMENTS_ADDENDUM C).

Two MadLoop-style checks, individually configurable and combinable:

* **Rotation checks.** The point is evaluated again with its external momenta
  rotated by each user-supplied rotation (any number, including none). A
  rotation is an exact rational orthogonal matrix, so the rotated point is
  exactly on shell and every reduction lane's integral (a Lorentz scalar) is
  unchanged in exact arithmetic; the differences measure the rounding of the
  whole evaluation (trees, dressing, reduction, counterterms).
* **Reduction-direction check.** Every family lane is reduced a second time
  with the loop direction reversed (``V2Reducer.reduce(check=True)``, the
  NinjaRS adapter's reversed frame), reusing the numerator coefficients and
  the cached masters. It measures the reduction alone, at the cost of a
  second reduction pass.
* **Rescaled reduction check** (``StabilityChecks(rescaled=True)``). The
  same re-reduction with every dimensionful input scaled by a non-dyadic
  factor (``ReductionFrame::RESCALED``): the masters are evaluated anew at
  differently rounded arguments instead of hitting the cache, so a master
  integral's own rounding (e.g. a binary64 box near a vanishing Gram
  determinant) is measured too. Neither the cached re-reduction nor a
  rotation can see it: both keep every invariant, so every master returns
  the same value. It costs the master evaluations of a second pass.
* **Cross-backend check** (``StabilityChecks(cross_backend=True)``, m-lit
  PLAN §2.b.6, RULED Q21). The point is evaluated a second time on the
  other reduction backend (``reduction.cross``: NinjaRS for a call reduced
  by TIRust, TIRust for one reduced by NinjaRS), in the same carrier and at
  the same exact inputs; its budget is the reduction lanes' deviation
  between the two backends (``CROSS``), as a rotation's at the unrotated
  point. Integrand and tensor-integral reduction round differently
  everywhere (masters, Gram determinants, contraction), so it samples what
  a backend's own frames cannot (a TIR frame keeps the PV reference
  propagators' Gram conditioning). A point the other backend fails or
  refuses (a capability or conditioning refusal says nothing about the
  served value) keeps the estimate of the other checks, with the check in
  ``StabilityEstimate.not_run`` and the reason (DECISIONS 2026-11-07 21:50
  R6); the check therefore combines with at least one other.

Every check yields an absolute error budget per squared-order sector and
Laurent coefficient that lets no two contributions cancel (design §10.2):
sum_f |Omega_f - Omega'_f| over the reduction lanes (bare part), plus, for a
rotation, the deviations of the local components (R2, UV vertex, fields). A
single difference of two evaluations can be accidentally small (its ratio to
the true error has a heavy tail), the budget over many lanes cannot.

The estimate of a configuration combines its checks per sector and Laurent
coefficient as

    E = SAFETY x (max over rotations of their budgets + reduction budget)
        + FLOOR_UNITS x u x S,

with u the unit roundoff of the carrier and S the absolute sum of the
contributions (sum_f |bare_f| plus the absolute local components): the
factor 2 covers a difference of two evaluations with comparable rounding,
which understates either one's error by up to about 2, and the floor covers
roundings no check samples (couplings, kernel constants, the final sums, and
the master integrals unless the rescaled check runs).

Calibration against Float(256) (DESIGN §10.4, O15: 2000 binary64 and 400
double-double points each of gg -> ggg and uu~ -> e-e+dd~g) gives the rate at
which the true error exceeds E in binary64: about 2.5 % for the cached
reduction check alone (it cannot see the masters' own rounding), up to 0.6 %
for one rotation, 0.2 % for the rescaled check and at most 0.1 % for the
rescaled check with a rotation, the recommended configuration (none of the
800 double-double points).

``StabilityEstimate.relative`` measures E against each sector's largest
Laurent coefficient (the comparison metric of the oracles); ``coefficients``
measures it against each coefficient itself, so ``finite`` is the relative
accuracy of the finite part. Without a requested check no estimate exists:
``LoopSlice.evaluate`` then returns ``stability=None``. Escalating the
precision is the caller's decision.

``RECOMMENDED_CHECKS`` is that configuration with ``default_rotation(0)``
(a generic axis, as calibrated); it is the estimate of the O1 rule. Two
weaker choices miss errors by more than ten times their estimate (package
mq-stab, binary64 against double-double):

* A rotation about the beam axis (z for the collision-frame points of a
  2 -> n process) leaves both incoming momenta and the energy and p_z of
  every momentum bitwise unchanged. With the cached reduction check it
  reported 7.6e-14 for the double pole of u u~ > u u~ e+ e- at the fixture's
  point 2, whose error is 1.7e-12 (22.8 x); any rotation moving the beams
  brackets it (error 0.5 x the estimate). The masters are not the cause
  there: every rotation keeps their exact invariants and so their binary64
  inputs bitwise.
* Without the rescaled check no master is evaluated twice: at
  g g > b b~ h (4FS, m_b = 4.7) the finite part's binary64 error sits in
  the masters with the light b and is 23-96 x the estimate of every
  rotation with the cached check; the rescaled check brings it to
  0.14-0.24 x (7.5 x at the worst point).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

SAFETY = 2
FLOOR_UNITS = 16
ROTATION = "rotation-{}"
REDUCTION = "reduction-direction"
RESCALED = "reduction-rescaled"
CROSS = "cross-backend"
# The re-reduction frame of each reduction check (``V2Reducer.reduce``).
FRAMES = {REDUCTION: "reversed", RESCALED: "rescaled"}
# An input budget: an output's change over the inputs its given input
# determines only to within a part the output cannot read (a loop-induced
# spin vector's non-transverse part, ``induced_observables.PointOutputs``).
# Budgets in FRAMES or ADDITIVE add up (the cross-backend check's as a
# reduction check's); the rotations' is their largest.
TRANSVERSE = "spin-transverse"
ADDITIVE = (*FRAMES, TRANSVERSE, CROSS)

_AXES = {"x": 1, "y": 2, "z": 3}
# The binary64 estimates of the native executor's Born-interfered points
# through the one Rust implementation the native SDKs share
# (``native_estimate``; m-lit LIT-SDK, PLAN §3.5); False forms them with this
# module's assembly, the oracle, as every other estimate.
NATIVE_BINARY64 = True


@dataclass(frozen=True, slots=True)
class Rotation:
    """A spatial rotation as an exact rational orthogonal 3x3 matrix."""

    matrix: tuple[tuple[Fraction, Fraction, Fraction], ...]

    def __post_init__(self) -> None:
        matrix = tuple(tuple(Fraction(x) for x in row) for row in self.matrix)
        if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
            raise ValueError("a rotation is a 3x3 matrix")
        identity = tuple(
            tuple(sum(matrix[k][i] * matrix[k][j] for k in range(3)) for j in range(3))
            for i in range(3)
        )
        if identity != ((1, 0, 0), (0, 1, 0), (0, 0, 1)) or _determinant(matrix) != 1:
            raise ValueError("a rotation matrix must be exactly orthogonal, det +1")
        object.__setattr__(self, "matrix", matrix)

    @classmethod
    def about(cls, axis: str, cos: Any, sin: Any) -> Rotation:
        """Rotation about ``axis`` ('x', 'y' or 'z') with exact cos and sin
        (e.g. 3/5 and 4/5); cos^2 + sin^2 must be exactly 1."""

        c, s = Fraction(cos), Fraction(sin)
        if c * c + s * s != 1:
            raise ValueError("cos^2 + sin^2 must be exactly 1")
        index = _AXES[axis] - 1
        a, b = [k for k in range(3) if k != index]
        rows = [[Fraction(int(i == j)) for j in range(3)] for i in range(3)]
        rows[a][a], rows[a][b], rows[b][a], rows[b][b] = c, -s, s, c
        return cls(tuple(tuple(row) for row in rows))

    @classmethod
    def from_axis_angle(
        cls, axis: Sequence[float], angle: float, *, resolution: int = 10**6
    ) -> Rotation:
        """An exact rotation within ~1/``resolution`` of ``angle`` (radians,
        right-handed) about the direction ``axis``.

        The unit quaternion (cos(angle/2), sin(angle/2) n) is rounded to
        rationals with denominators up to ``resolution``; the Euler-Rodrigues
        matrix of any rational quaternion is exactly orthogonal, so the rotated
        point stays exactly on shell.
        """

        norm = math.sqrt(sum(float(x) ** 2 for x in axis))
        if len(axis) != 3 or not norm or not math.isfinite(angle):
            raise ValueError("a rotation needs a non-zero 3-axis and a finite angle")
        half = angle / 2
        return cls._from_quaternion(
            tuple(
                Fraction(value).limit_denominator(resolution)
                for value in (
                    math.cos(half),
                    *(math.sin(half) * float(x) / norm for x in axis),
                )
            )
        )

    @classmethod
    def from_euler(
        cls, alpha: float, beta: float, gamma: float, *, resolution: int = 10**12
    ) -> Rotation:
        """The exact rotation closest to Rz(gamma) Ry(beta) Rx(alpha)
        (radians, active right-handed rotations: Rx acts first), gammaloop's
        ``euler_angles`` convention (``RotationSetting::EulerAngles``).

        The unit quaternion of the composed rotation, qz(gamma) qy(beta)
        qx(alpha), is rounded once to integers over the common denominator
        ``resolution``; the Euler-Rodrigues matrix of the integer quaternion
        is exactly orthogonal, its entries integers over the quaternion's norm
        (about ``resolution``^2), within about 1e-12 of the floating-point
        matrix at the default resolution."""

        if not all(math.isfinite(x) for x in (alpha, beta, gamma)):
            raise ValueError("Euler angles must be finite")

        def half(angle: float, axis: int) -> tuple[float, float, float, float]:
            q = [math.cos(angle / 2), 0.0, 0.0, 0.0]
            q[axis] = math.sin(angle / 2)
            return q[0], q[1], q[2], q[3]

        Quaternion = tuple[float, float, float, float]

        def product(p: Quaternion, q: Quaternion) -> Quaternion:  # Hamilton
            a1, b1, c1, d1 = p
            a2, b2, c2, d2 = q
            return (
                a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
                a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
                a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
            )

        quaternion = product(
            product(half(gamma, _AXES["z"]), half(beta, _AXES["y"])),
            half(alpha, _AXES["x"]),
        )
        return cls._from_quaternion(
            tuple(Fraction(round(x * resolution)) for x in quaternion)
        )

    @classmethod
    def _from_quaternion(cls, quaternion: Sequence[Fraction]) -> Rotation:
        """The Euler-Rodrigues matrix of a rational quaternion (a, b, c, d),
        exactly orthogonal for any nonzero quaternion."""

        a, b, c, d = quaternion
        n = a * a + b * b + c * c + d * d
        if not n:
            raise ValueError("the rounded quaternion vanishes; raise the resolution")
        rows = (
            (a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)),
            (2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)),
            (2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d),
        )
        return cls(tuple(tuple(x / n for x in row) for row in rows))

    @property
    def is_identity(self) -> bool:
        return self.matrix == ((1, 0, 0), (0, 1, 0), (0, 0, 1))

    def __matmul__(self, other: Rotation) -> Rotation:
        return Rotation(
            tuple(
                tuple(
                    sum(self.matrix[i][k] * other.matrix[k][j] for k in range(3))
                    for j in range(3)
                )
                for i in range(3)
            )
        )

    def apply(
        self, point: Sequence[Sequence[Fraction]]
    ) -> tuple[tuple[Fraction, ...], ...]:
        """Rotate the spatial parts of every momentum (E, px, py, pz)."""

        out = []
        for momentum in point:
            energy, *space = (Fraction(x) for x in momentum)
            rotated = [
                sum(self.matrix[i][k] * space[k] for k in range(3)) for i in range(3)
            ]
            out.append((energy, *rotated))
        return tuple(out)


def default_rotation(index: int) -> Rotation:
    """The ``index``-th default rotation check: exact, deterministic, generic.

    Axes follow a spherical Fibonacci sequence and angles golden-ratio steps
    of a turn, so no two checks share an axis or a rotation angle, none is a
    coordinate axis (the beam axis included: module docstring) and none is
    a symmetry of a typical point. ``Rotation.from_axis_angle`` rounds the
    quaternion to rationals, which keeps every rotated point exactly on shell.
    """

    golden = (1 + math.sqrt(5)) / 2
    z = 1 - (2 * index + 1) / (2 * (index + 2))
    phi = 2 * math.pi * index / golden
    radius = math.sqrt(1 - z * z)
    angle = 2 * math.pi * (((index + 1) / golden) % 1)
    return Rotation.from_axis_angle(
        (radius * math.cos(phi), radius * math.sin(phi), z), max(angle, 0.5)
    )


def _determinant(m) -> Fraction:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


@dataclass(frozen=True, slots=True)
class StabilityChecks:
    """Which checks an evaluation runs: rotations (any number) and/or one
    reduction check, the reduction-direction check (cached masters) or, with
    ``rescaled``, its rescaled variant (masters recomputed); and, with
    ``cross_backend``, the evaluation on the other reduction backend (with
    at least one other check: module docstring). The default runs none."""

    rotations: tuple[Rotation, ...] = ()
    reduction_direction: bool = False
    rescaled: bool = False
    cross_backend: bool = False

    def __post_init__(self) -> None:
        rotations = tuple(self.rotations)
        if any(not isinstance(r, Rotation) for r in rotations):
            raise TypeError("rotation checks take Rotation records")
        if any(r.is_identity for r in rotations):
            raise ValueError("the identity is no rotation check")
        if self.rescaled and not self.reduction_direction:
            raise ValueError("rescaled is a variant of the reduction-direction check")
        if self.cross_backend and not (rotations or self.reduction_direction):
            raise ValueError(
                "the cross-backend check adds to the estimate of the other "
                "checks (a point the other backend does not serve keeps "
                "theirs): combine it with a rotation or the reduction-direction "
                "check"
            )
        object.__setattr__(self, "rotations", rotations)

    @property
    def enabled(self) -> bool:
        return bool(self.rotations) or self.reduction_direction

    @property
    def reduction(self) -> str | None:
        """The name of the reduction check that runs (``REDUCTION`` or
        ``RESCALED``), or None."""
        if not self.reduction_direction:
            return None
        return RESCALED if self.rescaled else REDUCTION

    @property
    def frame(self) -> str | None:
        """The re-reduction frame of the reduction check, or None."""
        return FRAMES.get(self.reduction)


# The recommended configuration (DESIGN §10.4) and the O1 rule's estimate:
# one generic rotation and the rescaled reduction check (module docstring).
RECOMMENDED_CHECKS = StabilityChecks(
    (default_rotation(0),), reduction_direction=True, rescaled=True
)

# The default ladder of the built-in rescue (``api.virtual_results.
# VirtualRescuePolicy()``, ``loops.v2.rescue``), modelled on gammaloop's
# ``StabilitySettings`` defaults (f64 -> f128 = double-double -> arb, each at
# 1e-5 with the one rotation ``euler_angles(0.1, 0.2, 0.3)``). Deviations:
# every level adds the rescaled reduction check (rotations keep every master
# invariant and cannot see a master integral's own rounding, module
# docstring); the last level is 57 digits (O15's truth carrier) instead of
# 1000 bits; the target applies to pyAmpliCol's estimate E.
RESCUE_ROTATION = "euler:0.1,0.2,0.3"
RESCUE_TARGET = 1e-5
DEFAULT_RESCUE: tuple[tuple[str, int | None], ...] = (
    ("double", None),
    ("double-double", None),
    ("arbitrary", 57),
)


def relative(reference: Mapping[Any, Any], absolute: Mapping[Any, Any]) -> float:
    """Largest absolute deviation relative to the size of its Laurent vector.

    ``reference`` maps sectors to ``Laurent`` values and ``absolute`` to
    ``Laurent`` absolute deviations. Each sector's deviations are measured
    against its largest coefficient (the comparison metric of the oracles), so
    a pole that nearly cancels between components does not turn rounding of
    its parts into an arbitrarily large relative error.
    """

    worst = 0.0
    for sector, laurent in reference.items():
        scale = max(abs(float(x)) for x in laurent.coefficients)
        errors = absolute[sector].coefficients
        if scale:
            worst = max(worst, max(abs(float(x)) for x in errors) / scale)
    return worst


def per_coefficient(
    reference: Mapping[Any, Any], absolute: Mapping[Any, Any]
) -> tuple[float, ...]:
    """Largest absolute deviation of each Laurent coefficient relative to the
    coefficient itself, over sectors (``inf`` for a deviation of an exact 0)."""

    width = len(next(iter(reference.values())).coefficients)
    worst = [0.0] * width
    for sector, laurent in reference.items():
        for k, (value, error) in enumerate(
            zip(laurent.coefficients, absolute[sector].coefficients, strict=True)
        ):
            value, error = abs(float(value)), abs(float(error))
            if error:
                worst[k] = max(worst[k], error / value if value else math.inf)
    return tuple(worst)


def deviation(reference: Mapping[Any, Any], other: Mapping[Any, Any]) -> float:
    """Relative deviation of ``other``'s Laurent coefficients from the
    ``reference`` ones over sectors (``relative`` of their differences)."""

    return relative(reference, differences(reference, other))


def differences(reference: Mapping[Any, Any], other: Mapping[Any, Any]) -> dict:
    """|other - reference| per sector and Laurent coefficient."""

    return {
        sector: Budget(
            tuple(
                abs(b - a)
                for a, b in zip(
                    laurent.coefficients, other[sector].coefficients, strict=True
                )
            )
        )
        for sector, laurent in reference.items()
    }


@dataclass(frozen=True, slots=True)
class Budget:
    """Absolute errors per Laurent coefficient (a ``Laurent``-like record)."""

    coefficients: tuple[float, ...]


@dataclass(frozen=True)
class StabilityEstimate:
    """The relative accuracy estimate of one evaluation (module docstring).

    ``relative`` is E against each sector's largest Laurent coefficient,
    ``coefficients`` E per Laurent coefficient against that coefficient
    (finite, then the poles when computed), ``absolute`` E itself per sector
    and Laurent coefficient. ``checks`` holds each check's own budget against
    the Laurent vector, without safety factor and floor. ``budgets``
    (absolute, per check and sector) and ``scale`` (the absolute sums S) let
    ``subset`` re-combine fewer checks; ``families`` holds |Omega - Omega'|
    per reduction lane when the reduction check ran. ``not_run`` names each
    requested check that did not run at the point with its reason (the
    cross-backend check where the other backend fails the point).

    A relative estimate of order 1 or more says that the value has no
    correct digit: it is then measured against a value that is itself
    wrong, and only ``absolute`` keeps its meaning.
    """

    relative: float
    coefficients: tuple[float, ...]
    checks: Mapping[str, float]
    absolute: Mapping[Any, Budget] = field(repr=False)
    budgets: Mapping[str, Mapping[Any, Budget]] = field(repr=False)
    reference: Mapping[Any, Any] = field(repr=False)
    scale: Mapping[Any, Budget] = field(repr=False)
    roundoff: float = field(repr=False)
    families: Mapping[Any, Any] | None = field(default=None, repr=False)
    not_run: Mapping[str, str] = field(default_factory=dict)

    @property
    def finite(self) -> float:
        """The relative accuracy estimate of the finite part."""
        return self.coefficients[0]

    @classmethod
    def from_checks(
        cls,
        reference: Mapping[Any, Any],
        budgets: Mapping[str, Mapping[Any, Any]],
        *,
        scale: Mapping[Any, Any],
        roundoff: float,
        families: Mapping[Any, Any] | None = None,
        not_run: Mapping[str, str] | None = None,
    ) -> StabilityEstimate:
        """Combine the absolute budgets of named checks (``ROTATION``,
        ``REDUCTION`` and ``ADDITIVE`` names) for the reference Laurent values
        per sector; ``scale`` are the absolute sums S and ``roundoff`` the
        carrier's unit roundoff; ``not_run`` the requested checks that did
        not run, with their reasons."""

        if not budgets:
            raise ValueError("a stability estimate needs at least one check")
        unknown = [
            name
            for name in budgets
            if name not in ADDITIVE and not name.startswith(ROTATION.format(""))
        ]
        if unknown:
            raise ValueError(f"unknown stability checks {unknown}")
        rotations = [b for name, b in budgets.items() if name not in ADDITIVE]
        reductions = [b for name, b in budgets.items() if name in ADDITIVE]
        combined = {}
        for sector, laurent in reference.items():
            width = len(laurent.coefficients)

            def sampled(k: int, sector=sector) -> float:
                rotated = max(
                    (_at(r[sector], k) for r in rotations),
                    default=0.0,
                )
                return rotated + sum(_at(r[sector], k) for r in reductions)

            combined[sector] = Budget(
                tuple(
                    SAFETY * sampled(k) + FLOOR_UNITS * roundoff * _at(scale[sector], k)
                    for k in range(width)
                )
            )
        return cls(
            relative(reference, combined),
            per_coefficient(reference, combined),
            {name: relative(reference, item) for name, item in budgets.items()},
            combined,
            dict(budgets),
            reference,
            scale,
            roundoff,
            families,
            dict(not_run or {}),
        )

    def subset(self, names: Iterable[str]) -> StabilityEstimate:
        """The estimate an evaluation with only the checks ``names`` returns."""

        names = tuple(names)
        missing = [name for name in names if name not in self.budgets]
        if missing:
            raise KeyError(f"checks {missing} did not run")
        return StabilityEstimate.from_checks(
            self.reference,
            {name: self.budgets[name] for name in names},
            scale=self.scale,
            roundoff=self.roundoff,
            families=self.families if set(names) & set(FRAMES) else None,
            not_run=self.not_run,
        )


@dataclass(frozen=True)
class CrossRuns:
    """The cross-backend check's evaluations of one batch on the other
    backend (named ``backend``; ``plan_export.NativeExecutor._cross_runs``):
    the per-point ``runs``, the ``failures`` per point index, or the
    ``refusal`` of the whole call (the other backend's typed refusal of the
    artifact, e.g. a family outside its capability)."""

    backend: str
    runs: Sequence[Any]
    failures: Mapping[int, Sequence[str]]
    refusal: str | None = None

    def of(self, index: int) -> tuple[Any, str]:
        """(the point's run, "") or (None, why the check did not run)."""

        if self.refusal is not None:
            return None, f"{self.backend} refused the call: {self.refusal}"
        reasons = self.failures.get(index)
        if reasons:
            return None, f"{self.backend} failed the point: " + "; ".join(reasons)
        return self.runs[index], ""

    def attach(self, evaluations: Sequence[Any]) -> None:
        """Set ``cross_run`` (``of``) on each point's reference evaluation."""

        for index, evaluation in enumerate(evaluations):
            evaluation.cross_run = self.of(index)


def cross_check(reference: Any, budgets: dict) -> dict[str, str]:
    """Add the cross-backend check's budget of one point to ``budgets``:
    the deviation of ``reference`` (a point evaluation) from its evaluation
    on the other backend (``reference.cross_run``, ``CrossRuns.of``) as a
    rotation's at the unrotated point (every lane and local component;
    loop-induced: lane by lane without a little-group phase,
    ``induced.cross_budget``). Returns the ``not_run`` record: the check and
    the reason when the other backend did not serve the point."""

    run, why = reference.cross_run
    if run is None:
        return {CROSS: why}
    if reference.slice.amplitude is not None:
        from . import induced

        budgets[CROSS] = induced.cross_budget(reference, run)
    else:
        budgets[CROSS] = reference.rotation_budget(run)
    return {}


def native_estimate(
    reference: Any, rotated: Sequence[Any], checks: StabilityChecks, roundoff: float
) -> StabilityEstimate | None:
    """The stability estimate of one binary64 Born-interfered point of the
    native executor (``plan_export.point_estimate``'s) by the Rust
    implementation the native SDKs run (``rusticol_loop_reduction::
    v2_stability``, bitwise this module's assembly): the point's reduction
    lanes and local components, those of its ``rotated`` evaluations, its
    re-test lanes and its cross-backend run. None where it does not apply
    (``NATIVE_BINARY64`` off, an extended carrier, a loop-induced point):
    the Python assembly forms that estimate."""

    if (
        not NATIVE_BINARY64
        or not reference.numbers.double
        or reference.slice.amplitude is not None
    ):
        return None
    from pyamplicol.loops.native import load_loop_native

    from .assemble import LOCAL_COMPONENTS

    sectors = [sector for sector, _pairs in reference.slice.sectors]
    index = {sector: k for k, sector in enumerate(sectors)}
    components = reference.result.components
    # The local components enter the budgets where the point has them (a
    # native point without counterterm outputs has the bare part alone).
    local = (
        LOCAL_COMPONENTS if all(name in components for name in LOCAL_COMPONENTS) else ()
    )
    keys: dict[Any, int] = {}

    def lanes(families: Mapping[Any, Any]) -> list:
        return [
            (
                keys.setdefault(lane, len(keys)),
                index[lane[2]],
                [(x.real, x.imag) for x in laurent.coefficients],
            )
            for lane, laurent in families.items()
        ]

    def point(evaluation: Any) -> tuple:
        found = evaluation.result.components
        return (
            lanes(evaluation.families),
            [
                [
                    [float(x) for x in found[name][sector].coefficients]
                    for sector in sectors
                ]
                for name in local
            ],
        )

    values = components["virtual"] if "virtual" in components else components["bare"]
    retest = (
        None
        if checks.reduction is None
        else (checks.reduction, lanes(reference.retest_families))
    )
    cross = reason = None
    if checks.cross_backend:
        run, why = reference.cross_run
        if run is None:
            reason = why
        else:
            cross = point(run)
    found = load_loop_native().stability_born_interfered(
        (len(sectors), 3 if reference.poles else 1, float(reference.average)),
        roundoff,
        [[float(x) for x in values[sector].coefficients] for sector in sectors],
        point(reference),
        [point(run) for run in rotated],
        retest,
        cross,
        reason,
    )
    relative_, coefficients, relatives, absolute, budgets, scale, not_run = found

    def budget(rows: Sequence[Sequence[float]]) -> dict:
        return {
            sector: Budget(tuple(row))
            for sector, row in zip(sectors, rows, strict=True)
        }

    return StabilityEstimate(
        relative_,
        tuple(coefficients),
        dict(relatives),
        budget(absolute),
        {name: budget(rows) for name, rows in budgets},
        values,
        budget(scale),
        roundoff,
        reference.family_deviations,
        dict(not_run),
    )


def _at(budget: Any, k: int) -> float:
    return abs(float(budget.coefficients[k]))


__all__ = [
    "ADDITIVE",
    "CROSS",
    "DEFAULT_RESCUE",
    "FLOOR_UNITS",
    "FRAMES",
    "NATIVE_BINARY64",
    "RECOMMENDED_CHECKS",
    "REDUCTION",
    "RESCALED",
    "RESCUE_ROTATION",
    "RESCUE_TARGET",
    "ROTATION",
    "SAFETY",
    "TRANSVERSE",
    "Budget",
    "CrossRuns",
    "Rotation",
    "StabilityChecks",
    "StabilityEstimate",
    "cross_check",
    "default_rotation",
    "deviation",
    "differences",
    "native_estimate",
    "per_coefficient",
    "relative",
]
