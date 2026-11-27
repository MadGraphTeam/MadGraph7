# SPDX-License-Identifier: 0BSD
"""Explicit general-SU(Nc) continuation of the local four-gluon QCD R2 rule.

The HV coefficients are arXiv:2007.03713, Eqs. (5.30)--(5.38), at one loop,
with Tr(Ta Tb)=delta_ab/2. They independently agree with arXiv:0903.0356,
Fig. 2, at lambda_HV=1. This is a whole-rule replacement, never an inferred
continuation of individual numerical UFO coefficients. Dependency tensors own
all colour/Lorentz operations; original loop alternatives own multiplicities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Any, Literal

from .._internal.physics.symbols import symbols
from .exact import require_exact_numbers
from .model import CovariantKernelCatalogue, KernelBasisEntry, NativeLocalKernel
from .model_expressions import parse_exact_ufo

# The four-gluon R2 Lorentz structures of the UFO source (compared with
# generated ALOHA).
_FOUR_R2_LORENTZ = (
    "Metric(1,4)*Metric(2,3)",
    "Metric(1,3)*Metric(2,4)",
    "Metric(1,2)*Metric(3,4)",
    "Metric(1,4)*Metric(2,3)+Metric(1,3)*Metric(2,4)+Metric(1,2)*Metric(3,4)",
)


@dataclass(frozen=True, slots=True)
class QCDR2Term:
    kernel: NativeLocalKernel
    coefficient: Any


@dataclass(frozen=True, slots=True)
class QCDR2Continuation:
    """A locally checked replacement for exactly one source CT loop group.

    Colour ports remain in their physical source frame. ``SUNColorSpace`` must
    lift these declared fundamental/adjoint tensors before formal projection.
    The explicit Nc dependence of their scalar coefficients is already present.
    Source entries retain paired diagnostics, eligibility, orders and species;
    consumers must preserve those when binding this whole replacement.
    The scheme is an explicit integration precondition. Any adjustable source
    scheme parameters must remain tied to HV; the Nc=3 tensor check is not a
    certificate for a different dimensional or gamma-five prescription.
    """

    source_entries: tuple[KernelBasisEntry, ...] = field(repr=False)
    terms: tuple[QCDR2Term, ...]
    nc: Any
    gauge_coupling: Any
    contribution: Literal["gauge", "fundamental-dirac"]
    scheme: str = field(default="HV", init=False)

    @property
    def key(self):
        entry = self.source_entries[0]
        return entry.collection, entry.vertex_id, entry.loop_group_index, entry.order

    @property
    def loop_particles(self):
        return self.source_entries[0].loop_particles


def _terms(nc, coupling, contribution, frame, names):
    from symbolica import E
    from symbolica.community import hepkit as h
    from symbolica.community import tensor as t

    ports = tuple(h.Symbols.half_edge()(half, 1) for half in frame)
    metric = t.TensorExpression.g(t.Representation.mink(4))
    delta = t.TensorExpression.g(t.Representation.coad(8))
    generator = t.TensorExpression.color_t(8, 3)
    prefactor = E("1i") * coupling**4 / (16 * symbols.qualified("symbolica::pi") ** 2)
    terms = []
    for order in permutations((1, 2, 3)):
        a, b, c, d = (ports[i] for i in (0, *order))
        vi = metric(a, b) * metric(c, d)
        vii = metric(a, c) * metric(b, d) + metric(a, d) * metric(b, c)
        cb = delta(a, b) * delta(c, d) / 2
        cc = 2 * t.trace(
            t.Representation.cof(3),
            *(generator(label, t.AUTO, t.AUTO) for label in (a, c, b, d)),
        )
        if contribution == "gauge":
            coefficients = (
                (cb, vi, -E("1/3")),
                (cb, vii, -E("1/3")),
                (cc, vi, -E("8/3") * nc),
                (cc, vii, E("7/6") * nc),
            )
        else:
            # One Dirac fundamental species per retained source alternative.
            coefficients = ((cc, vi, -E("5/3")), (cc, vii, E("1")))
        terms.extend(
            QCDR2Term(NativeLocalKernel(color, lorentz, frame, names), prefactor * c)
            for color, lorentz, c in coefficients
        )
    return tuple(terms)


def _metric_components(kernel):
    """Native components separating the three certified metric products."""
    from symbolica.community import hepkit as h

    value = kernel.lorentz.to_tensor().to_dense()
    ports = tuple(h.Symbols.half_edge()(half, 1) for half in kernel.leg_half_edges)
    axes = tuple(ports.index(axis.index) for axis in value.axes)
    return tuple(
        value[tuple(component[index] for index in axes)]
        for component in ((0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 0))
    )


def _check_source_tensor(entries, catalogue, terms, nc, coupling):
    """Compare every SU(3) colour component in an exact Lorentz basis."""
    from symbolica import E

    # The accepted source Lorentz structures and the analytic rule both lie in
    # the three metric-product span. These three independent components then
    # test the entire off-shell tensor, not selected on-shell polarizations.
    known = tuple(parse_exact_ufo(value) for value in _FOUR_R2_LORENTZ)
    source, replacement = [None] * 3, [None] * 3
    prefactor = E("1i") * coupling**4 / (16 * symbols.qualified("symbolica::pi") ** 2)
    for entry in entries:
        if not any(
            (parse_exact_ufo(entry.lorentz) - value).expand() == 0 for value in known
        ):
            raise NotImplementedError(
                "four-gluon R2 source is outside the checked metric basis"
            )
        if entry.coupling.value.expression != entry.coupling.value.coefficient(0):
            raise ValueError(
                "four-gluon R2 source contains non-finite epsilon coefficients"
            )
        # This is symbolic conformance, not value-only numerical evaluation.
        # Keep the paired diagnostic on the original entry in the returned
        # record; the consumer must bind it along with the replacement rule.
        coefficient = (entry.scalar_expression(0).value / prefactor).together()
        kernel = catalogue.lower(entry)
        for index, component in enumerate(_metric_components(kernel)):
            if component != 0:
                term = coefficient * component * kernel.color
                source[index] = term if source[index] is None else source[index] + term
    for term in terms:
        coefficient = (term.coefficient.replace(nc, 3) / prefactor).together()
        for index, component in enumerate(_metric_components(term.kernel)):
            if component != 0:
                value = coefficient * component * term.kernel.color
                replacement[index] = (
                    value if replacement[index] is None else replacement[index] + value
                )
    for left, right in zip(source, replacement, strict=True):
        assert left is not None and right is not None
        difference = (left - right).to_tensor().to_dense()
        if any(difference[index].together() != 0 for index in range(len(difference))):
            raise ValueError(
                "four-gluon R2 whole source tensor differs from the HV rule at Nc=3"
            )


def prepare_qcd_four_gluon_r2(
    source,
    catalogue: CovariantKernelCatalogue,
    entries,
    nc,
    *,
    gauge_coupling,
    contribution: Literal["gauge", "fundamental-dirac"],
    scheme: str,
) -> QCDR2Continuation:
    """Prepare one explicit HV rule, after exact whole-source SU(3) conformance.

    ``entries`` must contain a complete original vertex/group/order collection.
    No model, vertex, coupling or particle name grants a physics certificate.
    The coupling expression is explicit; constants are never guessed to be Nc.
    This validates the local rational tensor only, not UV or process coverage.
    Single-species loop alternatives are the certified multiplicity domain of
    this published local rule, not a restriction on general CT loop groups.
    Each retained fundamental alternative represents one Dirac species; the
    gauge contribution includes the complete Yang--Mills gauge/ghost sector.
    """
    from symbolica import AtomType, Expression

    if scheme != "HV":
        raise NotImplementedError("four-gluon general-Nc R2 is prepared only for HV")
    if contribution not in ("gauge", "fundamental-dirac"):
        raise ValueError("unknown four-gluon R2 loop contribution")
    if not isinstance(nc, Expression) or nc.get_type() != AtomType.Var:
        raise TypeError("formal Nc must be a symbolic variable")
    if not isinstance(gauge_coupling, Expression) or gauge_coupling == 0:
        raise ValueError("gauge coupling must be an explicit nonzero exact expression")
    require_exact_numbers(gauge_coupling)
    if nc in gauge_coupling.get_all_symbols():
        raise ValueError("source gauge coupling must be independent of formal Nc")
    if catalogue.model is not source:
        raise ValueError("kernel catalogue belongs to a different source")
    entries = tuple(entries)
    if not entries:
        raise ValueError("four-gluon R2 source group is empty")
    first = entries[0]

    def key(entry):
        return entry.collection, entry.vertex_id, entry.loop_group_index, entry.order

    expected = tuple(entry for entry in source.kernels if key(entry) == key(first))
    if len(entries) != len(expected) or {id(e) for e in entries} != {
        id(e) for e in expected
    }:
        raise ValueError("four-gluon R2 requires the complete original source group")
    if first.kind != "R2" or first.loop_group_index is None:
        raise ValueError("four-gluon continuation requires an R2 loop group")
    particles = source.raw.collections["particles"]
    fields = tuple(particles[i]["fields"] for i in first.particle_ids)
    if (
        len(fields) != 4
        or len(set(first.particle_ids)) != 1
        or any((p["spin"], p["color"]) != (3, 8) for p in fields)
        or any(
            e.particle_ids != first.particle_ids
            or e.loop_particles != first.loop_particles
            for e in entries
        )
    ):
        raise ValueError(
            "four-gluon R2 requires identical adjoint vector ports and loop groups"
        )
    if not first.loop_particles or any(len(alt) != 1 for alt in first.loop_particles):
        raise NotImplementedError(
            "four-gluon R2 requires explicit single-species alternatives"
        )
    for alternative in first.loop_particles:
        particle = particles[alternative[0]]["fields"]
        if contribution == "gauge":
            valid = alternative[0] == first.particle_ids[0]
        else:
            valid = particle["spin"] == 2 and abs(particle["color"]) == 3
        if not valid:
            raise ValueError(
                "four-gluon R2 contribution disagrees with source loop species"
            )
    frame = catalogue.lower(first).leg_half_edges
    terms = _terms(nc, gauge_coupling, contribution, frame, first.particle_names)
    _check_source_tensor(entries, catalogue, terms, nc, gauge_coupling)
    return QCDR2Continuation(entries, terms, nc, gauge_coupling, contribution)
