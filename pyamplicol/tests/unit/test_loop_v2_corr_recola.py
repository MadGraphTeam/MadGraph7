# SPDX-License-Identifier: 0BSD
"""Correlated Borns and virtuals against Recola's correlation APIs (MF-CORR).

References: ``tests/fixtures/loop/required/corr/<process>.json``, written by
``tools/loop_reference/recola_correlations.py`` (Recola2 2.2.5 from the
recola2-2.3.0 tree, binary64): at two exact points every ordered colour
dipole (``compute_colour_correlation_rcl``), every gluon spin correlation
(``compute_spin_correlation_rcl``) and every spin-colour correlation
(``compute_spin_colour_correlation_rcl``: gluon spin leg; quark, antiquark,
massive-top or gluon colour partner), at LO and as the Born-virtual
interference with its IR poles. Spin vectors are complex and transverse:
Recola projects them onto the two physical helicities, pyAmpliCol replaces
the source literally, and the two agree for v.k = 0.

pyAmpliCol's public outputs (docs/correlator-conventions.md): the tree
engine's ``Runtime.evaluate_correlated_many`` and the one-loop runtime's
``evaluate_correlated_many`` (Born and Laurent coefficients), dipoles
``T_i.T_j`` and literal spin replacement. Conversions, applied once here:
``<T_i.T_j> = C_i cc(i, j)``, ``<M(v)|T_g.T_j|M(v)> = C_A scc(g, j)`` and
``F_BLHA = F_Recola + pi^2/6 IR2``. Neither tool offers a quark-leg (spinor)
spin correlation: the quark rows are the spin and spin-colour correlations
of the gluons of processes with external quarks.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal
from itertools import combinations

import pytest

from tools.loop_reference import required_spec

pytestmark = pytest.mark.loop_nightly

PROCESSES = ("uux_gg", "gg_ttx", "uux_ddxg", "gg_ggg")
# Born: exact trees on both sides in binary64 (observed <= 2e-14).
BORN_RELATIVE = 1e-11
# Virtual, relative to the row's largest Laurent coefficient: Recola's
# binary64 COLLIER integrals bound the agreement (observed <= 1.3e-9).
VIRTUAL_RELATIVE = 1e-8
CASIMIR = {3: 4 / 3, 8: 3.0}  # C_F, C_A by colour dimension


def _fixture(loop_reference, name: str) -> dict:
    return loop_reference(f"required/corr/{name}")


def _spec(name: str) -> dict:
    return required_spec.load_process(required_spec.DEFAULT_SPEC, name)


def _expression(name: str) -> str:
    """The MG5-syntax process of the spec (coupling orders and the
    correction bracket removed)."""

    tokens = _spec(name)["madloop"]["process"].split()
    return " ".join(t for t in tokens if "=" not in t and not t.startswith("["))


def _colours(data: dict) -> list[int]:
    return [required_spec.particle(pdg)["colour"] for pdg in data["pdgs"]]


def _declarations(data: dict):
    """Every dipole of the coloured legs and every spin leg of the fixture."""

    from pyamplicol import ColorCorrelator, CorrelatorConfig

    coloured = [k for k, colour in enumerate(_colours(data), 1) if colour != 1]
    spin_legs = sorted({row["legs"][0] for row in data["rows"] if row["kind"] == "sc"})
    return CorrelatorConfig(
        color_correlations=tuple(
            ColorCorrelator.dipole(f"T{i}_{j}", i, j)
            for i, j in combinations(coloured, 2)
        ),
        spin_correlations=tuple((leg,) for leg in spin_legs),
    )


def _requests(data: dict):
    """One public request per Recola (kind, legs), with per-point spin
    vectors, and every Recola row with its request and conversion factor."""

    from pyamplicol import CorrelatedRequest

    colours = _colours(data)
    count = len(data["points"])
    vectors: dict[tuple, list] = {}
    for row in data["rows"]:
        if row["vector"] is not None:
            batch = vectors.setdefault(
                (row["kind"], tuple(row["legs"])), [None] * count
            )
            batch[row["point"]] = row["vector"]
    requests, answers = {}, []
    for row in data["rows"]:
        kind, legs = row["kind"], tuple(row["legs"])
        label = f"{kind}:{','.join(map(str, legs))}"
        if kind == "born":
            request, factor = CorrelatedRequest("born", {}), 1.0
        elif kind == "cc":
            i, j = sorted(legs)
            request = CorrelatedRequest(f"T{i}_{j}", {})
            factor = CASIMIR[colours[legs[0] - 1]]
        elif kind == "sc":
            request, factor = (
                CorrelatedRequest("born", {legs[0]: vectors[kind, legs]}),
                1.0,
            )
        elif kind == "scc":
            i, j = sorted(legs)
            request = CorrelatedRequest(f"T{i}_{j}", {legs[0]: vectors[kind, legs]})
            factor = CASIMIR[colours[legs[0] - 1]]
        else:
            raise ValueError(f"unknown Recola row kind {kind!r}")
        requests[label] = request
        answers.append((label, row, factor))
    return requests, answers


def _parameters(name: str) -> tuple[dict[str, str], dict[str, str]]:
    scheme = _spec(name)["scheme_data"]
    return dict(scheme["parameters"]), dict(scheme["fixed_parameters"])


def _card(name: str, output) -> str:
    """The public route on the shipped loop_sm: the spec's restriction and
    fixed parameters, QCD corrections, the model's R2, minimal Born orders."""

    scheme = _spec(name)["scheme_data"]
    fixed = ", ".join(
        f"{k} = {json.dumps(v)}" for k, v in scheme["fixed_parameters"].items()
    )
    return f"""action = "generate"
[model]
source = "loop_sm"
restriction = {json.dumps(scheme["restriction"])}
[process]
entries = [{{expression = {json.dumps(_expression(name))}, name = {json.dumps(name)}}}]
loop_coupling_orders = ["QCD"]
coupling_order_policy = "minimal"
[loop]
r2_strategy = "model"
[loop.preparation]
fixed_parameters = {{{fixed}}}
[generation]
output = {json.dumps(str(output))}
emit_api_bundle = false
[generation.validation]
post_build_validation = false
"""


@pytest.fixture(scope="module")
def artifacts(loop_native, loop_reference, tmp_path_factory):
    """Per process: (fixture, one-loop runtime, tree runtime), generated once
    with the fixture's dipoles and spin legs."""

    from pyamplicol import Generator, ModelSource, Runtime
    from pyamplicol.api import ProcessRequest, ProcessSet
    from pyamplicol.config import ColorConfig, RunConfig, resolve_config
    from pyamplicol.config.models import ProcessConfig

    built: dict[str, tuple] = {}

    def build(name: str):
        if name in built:
            return built[name]
        data = _fixture(loop_reference, name)
        root = tmp_path_factory.mktemp(f"corr-{name}")
        parameters, fixed = _parameters(name)
        declarations = _declarations(data)
        card = root / "card.toml"
        card.write_text(_card(name, root / "virtual"))
        Generator(resolve_config(card)).generate(
            ProcessSet((ProcessRequest.parse(_expression(name), name=name),)),
            root / "virtual",
            correlators=declarations,
        )
        virtual = Runtime.load(root / "virtual", model_parameters=parameters)
        config = RunConfig(
            action="generate",
            color=ColorConfig(accuracy="full"),
            process=ProcessConfig(
                coupling_order_policy="explicit",
                max_coupling_orders=dict(data["born_orders"]),
            ),
        )
        Generator(config).generate(
            _expression(name),
            root / "tree",
            model=ModelSource.from_path(
                "loop_sm", restriction=_spec(name)["scheme_data"]["restriction"]
            ),
            correlators=declarations,
        )
        tree = Runtime.load(root / "tree")
        mutable = {p.name for p in tree.physics.model_parameters if p.mutable}
        tree.set_model_parameters(
            {k: float(v) for k, v in {**parameters, **fixed}.items() if k in mutable}
        )
        built[name] = data, virtual, tree
        return built[name]

    return build


def _close(actual: complex, expected: float, bound: float) -> bool:
    return abs(actual - expected) <= bound


@pytest.mark.parametrize("name", PROCESSES)
def test_recola_references_conserve_colour(loop_reference, name):
    """The fixture's own consistency: sum over j != i of C_i cc(i, j) =
    -C_i Born at LO and for every Laurent coefficient of the interference,
    and the IR coefficients are linear in Delta_IR."""

    data = _fixture(loop_reference, name)
    for point in range(len(data["points"])):
        rows = [row for row in data["rows"] if row["point"] == point]
        (born,) = (row for row in rows if row["kind"] == "born")
        dipoles = [row for row in rows if row["kind"] == "cc"]
        for leg in sorted({row["legs"][0] for row in dipoles}):
            mine = [row for row in dipoles if row["legs"][0] == leg]
            assert sum(row["lo"] for row in mine) == pytest.approx(
                -born["lo"], rel=1e-12
            )
            for k in range(3):
                total = sum(row["nlo"][k] for row in mine)
                assert total == pytest.approx(
                    -born["nlo"][k], rel=1e-8, abs=1e-8 * abs(born["nlo"][0])
                )
        for row in rows:
            if row["kind"] != "born":
                assert abs(row["linearity"]) <= 1e-8 * max(map(abs, row["nlo"]))


@pytest.mark.parametrize("name", PROCESSES)
def test_correlated_virtual_matches_recola(artifacts, name):
    """The one-loop runtime's correlated Born and Born-virtual interference
    (finite part, single and double pole) for every dipole, gluon spin
    correlation and spin-colour correlation = Recola, binary64."""

    data, virtual, _ = artifacts(name)
    requests, answers = _requests(data)
    scales = [str(Decimal(mu) ** 2) for mu in data["mu_r"]]
    results = virtual.evaluate_correlated_many(data["points"], requests, mu_r2=scales)
    failures = []
    for label, row, factor in answers:
        point = row["point"]
        result = results[label]
        born = complex(result.born_values[point][0])
        laurent = [complex(value) for value in result.values[point][0]]
        finite, ir1, ir2 = row["nlo"]
        expected = [
            factor * (finite + math.pi**2 / 6 * ir2),
            factor * ir1,
            factor * ir2,
        ]
        scale = max(map(abs, expected))
        lo = factor * row["lo"]
        if not _close(born, lo, BORN_RELATIVE * abs(lo)):
            failures.append((label, point, "born", born, lo))
        for coefficient, (ours, theirs) in zip(
            ("finite", "single", "double"),
            zip(laurent, expected, strict=True),
            strict=True,
        ):
            if not _close(ours, theirs, VIRTUAL_RELATIVE * scale):
                failures.append((label, point, coefficient, ours, theirs))
    assert not failures, failures


@pytest.mark.parametrize("name", PROCESSES)
def test_correlated_tree_born_matches_recola(artifacts, name):
    """The tree engine's correlated Born (``Generator.generate(...,
    correlators=...)``, full colour) for every dipole, gluon spin correlation
    and spin-colour correlation = Recola's LO values."""

    data, _, tree = artifacts(name)
    requests, answers = _requests(data)
    momenta = [
        tuple(tuple(Decimal(x) for x in leg) for leg in point)
        for point in data["points"]
    ]
    results = tree.evaluate_correlated_many(momenta, requests, precision=20)
    failures = []
    for label, row, factor in answers:
        value = results[label][row["point"]]
        ours = complex(float(value.real), float(value.imag))
        lo = factor * row["lo"]
        if not _close(ours, lo, BORN_RELATIVE * abs(lo)):
            failures.append((label, row["point"], ours, lo))
    assert not failures, failures
