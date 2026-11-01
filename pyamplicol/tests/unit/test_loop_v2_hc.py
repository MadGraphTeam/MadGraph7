# SPDX-License-Identifier: 0BSD
"""HC_NLO_X0, the genericity model of M-FINAL (MF-GM-HC; user ruling Q10).

The Higgs Characterisation NLO UFO (v1.3, sha256 and source in
``tools/loop_reference/hc_nlo_x0.json``) needed three generic fixes, each
checked here on a small synthetic model (default tier):

* P1, the process-global ``UFO::`` head is matched at a symbol-head boundary
  (``symbols.UFO_HEAD``), so a model namespace from a UFO directory named
  ``*_UFO`` (``model_HC_NLO_X0_UFO::``) compiles, and the registry's
  re-namespacing keeps such a namespace intact; a genuine leftover head is
  the typed ``ProcessGlobalUFOSymbol`` (a ``ModelError``);
* P2, ``feynman-gauge`` is refused only where the loops need 't Hooft-Feynman
  gauge (``ew_policy.require_loop_gauge``: an electroweak request or a
  massive vector loop species); QCD loops of a unitary-gauge source (HC
  declares ``gauge = [0]``) keep the compiled gauge;
* P4, particle counterterm keys with extra zero order positions (HC's
  ``(1,0,0,0,k)`` with three orders) are read positionally, as MG5_aMC does;
  any other length is the listed source refusal ``ufo-counterterm-keys``.

Nightly, the rows of the card on the downloaded model (FC, QCD corrections,
model R2, the card's HC scheme declaration, every external parameter given
per point, zero widths, MU_R = 125 and 91.188 GeV): binary64 Born and
Laurent coefficients equal MadLoop 3.7.0 (fixtures ``required/hc/``, from
``tools/loop_reference/hc_nlo_x0.py``; one squared order per row) within
1e-5, the poles equal Catani's (at most three coloured partons: the colour
correlators are Casimirs), one rotation check passes, and HC02 (rank N+1
boxes) agrees in double-double.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "tools/loop_reference/hc_nlo_x0.json"
FIXTURES = ROOT / "tests/fixtures/loop/required/hc"
TOLERANCE = 1e-5
POLE_TOLERANCE = 1e-8
ROTATION_BOUND = 1e-6


# ------------------------------------------------------------------- P1


@pytest.mark.loop_fast
def test_ufo_head_is_matched_at_a_symbol_head_boundary():
    """``UFO_HEAD`` matches the ``UFO`` namespace itself, never a namespace
    ending in ``UFO`` nor a nested one; the registry of a model named
    ``*_UFO`` re-namespaces UFO heads and leaves its own namespace intact,
    also on its own canonical output."""

    from pyamplicol._internal.physics.symbols import UFO_HEAD, symbols

    assert UFO_HEAD.search("UFO::aS")
    assert UFO_HEAD.search("2*UFO::{}::MU_R")
    assert UFO_HEAD.search("f(UFO::x,1)")
    assert not UFO_HEAD.search("model_HC_NLO_X0_UFO::{}::MU_R")
    assert not UFO_HEAD.search("model_X_UFO::MU_R + other::UFO::y")
    registry = symbols.model("HC_NLO_X0_UFO")
    assert registry.namespace == "model_HC_NLO_X0_UFO"
    assert registry.expression("UFO::MU_R + UFO::{}::aS") == registry.symbol(
        "MU_R"
    ) + registry.symbol("aS")
    text = registry.expression_string("UFO::MU_R*UFO::aS")
    assert registry.expression(text) == registry.symbol("MU_R") * registry.symbol("aS")


@pytest.mark.loop_fast
def test_compiled_model_refuses_a_ufo_head_typed_and_admits_a_ufo_named_namespace():
    from pyamplicol.api.errors import ModelError
    from pyamplicol.models.contracts import (
        CompiledCouplingRecord,
        CompiledModelIR,
        ProcessGlobalUFOSymbol,
    )

    def model(expression: str) -> CompiledModelIR:
        return CompiledModelIR(
            name="synthetic_UFO",
            orders=(),
            parameters=(),
            particles=(),
            couplings=(
                CompiledCouplingRecord(
                    name="GC_1",
                    expression=expression,
                    resolved_expression=expression,
                    value=(1.0, 0.0),
                    orders=(),
                ),
            ),
            propagators=(),
            vertex_terms=(),
            oriented_kernels=(),
            direct_contractions=(),
            closure_contractions=(),
        )

    with pytest.raises(ProcessGlobalUFOSymbol, match="process-global UFO") as error:
        model("2*UFO::aS")
    assert isinstance(error.value, ModelError) and isinstance(error.value, ValueError)
    assert error.value.context.startswith("coupling GC_1")
    model("2*model_synthetic_UFO::{}::aS")


@pytest.mark.loop_fast
def test_a_model_directory_named_ufo_compiles(tmp_path):
    """The packaged scalars model copied to ``scalars_UFO``: its namespace
    ``model_scalars_UFO`` reaches the executable expressions (the former
    substring test refused every such model) and the compile succeeds."""

    import pyamplicol
    from pyamplicol.models.loading import compile_model_source

    packaged = Path(pyamplicol.__file__).parent / "assets/models/ufo/scalars"
    source = tmp_path / "scalars_UFO"
    shutil.copytree(packaged, source, ignore=shutil.ignore_patterns("__pycache__"))
    compiled = compile_model_source(source, cache_dir=tmp_path / "cache")
    expressions = [text for _context, text in compiled.ir._executable_expressions()]
    assert any("model_scalars_UFO::" in text for text in expressions)
    assert any("UFO::" in text for text in expressions)


# ------------------------------------------------------------------- P2


def _stub_model(gauge: str, declared, particles):
    record = SimpleNamespace(
        metadata={} if declared is None else {"gauge": tuple(declared)}
    )
    return SimpleNamespace(
        name="stub",
        ir=SimpleNamespace(
            gauge=gauge,
            particles=tuple(particles),
            parameters=(SimpleNamespace(name="MV", nature="external"),),
        ),
        raw_loop_model=record,
        prepared_bundle=None,
    )


def _particle(name, pdg, spin, color, mass, ghost=0):
    return SimpleNamespace(
        name=name,
        pdg_code=pdg,
        spin=spin,
        color=color,
        mass=mass,
        ghost_number=ghost,
        auxiliary_kind=None,
    )


# A gluon, a light quark, a massive colour-octet vector and its ghost.
PARTICLES = (
    _particle("g", 21, 3, 8, "ZERO"),
    _particle("q", 1, 2, 3, "ZERO"),
    _particle("V8", 9000005, 3, 8, "MV"),
    _particle("ghV8", 9000006, -1, 8, "MV", ghost=1),
)
SPECIES = {21: {"QCD"}, 1: {"QCD", "QED"}, 9000005: {"QCD"}, 9000006: {"QCD"}}


def _request(veto=(), corrections=("QCD",)):
    return SimpleNamespace(
        process="q q~ > g g",
        corrections=corrections,
        particle_veto=tuple(veto),
        fixed_parameters={},
    )


@pytest.mark.loop_fast
def test_massive_vector_loop_species_are_the_unvetoed_massive_vectors():
    from pyamplicol.loops.v2 import ew_policy

    model = _stub_model("unitary", (0,), PARTICLES)
    assert ew_policy.massive_vector_loop_species(model, _request(), SPECIES) == ("V8",)
    assert (
        ew_policy.massive_vector_loop_species(model, _request([9000005]), SPECIES) == ()
    )
    assert (
        ew_policy.massive_vector_loop_species(
            model, _request(corrections=("QED",)), SPECIES
        )
        == ()
    )


@pytest.mark.loop_fast
def test_loop_gauge_is_refused_only_where_massive_vectors_run_in_the_loops(
    monkeypatch,
):
    """On a source declaring only unitary gauge: loops without massive
    vectors are admitted in the compiled gauge; a massive vector loop
    species or an electroweak request is the listed ``feynman-gauge``; a
    model compiled in 't Hooft-Feynman gauge takes either."""

    from pyamplicol.loops.v2 import ew_policy
    from pyamplicol.loops.v2.refusals import LoopRequestUnsupported

    monkeypatch.setattr(ew_policy, "electroweak", lambda *_args: False)
    unitary = _stub_model("unitary", (0,), PARTICLES)
    ew_policy.require_loop_gauge(unitary, _request([9000005]), SPECIES)
    with pytest.raises(LoopRequestUnsupported, match=r"\['V8'\]") as error:
        ew_policy.require_loop_gauge(unitary, _request(), SPECIES)
    assert error.value.key == "feynman-gauge"
    assert isinstance(error.value, ValueError)
    # Declaring gauge 1 is not enough while the model is compiled unitary.
    with pytest.raises(LoopRequestUnsupported, match="compiled in unitary") as error:
        ew_policy.require_loop_gauge(
            _stub_model("unitary", (0, 1), PARTICLES), _request(), SPECIES
        )
    assert error.value.key == "feynman-gauge"
    ew_policy.require_loop_gauge(
        _stub_model("feynman", (0, 1), PARTICLES), _request(), SPECIES
    )
    monkeypatch.setattr(ew_policy, "electroweak", lambda *_args: True)
    with pytest.raises(LoopRequestUnsupported, match="does not declare Feynman") as e:
        ew_policy.require_loop_gauge(unitary, _request([9000005]), SPECIES)
    assert e.value.key == "feynman-gauge"


# ------------------------------------------------------------------- P4


@pytest.fixture(scope="module")
def hepkit():
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("the synthetic loop model needs licensed Symbolica")
    return pytest.importorskip("symbolica.community.hepkit")


def _field_rules(hepkit, counterterm):
    """A synthetic NLO UFO (the HepKit QCD+QED particles, two coupling
    orders) whose up quark carries ``counterterm``."""

    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.loops.wavefunction import prepare_external_field_rules
    from pyamplicol.models.loop_import import capture_raw_loop_model

    module = ModuleType("hc_counterterm_keys")
    module.all_particles = [
        SimpleNamespace(**record)
        for record in json.loads(hepkit.Model.qcd_qed().to_json())["particles"]
    ]
    module.all_orders = [
        SimpleNamespace(name=name, perturbative_expansion=1) for name in ("QCD", "QED")
    ]
    module.all_CTparameters = [SimpleNamespace(name="dZ", value={-1: "a", 0: "b"})]
    up = next(p for p in module.all_particles if p.pdg_code == 2)
    up.loop_particles = [[[2, 21]], [[2, 22]]]
    up.counterterm = counterterm
    return prepare_external_field_rules(
        build_loop_model(capture_raw_loop_model(module))
    )


@pytest.mark.loop_fast
def test_counterterm_keys_with_extra_zero_positions_are_read_positionally(hepkit):
    exact = _field_rules(hepkit, {(1, 0, 0): {-1: "dZ"}, (0, 1, 1): {0: "dZ"}})
    padded = _field_rules(
        hepkit, {(1, 0, 0, 0, 0): {-1: "dZ"}, (0, 1, 0, 0, 1): {0: "dZ"}}
    )
    assert [rule.order for rule in padded] == [(("QCD", 2),), (("QED", 2),)]
    assert [(rule.order, rule.alternatives, rule.coefficient) for rule in padded] == [
        (rule.order, rule.alternatives, rule.coefficient) for rule in exact
    ]


@pytest.mark.loop_fast
@pytest.mark.parametrize("key", [(1, 0, 1, 0), (0, 0, 1, 0), (1, 0)])
def test_other_counterterm_key_lengths_are_the_listed_source_refusal(hepkit, key):
    from pyamplicol.api.errors import ModelError
    from pyamplicol.models.source_refusals import (
        ModelSourceUnsupported,
        listed_model_source_refusal_keys,
    )

    assert "ufo-counterterm-keys" in listed_model_source_refusal_keys()
    with pytest.raises(ModelSourceUnsupported, match="ufo-counterterm-keys") as error:
        _field_rules(hepkit, {key: {0: "dZ"}})
    assert error.value.key == "ufo-counterterm-keys"
    assert isinstance(error.value, ModelError)


# ------------------------------------------------------------------ rows

NC = 3.0
CA, CF, TR = NC, (NC * NC - 1) / (2 * NC), 0.5
# Massless quark flavours in the loops of every row (u d s c b; the top is
# vetoed, or no gluon is external).
LIGHT_FLAVOURS = 5
CASIMIR = {"g": CA, "q": CF}
GAMMA = {"g": 11 / 6 * CA - 2 / 3 * TR * LIGHT_FLAVOURS, "q": 1.5 * CF}
# Coloured external particles of the rows by process token.
PARTON = {"g": "g", "u": "q", "u~": "q"}


def _card() -> dict:
    return json.loads(CARD.read_text())


def _row(card: dict, row_id: str) -> dict:
    (row,) = [row for row in card["rows"] if row["id"] == row_id]
    return row


def _loop_card(source: Path, card: dict, row: dict, output: Path) -> str:
    model = card["loop_model"]
    scheme = "\n".join(
        f"{key} = {json.dumps(value)}" for key, value in model["scheme"].items()
    )
    fixed = ", ".join(
        f"{name} = {json.dumps(value)}"
        for name, value in card["fixed_parameters"].items()
    )
    veto = (
        f"particle_veto = {json.dumps(row['particle_veto'])}"
        if row["particle_veto"]
        else ""
    )
    return f"""
action = "generate"
[model]
source = {json.dumps(str(source))}
restriction = {json.dumps(row["restriction"])}
[process]
entries = [{{expression = {json.dumps(row["process"])}, name = "p"}}]
loop_coupling_orders = {json.dumps(model["loop_coupling_orders"])}
coupling_order_policy = "minimal"
[color]
accuracy = "full"
[loop]
r2_strategy = "model"
[loop.preparation]
fixed_parameters = {{{fixed}}}
[loop.model]
loop_coupling_orders = {json.dumps(model["loop_coupling_orders"])}
uv_complete = {json.dumps(model["uv_complete"])}
model_r2_complete = {json.dumps(model["model_r2_complete"])}
[loop.model.scheme]
{scheme}
[generation]
output = {json.dumps(str(output))}
emit_api_bundle = false
{veto}
[generation.validation]
post_build_validation = false
"""


def _generate(source: Path, card: dict, row: dict, tmp_path: Path):
    from pyamplicol import Runtime
    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    output = tmp_path / row["id"]
    path = tmp_path / f"{row['id']}.toml"
    path.write_text(_loop_card(source, card, row, output))
    Generator(resolve_config(path)).generate(
        ProcessSet((ProcessRequest.parse(row["process"], name="p"),)), output
    )
    return Runtime.load(output, process="p")


def _dot(p, q) -> float:
    return p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3]


def _catani(process: str, momenta, born: float, mu: float, alpha_s: float):
    """Catani's (single, double) poles of the renormalised virtual of a
    process with at most three coloured massless partons, whose colour
    correlators are Casimirs: T_i.T_j = (C_k - C_i - C_j)/2, or -C_i for
    two (normalisation (4 pi)^eps / Gamma(1 - eps), as MadLoop's)."""

    initial, final = process.split(">")
    tokens = initial.split() + final.split()
    partons = [(PARTON[t], momenta[k]) for k, t in enumerate(tokens) if t in PARTON]
    assert 2 <= len(partons) <= 3, process
    casimirs = [CASIMIR[kind] for kind, _ in partons]
    factor = alpha_s / (2 * math.pi)
    double = -factor * sum(casimirs) * born
    single = -factor * sum(GAMMA[kind] for kind, _ in partons) * born
    for i, (_, p) in enumerate(partons):
        for j, (_, q) in enumerate(partons):
            if i == j:
                continue
            if len(partons) == 2:
                correlator = -casimirs[i]
            else:
                (k,) = {0, 1, 2} - {i, j}
                correlator = (casimirs[k] - casimirs[i] - casimirs[j]) / 2
            log = math.log(mu * mu / abs(2 * _dot(p, q)))
            single += factor * correlator * log * born
    return single, double


def _relative(ours: float, theirs: float, floor: float) -> float:
    return abs(ours - theirs) / max(abs(ours), abs(theirs), floor)


def _check_row(runtime, card: dict, row: dict, *, double_double: bool = False):
    from pyamplicol.api.virtual_results import VirtualStabilityChecks

    fixture = json.loads((FIXTURES / f"{row['id'].lower()}.json").read_text())
    assert fixture["points"] == row["points"] and fixture["mu_r"] == card["mu_r"]
    for entry in fixture["cards"]:
        assert entry["parameters"] == {
            **card["restrictions"][row["restriction"]],
            **entry["parameters"],
            **card["fixed_parameters"],
        }
        runtime.set_model_parameters(
            {
                name: value
                for name, value in entry["parameters"].items()
                if name not in card["fixed_parameters"]
            }
        )
        alpha_s = float(entry["parameters"]["aS"])
        points = [
            [[float(x) for x in leg] for leg in fixture["points"][value["point"]]]
            for value in entry["values"]
        ]
        scales = [float(value["mu_r"]) for value in entry["values"]]
        result = runtime.evaluate(
            points,
            mu_r2=[mu * mu for mu in scales],
            stability_checks=VirtualStabilityChecks(1),
        )
        assert len(result.born_orders) == 1 and len(result.sectors) == 1
        for k, value in enumerate(entry["values"]):
            born = float(result.born_values[k][0])
            finite, single, double = (float(x) for x in result.values[k][0])
            label = (row["id"], entry["parameters"]["cosa"], k)
            assert _relative(born, value["born"], 0.0) <= TOLERANCE, label
            floor = abs(value["born"]) * alpha_s / (2 * math.pi)
            for ours, key in zip(
                (finite, single, double),
                ("finite", "single_pole", "double_pole"),
                strict=True,
            ):
                theirs = value[key]
                assert _relative(ours, theirs, floor) <= TOLERANCE, (label, key)
            catani = _catani(row["process"], points[k], born, scales[k], alpha_s)
            for ours, theirs in zip((single, double), catani, strict=True):
                assert _relative(ours, theirs, floor) <= POLE_TOLERANCE, label
            estimate = result.stability[k]
            assert [name for name, _ in estimate.checks] == ["rotation-0"]
            assert estimate.relative <= ROTATION_BOUND, (label, estimate)
        if double_double:
            extended = runtime.evaluate(
                points, mu_r2=[mu * mu for mu in scales], arithmetic="double-double"
            )
            for k, value in enumerate(entry["values"]):
                floor = abs(value["born"]) * alpha_s / (2 * math.pi)
                for i, key in enumerate(("finite", "single_pole", "double_pole")):
                    ours = float(extended.values[k][0][i])
                    binary64 = float(result.values[k][0][i])
                    assert _relative(ours, value[key], floor) <= TOLERANCE, (k, key)
                    assert _relative(ours, binary64, floor) <= ROTATION_BOUND, (k, key)


@pytest.mark.loop_nightly
@pytest.mark.parametrize("row_id", ["HC01", "HC02", "HC04"])
def test_hc_rows_equal_madloop(hc_nlo_x0_ufo, loop_native, tmp_path, row_id):
    card = _card()
    row = _row(card, row_id)
    runtime = _generate(hc_nlo_x0_ufo, card, row, tmp_path)
    _check_row(runtime, card, row, double_double=row_id == "HC02")
