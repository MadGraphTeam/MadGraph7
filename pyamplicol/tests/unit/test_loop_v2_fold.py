# SPDX-License-Identifier: 0BSD
"""F1 proportional-chain folding of loop-induced dressed-row plans (M-LIT
lit-fold; USER 2026-11-08 ~22:39 "Yes, and allow exact zeros"; m-lit
DECISIONS 2026-11-08 22:39; ``loops/v2/fold``).

Checked here:

* the option ``generation.loop_induced_folding`` (on by default; card, CLI
  and Python config; folding off writes the unfolded plans);
* g g > h g: the exact test certifies its three quark-loop orientation
  pairs across class tries (key vectors ``U_t - U_t'``) and its three
  within one class (the representative's weight doubled): 18 chains to 9,
  the recorded failure bound equal to its derivation, the values equal to
  the unfolded plan's within rounding; the fold is deterministic (same
  seed, same plan bytes) and route R1 (the oracle) stays unfolded;
* g g > z g (every chain carries the Z on the loop: vector and axial
  couplings) folds nothing and keeps its plan bytes;
* ratios through a free irrational base, a model parameter or i never fold
  (constructed through the certification's inputs: a fake probe whose
  second chain is the first times such a value);
* exact zeros (the ruling relaxes Q13 for zeros the exact algebra proves):
  g g > a a a and g g > h a fold to no chain at all; the public route
  stores the certificate, serves exactly 0 with it in the result detail and
  the inspect record, and the folding-off artifact keeps the numerical zero
  (``test_loop_v2_induced.py``);
* the dependent reduction lanes (F1d): a folded family's column that is an
  exact multiple ``c`` of another is not reduced; its raw lanes are ``c``
  times the representative's (bitwise for ``c = -1``) and the evaluation
  reduces fewer lanes.
"""

from __future__ import annotations

import json
import random
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from .test_loop_v2_induced import _card, _fixture, _Rows

FOLDING = "generation.loop_induced_folding"
MESSAGE = r"^generation\.loop_induced_folding must be true or false$"
# Rounding of the folded vs the unfolded dressing (one weighted dressing
# instead of two summed), in the colour-metric norm of the amplitude
# vectors relative to the unfolded one (binary64); as route R1 vs R2.
ROUNDING = 1e-12


@pytest.fixture(scope="module")
def li(loop_sm_ufo, loop_native, tmp_path_factory):
    return _Rows(loop_sm_ufo, loop_native, tmp_path_factory.mktemp("li-fold"))


def _norm(vectors, gram) -> float:
    total = 0.0
    for vector in vectors.values():
        for k in range(len(vector[0])):
            a = [complex(vector[t][k]) for t in range(len(vector))]
            total += sum(
                (a[t].conjugate() * complex(gram[t][u]) * a[u]).real
                for t in range(len(a))
                for u in range(len(a))
            )
    return max(total, 0.0) ** 0.5


def _bare(executor, li, name):
    from pyamplicol.loops.v2.assemble import Precision

    point, scale, parameters = li.point(name)
    _precision, (evaluation,), failures = executor._evaluated(
        [point], [scale], parameters, Precision.double(), "all", None
    )
    assert not failures
    return evaluation.native_bare


def _executor(slice_, folded):
    from pyamplicol.loops.v2.plan_export import NativeExecutor

    slice_.fold = folded
    try:
        return NativeExecutor(slice_, forward=False)
    finally:
        slice_.fold = None


# ------------------------------------------------------------------ option


@pytest.mark.loop_fast
def test_folding_is_on_by_default_and_set_in_config_cli_and_python(tmp_path):
    """``generation.loop_induced_folding``: on by default, switched off by
    the run config, the CLI (``--no-loop-induced-folding``) and the Python
    config; any non-boolean value is refused with the one message."""

    from pyamplicol.cli import parse_cli
    from pyamplicol.config import ConfigurationError, GenerationConfig, resolve_config

    fixture = _fixture("li_gg_hg")
    card = tmp_path / "card.toml"
    card.write_text(_card(tmp_path / "out", (("p", "g g > h g"),), fixture))
    assert resolve_config(card).effective.generation.loop_induced_folding is True
    off = tmp_path / "off.toml"
    off.write_text(
        card.read_text().replace(
            "[generation]\n", "[generation]\nloop_induced_folding = false\n"
        )
    )
    assert resolve_config(off).effective.generation.loop_induced_folding is False
    bad = tmp_path / "bad.toml"
    bad.write_text(
        off.read_text().replace(
            "loop_induced_folding = false", 'loop_induced_folding = "no"'
        )
    )
    with pytest.raises(ConfigurationError, match=MESSAGE):
        resolve_config(bad)
    cli = parse_cli(("generate",)).resolve().effective.generation
    assert cli.loop_induced_folding is True
    flag = parse_cli(("generate", "--no-loop-induced-folding")).resolve().effective
    assert flag.generation.loop_induced_folding is False
    assert GenerationConfig(loop_induced_folding=False).loop_induced_folding is False
    with pytest.raises(ConfigurationError, match=MESSAGE):
        GenerationConfig(loop_induced_folding="false")


# ------------------------------------------------------------ the fold


@pytest.mark.loop_physics
def test_g_g_to_h_g_folds_its_orientation_pairs(li):
    """g g > h g (three gluons on the top loop): every chain folds with its
    reversed orientation, 18 chains to 9; the box pairs sit in different
    class tries (key vector U_t - U_t', ratio -1), the triangle pairs with a
    two-gluon block in one (the representative's weight doubled). The
    failure bound equals its derivation, the values the unfolded ones
    within rounding, and route R1 stays unfolded."""

    from pyamplicol.loops.v2 import fold
    from pyamplicol.loops.v2 import helicity_filter as hf
    from pyamplicol.loops.v2.plan_export import NativeExecutor

    name = "li_gg_hg"
    slice_ = li.slice(name)
    folded = fold.fold_slice(slice_)
    verdict = folded.verdict
    assert (verdict.chains, verdict.chains_after) == (18, 9)
    assert verdict.groups == 9 and verdict.chains_folded == 18
    assert verdict.identities == 9 and verdict.failed == 0
    assert verdict.zero_groups == 0 and not verdict.exact_zero
    k = hf.points_needed(verdict.degree_bound, 9, verdict.failure_probability)
    assert verdict.points == k
    assert verdict.failure_bound == 9 * (verdict.degree_bound / (hf.PRIME - 1)) ** k
    assert verdict.failure_bound <= verdict.failure_probability
    record = verdict.record()
    assert record["prime"] == hf.PRIME and record["seed"] == fold.SEED
    # Across class tries: a seed key of two classes with weights 1 and -1.
    merged = [
        key
        for tries in folded.families.values()
        for (combination, _orders), _chains in tries
        for key in [combination]
        if len(combination) == 2
    ]
    assert len(merged) == 3
    for combination in merged:
        assert sorted(x.real for _c, x in combination) == [-1, 1]
    # Within one class: the representative's weight is the pair's sum.
    doubled = [
        weight
        for tries in folded.families.values()
        for (combination, _orders), chains in tries
        if len(combination) == 1
        for _skeleton, _ring, weight in chains
    ]
    assert len(doubled) == 6 and set(doubled) == {Fraction(2)}
    whole = _executor(slice_, None)
    once = _executor(slice_, folded)
    assert len(once.exported.stored()) < len(whole.exported.stored())
    a, b = _bare(whole, li, name), _bare(once, li, name)
    gram = slice_.amplitude.metric_at_three()
    difference = {
        key: [
            [complex(x) - complex(y) for x, y in zip(ra, rb, strict=True)]
            for ra, rb in zip(a[key], b[key], strict=True)
        ]
        for key in a
    }
    assert _norm(difference, gram) <= ROUNDING * _norm(a, gram)
    # Route R1 (the oracle) is never folded.
    slice_.fold = folded
    try:
        r1 = NativeExecutor(slice_, forward=False, route="R1")
    finally:
        slice_.fold = None
    unfolded_r1 = NativeExecutor(slice_, forward=False, route="R1")
    assert r1.exported.stored() == unfolded_r1.exported.stored()


@pytest.mark.loop_physics
def test_the_fold_is_deterministic_and_off_keeps_the_plan(li):
    """The same seed gives the same folded plan bytes; without a fold the
    plan is the slice's own, byte for byte (``LoopSlice.fold`` None)."""

    from pyamplicol.loops.v2 import fold
    from pyamplicol.loops.v2.plan_export import NativeExecutor

    slice_ = li.slice("li_gg_hg")
    first = _executor(slice_, fold.fold_slice(slice_)).exported.stored()
    second = _executor(slice_, fold.fold_slice(slice_)).exported.stored()
    assert first == second
    assert slice_.fold is None
    assert (
        _executor(slice_, None).exported.stored()
        == NativeExecutor(slice_, forward=False).exported.stored()
    )


@pytest.mark.loop_physics
def test_g_g_to_z_g_folds_nothing(li):
    """g g > z g: every chain carries the Z on the quark loop (vector and
    axial couplings): no orientation pair is proportional, nothing folds
    and the plan keeps its bytes."""

    from pyamplicol.loops.v2 import fold

    slice_ = li.slice("li_gg_zg")
    folded = fold.fold_slice(slice_)
    assert folded.families == {}
    assert folded.verdict.groups == 0 and folded.verdict.chains_folded == 0
    assert folded.verdict.chains_after == folded.verdict.chains
    assert (
        _executor(slice_, folded).exported.stored()
        == _executor(slice_, None).exported.stored()
    )


# --------------------------------------------- ratios that never fold


class _Probe:
    """A fake probe executor of one family with two chains of one row and
    one coefficient: chain 0 is 1, chain 1 is ``value(inputs)`` (a base's
    value, as the certification's inputs give it)."""

    def __init__(self, key, value):
        self.value = value
        self.exported = SimpleNamespace(
            rows=((),),
            families=("f",),
            orders=(("QCD", 1),),
            block_rows=(),
            masses=(),
            constants=((key, ()),),
        )
        self.slice = SimpleNamespace(
            blocks=SimpleNamespace(process=SimpleNamespace(legs=[None, None]))
        )

    def native(self, *_args):
        return self

    def numerators(self, inputs, _schedule):
        re, im = self.value(inputs)
        return [[(0, 0, 0, [[("1", "0")], [(str(re), str(im))]])]]


def _chains():
    from pyamplicol.loops.v2.fold import Chain

    return {
        "f": [
            Chain(None, (), (), ("QCD", 1), Fraction(1), ()),
            Chain(None, (), (), ("QCD", 1), Fraction(1), ()),
        ]
    }


@pytest.mark.loop_fast
@pytest.mark.parametrize(
    ("factors", "imaginary"),
    [
        # A prime under a square root (sqrt 2: a square modulo 2^61 - 1).
        ((), ((("prime", 2), "1/2"),)),
        # A model parameter.
        (((("param", "MT"), "1"),), ()),
        # The imaginary unit (a ratio outside F_p).
        (None, None),
    ],
)
def test_irrational_parameter_and_complex_ratios_never_fold(factors, imaginary):
    """The fold's test takes every base free (``fold._FreeValues``): a
    second chain equal to the first times sqrt(2), a parameter or i is no
    fold, though sqrt(2) has a root modulo p (the filter's own values would
    give it a constant residue)."""

    from pyamplicol.loops.v2 import fold
    from pyamplicol.loops.v2 import helicity_filter as hf

    if factors is None:
        key = ((0, 1), (), (), True)
        constant = None
    else:
        key = ((1, 0), factors, imaginary, True)
        constant = 0
    probe = _Probe(
        key,
        (lambda inputs: (0, 1))
        if constant is None
        else (lambda inputs: tuple(int(x) for x in inputs["constants"][constant])),
    )
    batch = fold._Batch(_chains())
    batch.executor = probe
    batch.lcm = hf._bases(probe.exported.constants)
    rng = random.Random(0)
    inputs = hf.random_inputs(probe, rng, batch.lcm, values=fold._FreeValues)
    (point,) = probe.numerators(inputs, "B")
    fold._discover(batch, point, rng)
    if factors is None:
        # A complex ratio forms no candidate group.
        assert batch.groups == {}
        return
    (group,) = batch.groups["f"]
    (_rep, (j, residue)) = group
    identities = [("f", j, 0, Fraction(residue))]
    assert fold._certify(batch, identities, 2, rng) == {0}
    if imaginary:
        # The filter's values: sqrt(2)'s own root, the same at every point
        # (a constant residue a free base never has).
        lcm = hf._bases(probe.exported.constants)
        first = hf.random_inputs(probe, random.Random(1), lcm)["constants"][0]
        second = hf.random_inputs(probe, random.Random(2), lcm)["constants"][0]
        assert first == second


# ------------------------------------------------------------ exact zeros


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["li_gg_aaa", "li_gg_ha"])
def test_exact_zeros_fold_to_no_chain(li, name):
    """USER ~22:39 "allow exact zeros": every chain of g g > a a a and
    g g > h a folds into a group whose weights cancel exactly (the quark
    loop's two orientations, certified by the exact test, never by Furry's
    theorem): no family is left, the process is an exact zero with its
    certificate, and its native amplitude is exactly 0."""

    from pyamplicol.loops.v2 import fold

    slice_ = li.slice(name)
    folded = fold.fold_slice(slice_)
    verdict = folded.verdict
    assert verdict.exact_zero and verdict.chains_after == 0
    assert verdict.zero_groups == verdict.groups > 0
    assert verdict.failure_bound <= verdict.failure_probability
    executor = _executor(slice_, folded)
    assert executor.exported.families == ()
    for vector in _bare(executor, li, name).values():
        assert all(complex(z) == 0 for laurent in vector for z in laurent)


def _generate(root: Path, name: str, expression: str, folding: bool) -> Path:
    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config

    root.mkdir(parents=True, exist_ok=True)
    card = root / "card.toml"
    text = _card(root / "artifact", ((name, expression),), _fixture(name))
    if not folding:
        text = text.replace(
            "[generation]\n", "[generation]\nloop_induced_folding = false\n"
        )
    card.write_text(text)
    Generator(resolve_config(card)).generate(
        ProcessSet((ProcessRequest.parse(expression, name=name),)), root / "artifact"
    )
    return root / "artifact"


def _filters(artifact: Path, name: str) -> dict:
    manifest = json.loads((artifact / "artifact.json").read_text())
    return {
        record["id"]: record["filters"]
        for record in manifest["extensions"]["generation"]["concrete_processes"]
    }[name]


@pytest.mark.loop_physics
def test_public_exact_fold_zero_and_the_folding_off_artifact(li, tmp_path):
    """g g > a a a through the public route: folded (the default), the
    artifact records the fold with its certificate, the filter is skipped,
    and the runtime serves exactly 0 with the certificate in the result
    detail and the inspect record; with folding off the artifact records no
    fold and its value is the numerical zero within its absolute estimate
    (``test_loop_v2_induced.py``)."""

    from pyamplicol import Runtime

    name = "li_gg_aaa"
    point, scale, parameters = li.point(name)
    folded = _generate(tmp_path / "on", name, "g g > a a a", True)
    filters = _filters(folded, name)
    record = filters["loop_induced_fold"]
    assert record["enabled"] and record["exact_zero"]
    assert record["chains_after"] == 0 and record["zero_groups"] == record["groups"]
    assert filters["loop_induced_helicity_filter"] == {"exact_fold_zero": True}
    runtime = Runtime.load(folded, process=name, model_parameters=parameters)
    info = runtime.inspect()
    assert info["exact_fold_zero"] is True and info["exact_colour_zero"] is False
    assert info["fold_certificate"]["failure_bound"] == record["failure_bound"]
    assert info["fold_zero_loop_orders"] == [{"QCD": 2, "QED": 3}]
    result = runtime.evaluate([point], mu_r2=float(scale), poles="none")
    assert result.values[0][0].finite == 0.0
    detail = result.diagnostics[0].detail
    assert detail.startswith("exact fold zero") and "failure bound" in detail
    unfolded = _generate(tmp_path / "off", name, "g g > a a a", False)
    assert "loop_induced_fold" not in _filters(unfolded, name)
    other = Runtime.load(unfolded, process=name, model_parameters=parameters)
    assert "exact_fold_zero" not in other.inspect()


# ---------------------------------------------------- dependent lanes


@pytest.mark.loop_physics
def test_dependent_columns_are_formed_after_the_reduction(li):
    """F1d: a folded family's column that is an exact multiple c of
    another (g g > h g's box pairs: U_t - U_t', c = -1) is not reduced: the
    evaluation reduces fewer lanes, and its raw lanes are c times its
    representative's (bitwise for c = -1)."""

    from pyamplicol.loops.v2 import fold

    name = "li_gg_hg"
    slice_ = li.slice(name)
    point, scale, parameters = li.point(name)
    counts = {}
    raw = {}
    for label, folded in (("off", None), ("on", fold.fold_slice(slice_))):
        executor = _executor(slice_, folded)
        executor.raw_lanes = True
        (result,) = executor.evaluate([point], [scale], parameters=parameters)
        counts[label] = executor.last_report["counts"]["reduction lanes"]
        raw[label] = {
            (key.family, key.order, key.column)
            if hasattr(key, "family")
            else key: value
            for key, value in result.families
        }
    assert counts["on"] < counts["off"]
    assert len(raw["on"]) == len(raw["off"])
    size = len(slice_.amplitude)
    pairs = 0
    for (family, order, column), value in raw["on"].items():
        partner = (family, order, column - column % size + (1 - column % size))
        other = raw["on"].get(partner)
        if other is not None and all(
            complex(x) == -complex(y) for x, y in zip(value, other, strict=True)
        ):
            pairs += 1
    assert pairs > 0


# ------------------------------------------------- consumers on folds


def _declared(root: Path, name: str, expression: str, folding: bool) -> Path:
    """``name``'s artifact with every dipole and gluon spin class declared
    (``loop_v2_induced_observables_rows.generate``'s declaration), folded or
    with folding off."""

    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config
    from pyamplicol.correlators import CorrelatorConfig

    from .loop_v2_induced_observables_rows import coloured, dipoles, gluons

    fixture = _fixture(name)
    pdgs = fixture["pdgs"]
    root.mkdir(parents=True, exist_ok=True)
    text = _card(root / "artifact", (("p", expression),), fixture)
    if not folding:
        text = text.replace(
            "[generation]\n", "[generation]\nloop_induced_folding = false\n"
        )
    (root / "card.toml").write_text(text)
    correlators = CorrelatorConfig.from_json_dict(
        {
            "color_correlations": dipoles(coloured(pdgs)),
            "spin_correlations": [[leg] for leg in gluons(pdgs)],
        }
    )
    Generator(resolve_config(root / "card.toml")).generate(
        ProcessSet((ProcessRequest.parse(expression, name="p"),)),
        root / "artifact",
        correlators=correlators,
    )
    return root / "artifact"


@pytest.fixture(scope="module")
def hgg(loop_sm_ufo, loop_native, tmp_path_factory):
    """g g > h g g (168 chains fold to 84) with its observables declared,
    folded (the default) and with folding off."""

    root = tmp_path_factory.mktemp("fold-hgg")
    return {
        folding: _declared(root / str(folding), "li_gg_hgg", "g g > h g g", folding)
        for folding in (True, False)
    }


def _points(name: str, count: int = 3):
    from decimal import Decimal

    fixture = _fixture(name)
    points = [
        [[float(Decimal(c)) for c in p] for p in point]
        for point in fixture["points"][:count]
    ]
    return points, [float(Decimal(mu)) ** 2 for mu in fixture["mu_r"][:count]]


def _checks(cross: bool = False):
    from pyamplicol.api.virtual_results import VirtualStabilityChecks
    from pyamplicol.loops.v2.stability import RECOMMENDED_CHECKS

    return VirtualStabilityChecks(
        RECOMMENDED_CHECKS.rotations,
        reduction_direction=True,
        rescaled=True,
        cross_backend=cross,
    )


@pytest.mark.loop_physics
def test_folded_outputs_agree_with_the_unfolded_ones_within_their_estimates(hgg):
    """F1f (DECISIONS 2026-11-08 22:39): on the folded g g > h g g the
    summed value, the helicity-resolved rows, the amplitude-basis anchors and
    the colour correlations equal the folding-off artifact's within the two
    calls' own estimates (recommended checks: they bound the
    folded-vs-unfolded difference); the resolved rows sum to the summed
    call and a complete helicity selection to the ordinary value (Q39)."""

    from pyamplicol import Runtime
    from pyamplicol.correlators import CorrelatedRequest

    runtimes = {f: Runtime.load(hgg[f], process="p") for f in (True, False)}
    points, scales = _points("li_gg_hgg")
    options = {"mu_r2": scales, "on_failure": "return", "stability_checks": _checks()}
    found = {f: {} for f in runtimes}
    for f, runtime in runtimes.items():
        summed = runtime.evaluate(points, **options)
        found[f]["summed"] = [
            [
                (complex(v.finite), e.absolute[s].finite)
                for s, v in enumerate(summed.values[p])
            ]
            for p, e in enumerate(summed.stability)
        ]
        for name, extra in (
            ("rows", {}),
            ("anchors", {"color_resolution": "amplitude-basis"}),
        ):
            resolved = runtime.evaluate_resolved(points, **options, **extra)
            found[f][name] = [
                [
                    (complex(value.finite), estimate[h][c].absolute[s].finite)
                    for s, sector in enumerate(resolved.values[p])
                    for h, row in enumerate(sector)
                    for c, value in enumerate(row)
                ]
                for p, estimate in enumerate(resolved.stability)
            ]
        ids = [item.id for item in runtime.available_color_correlations()][:4]
        many = runtime.evaluate_correlated_many(
            points, {i: CorrelatedRequest(i) for i in ids}, **options
        )
        for key, result in many.items():
            found[f][key] = [
                [(complex(result.values[p][0].finite), e.absolute[0].finite)]
                for p, e in enumerate(result.stability)
            ]
    for call, values in found[True].items():
        for p, entries in enumerate(values):
            other = found[False][call][p]
            assert len(entries) == len(other), call
            for (a, e_a), (b, e_b) in zip(entries, other, strict=True):
                assert abs(a - b) <= e_a + e_b, (call, p, a, b, e_a, e_b)
    total = runtimes[True].evaluate(points[:1], mu_r2=scales[:1])
    resolved = runtimes[True].evaluate_resolved(points[:1], mu_r2=scales[:1])
    rows = [
        resolved.values[0][0][h][0].finite for h in range(len(resolved.helicity_ids))
    ]
    finite = total.values[0][0].finite
    assert abs(sum(rows) - finite) <= 1e-12 * abs(finite)
    selected = runtimes[True].evaluate(
        points[:1], mu_r2=scales[:1], helicities=list(resolved.helicity_ids)
    )
    assert abs(selected.values[0][0].finite - finite) <= 1e-12 * abs(finite)


@pytest.mark.loop_physics
def test_both_backends_agree_on_the_folded_plan(hgg):
    """F1f: TIRust and NinjaRS on the folded g g > h g g agree within their
    estimates at every point both serve, and the cross-backend check runs on
    it."""

    from pyamplicol import Runtime

    runtime = Runtime.load(hgg[True], process="p")
    points, scales = _points("li_gg_hgg")
    options = {"mu_r2": scales, "on_failure": "return", "stability_checks": _checks()}
    tirust = runtime.evaluate(points, reduction="tirust", **options)
    ninjars = runtime.evaluate(points, reduction="ninjars", **options)
    served = 0
    for p in range(len(points)):
        if tirust.values[p] is None or ninjars.values[p] is None:
            continue
        served += 1
        for s, (a, b) in enumerate(
            zip(tirust.values[p], ninjars.values[p], strict=True)
        ):
            e_a = tirust.stability[p].absolute[s].finite
            e_b = ninjars.stability[p].absolute[s].finite
            assert abs(a.finite - b.finite) <= e_a + e_b, (p, a, b, e_a, e_b)
    assert served
    crossed = runtime.evaluate(
        points,
        mu_r2=scales,
        on_failure="return",
        reduction="tirust",
        stability_checks=_checks(cross=True),
    )
    assert any(value is not None for value in crossed.values)
