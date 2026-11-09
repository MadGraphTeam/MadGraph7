# SPDX-License-Identifier: 0BSD
"""lit-ewprep: the one-loop request-model preparation, shared within a
generation (``loops.preparation_cache.PreparationMemo``) and prepared ahead
in the persistent per-user cache (``PreparationStore``).

* (b) the lowered program is formed once per preparation and the pruned
  Laurent check decides as the full walk;
* (a) one preparation per distinct recipe and one lowering per raw model per
  generation (zero warm); every request keeps its own refusals on a hit;
* (c) the cache: byte-identical artifacts and dry runs with the cache off,
  cold and warm (a fresh process) on a Born-interfered, a loop-induced and
  a complex-mass row of the shipped loop_sm; invalidation by content, recipe
  and code identity (the old entry stays); corrupt or foreign entries
  ignored and replaced; concurrent writers; an unwritable directory; the
  opt-outs (config, CLI, environment, ``model.cache``).

Every cache test uses its own cache root (``tmp_path``), seeded with the
compiled-model and counterterm-IR entries of one cache-off generation per
row (``_seed``), never the per-tier cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT = "PYAMPLICOL_LOOP_PREPARATION_CACHE"

# The three rows of the byte-identity test (shipped loop_sm, no_b_mass): a
# Born-interfered virtual, two loop-induced processes (one recipe of the
# order inference shared) and a complex-mass virtual.
_ZERO = (
    '{MB = "0", ymb = "0", MTA = "0", ymtau = "0", '
    'WT = "0", WW = "0", WZ = "0", WH = "0"}'
)
ROWS = {
    "born": ("", '{MB = "0"}', (("uux_ddx", "u u~ > d d~"),)),
    "induced": ("", _ZERO, (("gg_h", "g g > h"), ("gg_hg", "g g > h g"))),
    "cms": (
        '[loop.model.scheme]\nmass_scheme = "complex-mass"\n',
        '{MB = "0", MTA = "0", WT = "1.4915", WZ = "0", WW = "0", WH = "0"}',
        (("uux_ddx", "u u~ > d d~"),),
    ),
}


def _card(output: Path, row: str, extra: str = "") -> str:
    scheme, fixed, entries = ROWS[row]
    listed = ", ".join(
        f"{{expression = {json.dumps(e)}, name = {json.dumps(n)}}}" for n, e in entries
    )
    return f"""
action = "generate"
[model]
source = "loop_sm"
restriction = "no_b_mass"
{extra}
[process]
entries = [{listed}]
loop_coupling_orders = ["QCD"]
[loop]
r2_strategy = "model"
{scheme}
[loop.preparation]
fixed_parameters = {fixed}
[generation]
output = {json.dumps(str(output))}
emit_api_bundle = false
[generation.validation]
post_build_validation = false
"""


def _processes(row: str):
    from pyamplicol.api import ProcessRequest, ProcessSet

    return ProcessSet(tuple(ProcessRequest.parse(e, name=n) for n, e in ROWS[row][2]))


def _generate(work: Path, row: str, extra: str = "") -> Path:
    """The public generation of ``row`` into ``work / "artifact"`` (in this
    process, with the cache root of ``$PYAMPLICOL_CACHE_DIR``)."""

    from pyamplicol.api import Generator
    from pyamplicol.config import resolve_config

    work.mkdir(parents=True, exist_ok=True)
    card = work / "card.toml"
    output = work / "artifact"
    card.write_text(_card(output, row, extra))
    Generator(resolve_config(card)).generate(_processes(row), output)
    return output


def _plan(work: Path, row: str, output: Path):
    """The public dry run of ``row`` (its card naming ``output``)."""

    from pyamplicol.api import Generator
    from pyamplicol.config import resolve_config

    work.mkdir(parents=True, exist_ok=True)
    card = work / "card.toml"
    card.write_text(_card(output, row))
    return Generator(resolve_config(card)).plan(_processes(row))


# Fields that differ between two cache-off generations: the manifest's
# creation time and phase timings (and, defensively, an output path in the
# configuration records).
def _identity(artifact: Path) -> dict[str, str]:
    digests = {}
    for path in sorted(artifact.rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(artifact).as_posix()
        data = path.read_bytes()
        if name == "artifact.json":
            manifest = json.loads(data)
            manifest.pop("created_utc")
            manifest["extensions"]["generation"].pop("phase_timings_seconds")
            data = json.dumps(manifest, sort_keys=True).encode()
        elif name.startswith("config/"):
            data = data.replace(str(artifact).encode(), b"<output>")
        digests[name] = hashlib.sha256(data).hexdigest()
    return digests


def _entries(root: Path) -> list[Path]:
    return sorted((root / "models" / "loop-preparations").glob("*.json"))


@pytest.fixture(scope="module")
def seeded(tmp_path_factory, loop_native):
    """Per row a cache-off generation (the reference artifact) in a seed
    cache root, whose compiled-model and counterterm-IR entries every test
    root copies (``root``)."""

    del loop_native
    base = tmp_path_factory.mktemp("preparation-seed")
    saved = {key: os.environ.get(key) for key in ("PYAMPLICOL_CACHE_DIR", ENVIRONMENT)}
    os.environ["PYAMPLICOL_CACHE_DIR"] = str(base / "cache")
    os.environ[ENVIRONMENT] = "0"
    try:
        artifacts = {row: _generate(base / row, row) for row in ROWS}
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    assert not _entries(base / "cache")
    return base / "cache", artifacts


def _root(seeded, path: Path) -> Path:
    """A new cache root with the seed's compiled-model and counterterm-IR
    entries (never a preparation entry)."""

    seed, _ = seeded
    models = path / "models"
    models.mkdir(parents=True)
    for entry in (seed / "models").glob("*.json"):
        (models / entry.name).write_bytes(entry.read_bytes())
    return path


def _subprocess_generate(root: Path, work: Path, row: str) -> dict:
    """The generation of ``row`` in a fresh process (cache root ``root``),
    with its preparation memo's report."""

    script = textwrap.dedent(
        f"""
        import json, sys
        sys.path.insert(0, {json.dumps(str(ROOT / "tests" / "unit"))})
        import pyamplicol.loops.preparation_cache as cache
        reports = []
        init = cache.PreparationMemo.__init__
        def spy(self, *args, **kwargs):
            init(self, *args, **kwargs)
            reports.append(self)
        cache.PreparationMemo.__init__ = spy
        from pathlib import Path
        from test_loop_preparation_cache import _generate
        _generate(Path({json.dumps(str(work))}), {json.dumps(row)})
        print(json.dumps([memo.report() for memo in reports]))
        """
    )
    environment = {**os.environ, "PYAMPLICOL_CACHE_DIR": str(root)}
    environment.pop(ENVIRONMENT, None)
    done = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


class _Memos:
    """The ``PreparationMemo`` objects created while it is installed."""

    def __init__(self, monkeypatch):
        from pyamplicol.loops import preparation_cache as cache

        self.memos = []
        init = cache.PreparationMemo.__init__

        def spy(memo, *args, **kwargs):
            init(memo, *args, **kwargs)
            self.memos.append(memo)

        monkeypatch.setattr(cache.PreparationMemo, "__init__", spy)


# ------------------------------------------------------------ (i) byte identity


@pytest.mark.loop_physics
@pytest.mark.parametrize("row", sorted(ROWS))
def test_artifacts_and_dry_runs_are_byte_identical_off_cold_and_warm(
    seeded, monkeypatch, tmp_path, row
):
    """Cache off, cold (an empty preparation cache) and warm (a fresh
    process reading the cold run's entry) give the same artifact files
    (stored plans, companion plans, native members, manifests: only the
    phase timings and the output path differ between two cache-off runs)
    and the same dry run; cold lowers once and writes one entry, warm loads
    it and lowers nothing."""

    _, reference = seeded
    root = _root(seeded, tmp_path / "cache")
    monkeypatch.setenv("PYAMPLICOL_CACHE_DIR", str(root))
    monkeypatch.delenv(ENVIRONMENT, raising=False)
    spy = _Memos(monkeypatch)
    cold = _generate(tmp_path / "cold", row)
    (memo,) = spy.memos
    assert memo.counts["lowerings"] == 1 and memo.counts["program_saves"] == 1
    assert len(_entries(root)) == 1
    warm = _subprocess_generate(root, tmp_path / "warm", row)
    (report,) = warm
    assert report.get("lowerings", 0) == 0 and report["program_loads"] == 1
    assert report["store"]["hits"] == 1
    expected = _identity(reference[row])
    assert _identity(cold) == expected
    assert _identity(tmp_path / "warm" / "artifact") == expected
    # The dry run fills nothing new and reports the same plan, cache on
    # (warm, in this process) and off.
    planned = _plan(tmp_path / "plan-warm", row, tmp_path / "planned")
    monkeypatch.setenv(ENVIRONMENT, "0")
    off = _plan(tmp_path / "plan-off", row, tmp_path / "planned")
    assert not planned.unsupported_features
    assert repr(off) == repr(planned)
    assert len(_entries(root)) == 1


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["uux_ddxg_lc", "uux_uuxg_full"])
def test_the_pinned_process_bytes_hold_cold_and_warm(
    loop_sm_compiled_model, loop_native, tmp_path, name
):
    """The byte pins of ``test_loop_v2_fail_closed``
    (``test_plans_the_guards_do_not_refuse_keep_their_bytes``: the plan,
    companion plan and native member digests of the slice before that
    package), here generated with a preparation memo and the persistent
    store: cold (an empty store: lowered and written) and warm (a new memo
    and store object reading the entry: nothing lowered)."""

    import importlib.util

    from pyamplicol.loops.preparation_cache import PreparationMemo, PreparationStore
    from pyamplicol.loops.v2.artifact import generate_process
    from pyamplicol.models.loading import load_compiled_model

    spec = importlib.util.spec_from_file_location(
        "_lit_ewprep_fail_closed_pins",
        ROOT / "tests" / "unit" / "test_loop_v2_fail_closed.py",
    )
    pins = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pins)
    compiled = load_compiled_model(loop_sm_compiled_model)
    process, born, accuracy = pins.CONTROLS[name]
    directory = tmp_path / "loop-preparations"
    counts = []
    for _arm in ("cold", "warm"):
        memo = PreparationMemo(PreparationStore(directory))
        generated = generate_process(
            compiled, pins._request(process, born, accuracy), loop_native, memo=memo
        )
        assert pins._identity(generated) == pins.CONTROL_BYTES[name]
        counts.append(memo.counts)
    cold, warm = counts
    assert cold["lowerings"] == 1 and cold["program_saves"] == 1
    assert warm["lowerings"] == 0 and warm["program_loads"] == 1
    assert len(list(directory.iterdir())) == 1


# ------------------------------------------------------------ (vii) sharing


@pytest.mark.loop_physics
def test_one_preparation_per_recipe_and_one_lowering_per_model(
    seeded, monkeypatch, tmp_path
):
    """Two loop-induced processes (g g > h, g g > h g): the order inference
    prepares their common recipe once, each slice its own (the inferred
    loop orders differ): three preparations, one reuse, one lowering cold;
    zero lowerings with a filled cache (a cleared memo: a new generation in
    this process); the dry run shares the same way."""

    root = _root(seeded, tmp_path / "cache")
    monkeypatch.setenv("PYAMPLICOL_CACHE_DIR", str(root))
    monkeypatch.delenv(ENVIRONMENT, raising=False)
    spy = _Memos(monkeypatch)
    _generate(tmp_path / "cold", "induced")
    _generate(tmp_path / "warm", "induced")
    planned = _plan(tmp_path / "plan", "induced", tmp_path / "planned")
    assert not planned.unsupported_features
    cold, warm, plan = (memo.counts for memo in spy.memos)
    for counts in (cold, warm):
        assert counts["requests"] == 3 and counts["request_hits"] == 1
        assert counts["restrictions"] == 1 and counts["restriction_hits"] == 2
    assert cold["lowerings"] == 1 and cold["program_loads"] == 0
    assert warm["lowerings"] == 0 and warm["program_loads"] == 1
    # The dry run: the order inference's recipe once (two members), each
    # process's own once.
    assert plan["requests"] == 3 and plan["request_hits"] == 1
    assert plan["lowerings"] == 0 and plan["program_loads"] == 1


@pytest.mark.loop_physics
def test_a_program_the_restriction_does_not_lower_is_shared_and_stored(
    seeded, monkeypatch, tmp_path
):
    """A model without a restriction card (``restriction = "none"``): the
    restriction lowers nothing, so the memo lowers the model's program
    itself and it is stored and reused like any other; cold and warm give
    the artifact files of the preparation as it was
    (``build_covariant_structure`` lowering the model)."""

    from pyamplicol.api import Generator
    from pyamplicol.config import resolve_config
    from pyamplicol.loops import preparation_cache as cache
    from pyamplicol.loops.model_restrictions import prepare_loop_restrictions

    root = _root(seeded, tmp_path / "cache")
    monkeypatch.setenv("PYAMPLICOL_CACHE_DIR", str(root))
    monkeypatch.delenv(ENVIRONMENT, raising=False)

    def generate(name):
        work = tmp_path / name
        work.mkdir()
        card = work / "card.toml"
        text = _card(work / "artifact", "born")
        card.write_text(
            text.replace('restriction = "no_b_mass"', 'restriction = "none"')
        )
        Generator(resolve_config(card)).generate(_processes("born"), work / "artifact")
        return work / "artifact"

    with monkeypatch.context() as patch:
        # The reference: the declarations as before (no memo lowering, no
        # store), the program lowered by the covariant structure.
        patch.setattr(
            cache.PreparationMemo,
            "restrictions",
            lambda memo, raw, *, simplify=True: prepare_loop_restrictions(
                raw, simplify=simplify
            ),
        )
        reference = generate("reference")
    assert not _entries(root)
    spy = _Memos(monkeypatch)
    cold = generate("cold")
    warm = generate("warm")
    cold_counts, warm_counts = (memo.counts for memo in spy.memos)
    assert cold_counts["lowerings"] == 1 and cold_counts["program_saves"] == 1
    assert warm_counts["lowerings"] == 0 and warm_counts["program_loads"] == 1
    # No card: the restriction returned the model itself, without a program.
    (declarations,) = spy.memos[0]._sources.values()
    assert declarations.derived is declarations.original
    assert len(_entries(root)) == 1
    expected = _identity(reference)
    assert _identity(cold) == expected and _identity(warm) == expected


@pytest.mark.loop_physics
def test_a_dry_run_prepares_a_generation_ahead(seeded, monkeypatch, tmp_path):
    """With the compiled model cached, the dry run (``Generator.plan``)
    fills the preparation cache; the generation that follows (a fresh
    process) lowers nothing and gives the cache-off artifact."""

    _, reference = seeded
    root = _root(seeded, tmp_path / "cache")
    monkeypatch.setenv("PYAMPLICOL_CACHE_DIR", str(root))
    monkeypatch.delenv(ENVIRONMENT, raising=False)
    spy = _Memos(monkeypatch)
    planned = _plan(tmp_path / "plan", "born", tmp_path / "planned")
    assert not planned.unsupported_features
    (memo,) = spy.memos
    assert memo.counts["lowerings"] == 1 and memo.counts["program_saves"] == 1
    assert len(_entries(root)) == 1
    (report,) = _subprocess_generate(root, tmp_path / "warm", "born")
    assert report.get("lowerings", 0) == 0 and report["program_loads"] == 1
    assert _identity(tmp_path / "warm" / "artifact") == _identity(reference["born"])


@pytest.mark.loop_physics
def test_probe_slices_share_the_preparation(seeded, monkeypatch, tmp_path):
    """With ``generation.loop_induced_probes`` the probe slice of g g > h
    (zero widths already: the production recipe) reuses the slice's
    prepared request model; the probe and Goldstone slices receive the
    generation's memo (``ward.ward_slice`` / ``goldstone.goldstone_slice``,
    which a QCD request on loop_sm does not reach)."""

    from pyamplicol.loops.v2 import goldstone, ward

    root = _root(seeded, tmp_path / "cache")
    monkeypatch.setenv("PYAMPLICOL_CACHE_DIR", str(root))
    spy = _Memos(monkeypatch)
    received = []
    original = ward.ward_slice

    def ward_slice(*args, **options):
        received.append(options.get("memo"))
        return original(*args, **options)

    monkeypatch.setattr(ward, "ward_slice", ward_slice)
    work = tmp_path / "probes"
    work.mkdir()
    card = work / "card.toml"
    card.write_text(
        _card(work / "artifact", "induced").replace(
            "[generation]\n", "[generation]\nloop_induced_probes = true\n"
        )
    )
    from pyamplicol.api import Generator
    from pyamplicol.config import resolve_config

    Generator(resolve_config(card)).generate(_processes("induced"), work / "artifact")
    (memo,) = spy.memos
    assert received and all(item is memo for item in received)
    # Two slices and their probe slices: the probes' recipes are the
    # slices' (zero widths, real masses), so they reuse them.
    assert memo.counts["requests"] == 3
    assert memo.counts["request_hits"] == 1 + len(received)
    # The Goldstone side hands the memo to its Ward slice.
    seen = []
    monkeypatch.setattr(
        ward, "ward_slice", lambda *args, **options: seen.append(options) or "slice"
    )
    monkeypatch.setattr(goldstone, "goldstone_process", lambda *args: "a > b")
    assert (
        goldstone.goldstone_slice(None, _Request(), None, None, 1, 2, memo=memo)
        == "slice"
    )
    assert seen == [{"memo": memo}]


@dataclass(frozen=True)
class _Request:
    process: str = "x > y"


# ------------------------------------------------------------ (viii) refusals


@pytest.mark.loop_physics
def test_a_shared_preparation_keeps_every_request_s_refusals(
    seeded, monkeypatch, tmp_path
):
    """Two requests of one recipe (the process and the veto are not in it):
    the second reuses the first's prepared model, from the memo and from a
    filled cache, and still gets its own refusals: the gauge check
    (``ew_policy.require_loop_gauge``) and the veto-unlabelled-counterterms
    check run for it (with its process and veto) and refuse it."""

    from dataclasses import replace

    from pyamplicol.loops.preparation_cache import PreparationMemo, PreparationStore
    from pyamplicol.loops.v2 import assemble, ew_policy
    from pyamplicol.loops.v2.assemble import SliceRequest, prepare_request_model
    from pyamplicol.loops.v2.refusals import LoopRequestUnsupported
    from pyamplicol.models.loading import load_cached_model_source

    seed, _ = seeded
    compiled = load_cached_model_source(
        "loop_sm",
        restriction="no_b_mass",
        simplify=True,
        cache_dir=seed / "models",
        require_supported=False,
    )
    assert compiled is not None
    first = SliceRequest(
        "u u~ > d d~",
        {"QCD": 2},
        {"QCD": 4},
        corrections=("QCD",),
        fixed_parameters={"MB": "0"},
    )
    second = replace(first, process="d d~ > u u~", particle_veto=(5,))
    calls = []
    gauge = ew_policy.require_loop_gauge

    def require_loop_gauge(compiled, request, species):
        calls.append(("gauge", request.process))
        return gauge(compiled, request, species)

    def veto(kernels, process, particle_veto, corrections, species):
        calls.append(("veto", process))
        if particle_veto:
            raise LoopRequestUnsupported("veto-unlabelled-counterterms", process)

    monkeypatch.setattr(ew_policy, "require_loop_gauge", require_loop_gauge)
    monkeypatch.setattr(assemble, "require_labelled_counterterms_for_veto", veto)
    lowered = []
    for memo in (
        PreparationMemo(PreparationStore(tmp_path / "loop-preparations")),
        PreparationMemo(PreparationStore(tmp_path / "loop-preparations")),
    ):
        calls.clear()
        prepare_request_model(compiled, first, memo=memo)
        with pytest.raises(LoopRequestUnsupported, match="d d~ > u u~"):
            prepare_request_model(compiled, second, memo=memo)
        assert memo.counts["requests"] == 1 and memo.counts["request_hits"] == 1
        assert calls == [
            ("gauge", "u u~ > d d~"),
            ("veto", "u u~ > d d~"),
            ("gauge", "d d~ > u u~"),
            ("veto", "d d~ > u u~"),
        ]
        lowered.append(memo.counts["lowerings"])
    # The second memo read the first's programs from the store (the
    # unrepaired source and its repaired form: two).
    assert lowered[0] >= 1 and lowered[1] == 0
    assert memo.counts["program_loads"] == lowered[0]


# ------------------------------------------- (b) one lowering, the same program


@pytest.mark.loop_physics
def test_the_handed_over_program_is_the_fresh_lowering_and_the_laurent_check(
    seeded,
):
    """``prepare_loop_restrictions`` hands its program to the final
    restricted model: every definition equals a fresh lowering of that
    model (canonical strings); the pruned Laurent check decides every
    lowered expression as the full walk; a program is refused for a model
    with other lowering inputs."""

    from symbolica import AtomType, E

    from pyamplicol.loops.model_expressions import epsilon_symbol
    from pyamplicol.loops.model_restrictions import prepare_loop_restrictions
    from pyamplicol.loops.scalar_frontend import (
        ScalarExpression,
        lower_scalar_program,
        require_laurent_scalar,
    )
    from pyamplicol.models.loading import load_cached_model_source
    from pyamplicol.models.loop_import import RawLoopModel

    seed, _ = seeded
    compiled = load_cached_model_source(
        "loop_sm",
        restriction="no_b_mass",
        simplify=True,
        cache_dir=seed / "models",
        require_supported=False,
    )
    declarations = prepare_loop_restrictions(compiled.raw_loop_model)
    handed = declarations.program
    assert handed.library.raw is declarations.derived
    fresh = lower_scalar_program(declarations.derived)
    assert list(handed.definitions) == list(fresh.definitions)
    for key, definition in fresh.definitions.items():
        other = handed.definitions[key]
        assert other.name == definition.name
        for a, b in (
            (other.value.expression, definition.value.expression),
            (other.diagnostic, definition.diagnostic),
        ):
            assert a.to_canonical_string() == b.to_canonical_string()

    function = "epsilon-dependent scalar helper or function arguments require expansion"
    diagnostic = "epsilon-dependent predicate diagnostic"

    def full_walk(expression):
        # ``require_laurent_scalar`` as it was (every atom visited).
        epsilon = epsilon_symbol()
        pending = [expression.value, expression.diagnostic]
        while pending:
            atom = pending.pop()
            if atom.get_type() == AtomType.Fn and epsilon in atom.get_all_symbols():
                raise NotImplementedError(function)
            if atom.get_type() not in {AtomType.Num, AtomType.Var}:
                pending.extend(atom)
        if epsilon in expression.diagnostic.get_all_symbols():
            raise NotImplementedError(diagnostic)

    def outcome(check, expression):
        try:
            check(expression)
        except NotImplementedError as error:
            return str(error)
        return None

    eps = "pyamplicol_loop::epsilon"
    synthetic = {
        (f"x*{eps}^-1 + sin(y)", "0"): None,
        (f"x*sin({eps})", "0"): function,
        (f"f(g(y, {eps}))*{eps}", "0"): function,
        (f"x+{eps}^2*f(y)", f"{eps}"): diagnostic,
        (f"x+{eps}", f"f({eps})"): function,
        ("x", "0"): None,
    }
    cases = [
        (ScalarExpression(definition.value.expression, definition.diagnostic), None)
        for definition in fresh.definitions.values()
    ] + [
        (ScalarExpression(E(value), E(diag)), expected)
        for (value, diag), expected in synthetic.items()
    ]
    for case, expected in cases:
        assert outcome(require_laurent_scalar, case) == outcome(full_walk, case)
        assert outcome(require_laurent_scalar, case) == expected
    # Another model (other lowering inputs): refused.
    payload = declarations.derived.to_dict()
    payload["name"] = payload["name"] + "_other"
    with pytest.raises(ValueError, match="another raw model"):
        handed.library.rebind(RawLoopModel.from_dict(payload))


# ------------------------------------- (ii) invalidation, (iii) corrupt entries


def _program_strings(program) -> list[tuple[str, str, str]]:
    return [
        (
            definition.name,
            definition.value.expression.to_canonical_string(),
            definition.diagnostic.to_canonical_string(),
        )
        for definition in program.definitions.values()
    ]


@pytest.fixture
def raw_model(seeded):
    from pyamplicol.models.loading import load_cached_model_source

    seed, _ = seeded
    compiled = load_cached_model_source(
        "loop_sm",
        restriction="no_b_mass",
        simplify=True,
        cache_dir=seed / "models",
        require_supported=False,
    )
    return compiled.raw_loop_model


def _changed(raw, change):
    from pyamplicol.models.loop_import import RawLoopModel

    payload = raw.to_dict()
    change(payload)
    return RawLoopModel.from_dict(payload)


@pytest.mark.loop_physics
def test_content_simplification_and_code_identity_invalidate(
    raw_model, monkeypatch, tmp_path
):
    """A changed parameter value, a changed restriction-card entry, the
    other simplification and a changed code identity each miss, are
    lowered again (equal to an uncached lowering) and written beside the old
    entry, which stays."""

    from pyamplicol.loops import preparation_cache as cache
    from pyamplicol.loops.model_restrictions import prepare_loop_restrictions

    store = cache.PreparationStore(tmp_path / "loop-preparations")

    def prepared(raw, simplify=True):
        # A new store per generation (each forms its keys once).
        memo = cache.PreparationMemo(cache.PreparationStore(store.directory))
        declarations = memo.restrictions(raw, simplify=simplify)
        program = declarations.program or None
        return memo, declarations, program

    memo, _, program = prepared(raw_model)
    assert memo.counts["lowerings"] == 1 and len(list(store.directory.iterdir())) == 1
    memo, _, again = prepared(raw_model)
    assert memo.counts["program_loads"] == 1
    assert _program_strings(again) == _program_strings(program)

    def value(payload):
        record = next(
            r
            for r in payload["collections"]["parameters"]
            if r["fields"].get("nature") == "internal"
            and r["fields"]["value"].get("kind") == "expression"
        )
        record["fields"]["value"]["expression"] = (
            "2*(" + record["fields"]["value"]["expression"] + ")"
        )

    def card(payload):
        # The first nonzero entry of the restriction card, doubled.
        import re
        from decimal import Decimal

        text = payload["metadata"]["restriction"]["card"]
        lines = text.splitlines()
        for index, line in enumerate(lines):
            match = re.match(r"^(\s*\d+\s+)([0-9.eE+-]+)(\s.*)?$", line)
            if match and Decimal(match[2]) != 0:
                doubled = str(Decimal(match[2]) * 2)
                lines[index] = match[1] + doubled + (match[3] or "")
                break
        else:
            pytest.fail("the restriction card has no nonzero entry")
        payload["metadata"]["restriction"]["card"] = "\n".join(lines) + "\n"

    entries = 1
    for raw in (_changed(raw_model, value), _changed(raw_model, card)):
        memo, _, program = prepared(raw)
        entries += 1
        assert memo.counts["lowerings"] == 1 and memo.counts["program_saves"] == 1
        assert len(list(store.directory.iterdir())) == entries
        uncached = prepare_loop_restrictions(raw).program
        assert _program_strings(program) == _program_strings(uncached)
    # The other simplification: the restriction lowers nothing, the memo
    # lowers the restricted model ``build_covariant_structure`` would lower,
    # under its own key, stored beside the others.
    from pyamplicol.loops.scalar_frontend import lower_scalar_program

    assert store.key(raw_model, False) != store.key(raw_model, True)
    assert prepare_loop_restrictions(raw_model, simplify=False).program is None
    memo, declarations, program = prepared(raw_model, simplify=False)
    entries += 1
    assert memo.counts["lowerings"] == 1 and memo.counts["program_saves"] == 1
    assert len(list(store.directory.iterdir())) == entries
    assert program.library.raw is declarations.derived
    assert _program_strings(program) == _program_strings(
        lower_scalar_program(declarations.derived)
    )
    memo, _, again = prepared(raw_model, simplify=False)
    assert memo.counts["program_loads"] == 1 and memo.counts["lowerings"] == 0
    assert _program_strings(again) == _program_strings(program)
    # A changed code identity: a miss, a new entry, the old ones stay.
    identity = cache.code_identity()
    monkeypatch.setattr(
        cache, "code_identity", lambda: {**identity, "sources_sha256": "edited"}
    )
    memo, _, program = prepared(raw_model)
    assert memo.counts["lowerings"] == 1 and memo.counts["program_saves"] == 1
    assert len(list(store.directory.iterdir())) == entries + 1


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    "damage", ["truncated", "garbage", "key", "schema", "identity", "payload"]
)
def test_damaged_or_foreign_entries_are_ignored_and_replaced(
    raw_model, tmp_path, damage
):
    """A truncated or garbage entry, a wrong embedded key, schema or code
    identity, or a payload that does not match its digest: ignored (never
    an error), lowered again with the same program, and replaced
    atomically by a valid entry."""

    from pyamplicol.loops import preparation_cache as cache

    store = cache.PreparationStore(tmp_path / "loop-preparations")
    reference = cache.PreparationMemo(store).restrictions(raw_model).program
    (path,) = store.directory.iterdir()
    data = path.read_bytes()
    header, _, payload = data.partition(b"\n")
    record = json.loads(header)
    if damage == "truncated":
        data = data[: len(data) // 2]
    elif damage == "garbage":
        data = b"\x00garbage" * 64
    elif damage == "payload":
        data = header + b"\n" + payload[:-8] + b"\x00" * 8
    else:
        record[{"key": "key", "schema": "schema", "identity": "identity"}[damage]] = (
            {"sources_sha256": "x"} if damage == "identity" else "other"
        )
        data = json.dumps(record).encode() + b"\n" + payload
    path.write_bytes(data)
    memo = cache.PreparationMemo(store)
    program = memo.restrictions(raw_model).program
    assert store.events["rejected"] == 1
    assert memo.counts["lowerings"] == 1 and memo.counts["program_saves"] == 1
    assert _program_strings(program) == _program_strings(reference)
    # Replaced by a valid entry: the next memo loads it.
    again = cache.PreparationMemo(store)
    assert _program_strings(again.restrictions(raw_model).program) == (
        _program_strings(reference)
    )
    assert again.counts["program_loads"] == 1
    assert [p.name for p in store.directory.iterdir()] == [path.name]


# ------------------------------------------- (iv) concurrency, (v) unwritable


@pytest.mark.loop_physics
def test_concurrent_writers_of_one_key_both_succeed(seeded, tmp_path):
    """Two processes filling one key at once both succeed and leave one
    valid entry (a reader never sees a partial file)."""

    seed, _ = seeded
    directory = tmp_path / "loop-preparations"
    script = textwrap.dedent(
        f"""
        import json
        from pathlib import Path
        from pyamplicol.loops import preparation_cache as cache
        from pyamplicol.models.loading import load_cached_model_source
        compiled = load_cached_model_source(
            "loop_sm", restriction="no_b_mass", simplify=True,
            cache_dir=Path({json.dumps(str(seed / "models"))}),
            require_supported=False,
        )
        memo = cache.PreparationMemo(
            cache.PreparationStore(Path({json.dumps(str(directory))}))
        )
        memo.restrictions(compiled.raw_loop_model)
        print(json.dumps(dict(memo.counts)))
        """
    )
    environment = {**os.environ}
    runs = [
        subprocess.Popen(
            [sys.executable, "-c", script],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    outputs = [run.communicate() for run in runs]
    assert [run.returncode for run in runs] == [0, 0], [e[-2000:] for _, e in outputs]
    entries = sorted(directory.iterdir())
    assert len(entries) == 1 and entries[0].name.endswith(cache_suffix())
    from pyamplicol.loops import preparation_cache as cache
    from pyamplicol.models.loading import load_cached_model_source

    compiled = load_cached_model_source(
        "loop_sm",
        restriction="no_b_mass",
        simplify=True,
        cache_dir=seed / "models",
        require_supported=False,
    )
    memo = cache.PreparationMemo(cache.PreparationStore(directory))
    memo.restrictions(compiled.raw_loop_model)
    assert memo.counts["program_loads"] == 1


def cache_suffix() -> str:
    from pyamplicol.loops.preparation_cache import ENTRY_SUFFIX

    return ENTRY_SUFFIX


@pytest.mark.loop_physics
def test_an_unwritable_cache_directory_leaves_the_generation_uncached(
    raw_model, tmp_path, caplog
):
    """A cache directory that cannot be written: the program is lowered and
    returned, nothing is written, one warning names the directory."""

    from pyamplicol.loops import preparation_cache as cache

    parent = tmp_path / "locked"
    parent.mkdir()
    parent.chmod(0o500)
    try:
        if os.access(parent, os.W_OK):
            pytest.fail("the test needs a directory it cannot write (not root)")
        store = cache.PreparationStore(parent / "loop-preparations")
        memo = cache.PreparationMemo(store)
        with caplog.at_level("WARNING", logger=cache.__name__):
            program = memo.restrictions(raw_model).program
        assert program is not None and memo.counts["lowerings"] == 1
        assert store.events["unusable"] == 1 and not memo.counts["program_saves"]
        assert len([r for r in caplog.records if "cannot write" in r.message]) == 1
        assert not (parent / "loop-preparations").exists()
    finally:
        parent.chmod(0o700)


# ------------------------------------------------------------------ (vi) opt-out


@pytest.mark.loop_fast
def test_every_switch_turns_the_cache_off(monkeypatch, tmp_path):
    """One rule: the cache is used only when every switch allows it
    (``model.cache``, ``model.loop_preparation_cache`` / the CLI flag,
    ``$PYAMPLICOL_LOOP_PREPARATION_CACHE``); its directory is the model
    cache's ``loop-preparations``; an invalid environment value is refused
    with the one message."""

    from pyamplicol.cli import parse_cli
    from pyamplicol.config import ConfigurationError, ModelConfig
    from pyamplicol.loops import preparation_cache as cache

    monkeypatch.delenv(ENVIRONMENT, raising=False)
    monkeypatch.setenv("PYAMPLICOL_CACHE_DIR", str(tmp_path / "root"))
    default = cache.store_for(ModelConfig())
    assert default.directory == tmp_path / "root" / "models" / "loop-preparations"
    explicit = cache.store_for(ModelConfig(cache_dir=tmp_path / "models"))
    assert explicit.directory == tmp_path / "models" / "loop-preparations"
    assert cache.store_for(ModelConfig(cache=False)) is None
    assert cache.store_for(ModelConfig(loop_preparation_cache=False)) is None
    on = parse_cli(("generate",)).resolve().effective.model
    assert on.loop_preparation_cache is True
    off = parse_cli(("generate", "--no-loop-preparation-cache")).resolve().effective
    assert off.model.loop_preparation_cache is False
    assert cache.store_for(off.model) is None
    flag = parse_cli(("generate", "--loop-preparation-cache")).resolve().effective
    assert flag.model.loop_preparation_cache is True
    no_model = parse_cli(("generate", "--no-model-cache")).resolve().effective
    assert cache.store_for(no_model.model) is None
    for text, expected in (("0", None), ("false", None), ("OFF", None), ("no", None)):
        monkeypatch.setenv(ENVIRONMENT, text)
        assert cache.store_for(ModelConfig()) is expected
    for text in ("1", "true", "on", "yes", ""):
        monkeypatch.setenv(ENVIRONMENT, text)
        assert cache.store_for(ModelConfig()) is not None
        # A true value never turns on what the configuration turned off.
        assert cache.store_for(ModelConfig(loop_preparation_cache=False)) is None
    monkeypatch.setenv(ENVIRONMENT, "maybe")
    with pytest.raises(
        ConfigurationError,
        match=r"^PYAMPLICOL_LOOP_PREPARATION_CACHE must be a boolean",
    ):
        cache.store_for(ModelConfig())


@pytest.mark.loop_physics
@pytest.mark.parametrize("switch", ["config", "cli", "environment", "model-cache"])
def test_an_opted_out_generation_reads_and_writes_no_entry(
    seeded, monkeypatch, tmp_path, switch
):
    """With the cache off (config, CLI, environment, or no model cache) a
    generation neither reads an existing entry nor writes one."""

    from pyamplicol.api import Generator
    from pyamplicol.cli import parse_cli
    from pyamplicol.config import resolve_config

    root = _root(seeded, tmp_path / "cache")
    monkeypatch.setenv("PYAMPLICOL_CACHE_DIR", str(root))
    monkeypatch.delenv(ENVIRONMENT, raising=False)
    # A filled cache first (an entry an opted-out run must not read).
    _generate(tmp_path / "fill", "born")
    (entry,) = _entries(root)
    stamp = entry.stat().st_mtime_ns
    spy = _Memos(monkeypatch)
    work = tmp_path / "off"
    work.mkdir()
    card = work / "card.toml"
    output = work / "artifact"
    text = _card(output, "born")
    if switch == "config":
        text = text.replace("[process]", "loop_preparation_cache = false\n[process]")
    elif switch == "model-cache":
        text = text.replace("[process]", "cache = false\n[process]")
    elif switch == "environment":
        monkeypatch.setenv(ENVIRONMENT, "off")
    card.write_text(text)
    if switch == "cli":
        config = parse_cli(
            ("generate", "--card", str(card), "--no-loop-preparation-cache")
        ).resolve()
    else:
        config = resolve_config(card)
    reads = []
    from pyamplicol.loops import preparation_cache as cache

    monkeypatch.setattr(
        cache.PreparationStore, "load", lambda *a, **k: reads.append(a) or None
    )
    Generator(config).generate(_processes("born"), output)
    (memo,) = spy.memos
    assert memo.store is None and memo.counts["lowerings"] == 1
    assert not reads
    assert _entries(root) == [entry] and entry.stat().st_mtime_ns == stamp
