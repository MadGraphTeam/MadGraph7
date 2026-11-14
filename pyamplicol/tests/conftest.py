# SPDX-License-Identifier: 0BSD
"""Shared inputs for the one-loop tests.

Loop modules carry one of the ``loop_fast``, ``loop_physics`` or
``loop_nightly`` markers. External inputs (UFO models, MadGraph source and
saved PACBIN/artifact controls) resolve here once: an environment variable
overrides the default location, which is derived from the checkout's own
location (no host path is assumed). A loop test whose input or staged native
runtime is missing fails, naming what is missing and where it was looked for,
whether it requests the input as a fixture or at run time
(``request.getfixturevalue``): a relocated checkout must never pass with its
loop tests skipped. Other tests skip with the input's name, as do opt-in
inputs without a default location. Committed provider references under
``tests/fixtures/loop`` are mandatory. Without ``SYMBOLICA_LICENSE`` the loop
tests skip, or fail with ``PYAMPLICOL_REQUIRE_SYMBOLICA_LICENSE=1`` (licensed
CI jobs).
"""

from __future__ import annotations

import json
import os
from functools import cache
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
LOOP_REFERENCES = TESTS / "fixtures" / "loop"
LOOP_MARKERS = ("loop_fast", "loop_physics", "loop_nightly")
DEFERRED_MARKER = "deferred_direct_r2"
# Set to 1, a missing SYMBOLICA_LICENSE fails the loop tests (licensed CI).
REQUIRE_LICENSE = "PYAMPLICOL_REQUIRE_SYMBOLICA_LICENSE"

_MG5 = Path("FOR_REFERENCE_ONLY/MG5_aMC_v3_7_0")


def _main_checkout() -> Path:
    """The repository's main checkout (it holds the untracked reference
    sources), derived from this checkout's own location: the parent of git's
    common directory, which a linked worktree shares with it, when this
    checkout is the working tree of that repository; else this checkout
    itself (a copy without its repository, or one inside another
    repository's working tree)."""

    import subprocess

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", *arguments],
            cwd=TESTS.parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    try:
        top, common = git("--show-toplevel"), git("--git-common-dir")
    except (OSError, subprocess.CalledProcessError):
        return TESTS.parent
    if not top or not common or Path(top).resolve() != TESTS.parent:
        return TESTS.parent
    return Path(common).resolve().parent


def _workspace(main: Path) -> tuple[Path | None, str]:
    """The loop workspace and how it was found: ``PYAMPLICOL_LOOP_WORKSPACE``,
    else a ``workspace`` directory beside the main checkout, else none."""

    if configured := os.environ.get("PYAMPLICOL_LOOP_WORKSPACE"):
        return Path(configured), f"PYAMPLICOL_LOOP_WORKSPACE={configured}"
    beside = main.parent / "workspace"
    if beside.is_dir():
        return beside, f"the workspace {beside} beside the main checkout {main}"
    return None, (
        f"no loop workspace: PYAMPLICOL_LOOP_WORKSPACE is unset and {beside} "
        f"(beside the main checkout {main}) does not exist"
    )


_MAIN_CHECKOUT = _main_checkout()
_WORKSPACE, _WORKSPACE_ORIGIN = _workspace(_MAIN_CHECKOUT)


def _under(root: Path | None, *relative: str | Path) -> tuple[Path, ...]:
    """Default candidates below a root that may be unknown."""

    return () if root is None else tuple(root / part for part in relative)


# The QCD+QED loop UFOs as UFOmodeldb ships them (the ground truth, user
# 2026-10-30) with the user's model patch of UFOmodeldb issue 1 (user rule
# 2026-10-30 ~12:45; README.txt there; = the shipped copies). The unpatched
# copies are in ufodb-6ca4b120/, MG5_aMC's online-database ones one level up.
_UFODB = "reference/models/ufodb-6ca4b120-patched"
# fixture name -> (overriding environment variables, default candidates)
_EXTERNAL_INPUTS: dict[str, tuple[tuple[str, ...], tuple[Path, ...]]] = {
    "loop_sm_ufo": (
        ("PYAMPLICOL_NLO_UFO_REFERENCE",),
        (
            TESTS.parent / _MG5 / "models/loop_sm",
            *_under(_MAIN_CHECKOUT, _MG5 / "models/loop_sm"),
            *_under(_WORKSPACE, "reference/sources/MG5_aMC_v3_7_0/models/loop_sm"),
        ),
    ),
    "loop_gmu_ufo": (
        ("PYAMPLICOL_EW_UFO_REFERENCE", "PYAMPLICOL_EW_UFO"),
        _under(_WORKSPACE, f"{_UFODB}/loop_qcd_qed_sm_Gmu_py3"),
    ),
    "loop_qcd_qed_sm_ufo": (
        ("PYAMPLICOL_EW_NLO_UFO_REFERENCE",),
        _under(_WORKSPACE, f"{_UFODB}/loop_qcd_qed_sm"),
    ),
    # Controls: UFOmodeldb 6ca4b120 without the issue-1 model patch.
    "loop_gmu_ufodb_unpatched_ufo": (
        ("PYAMPLICOL_EW_UFO_UNPATCHED_REFERENCE",),
        _under(_WORKSPACE, "reference/models/ufodb-6ca4b120/loop_qcd_qed_sm_Gmu_py3"),
    ),
    "loop_qcd_qed_sm_ufodb_unpatched_ufo": (
        ("PYAMPLICOL_EW_NLO_UFO_UNPATCHED_REFERENCE",),
        _under(_WORKSPACE, "reference/models/ufodb-6ca4b120/loop_qcd_qed_sm"),
    ),
    "sgluon_nlo_ufo": (
        ("PYAMPLICOL_SGLUON_UFO_REFERENCE",),
        _under(_WORKSPACE, "reference/models/bsm-candidates/sgluons_NLO_py3"),
    ),
    # The Higgs Characterisation NLO UFO as downloaded (M-FINAL MF-GM-HC;
    # sha256 in tools/loop_reference/hc_nlo_x0.json).
    "hc_nlo_x0_ufo": (
        ("PYAMPLICOL_HC_UFO_REFERENCE",),
        _under(_WORKSPACE, "reference/sources/hc-nlo-x0/extract/HC_NLO_X0_UFO"),
    ),
    "madgraph_source": (
        ("PYAMPLICOL_MADGRAPH_SOURCE",),
        _under(_WORKSPACE, "reference/sources/MG5_aMC_v3_7_0"),
    ),
    "saved_loop_fixtures": (
        ("PYAMPLICOL_LOOP_SAVED_FIXTURES",),
        _under(_WORKSPACE, "fixtures"),
    ),
    # An installed wheel (target directory with bin/rusticol-config) is a
    # release-build product of the checkout under test, so it has no default.
    "installed_virtual_bundle": (("PYAMPLICOL_VIRTUAL_BUNDLE_INSTALL",), ()),
}
# Inputs a run provides only on request: missing, they skip loop tests too.
_OPT_IN_INPUTS = frozenset({"installed_virtual_bundle"})


def _resolved_input(name: str) -> Path | None:
    """The input's path, ``None`` when no default candidate exists; a
    configured path that does not exist fails."""

    variables, candidates = _EXTERNAL_INPUTS[name]
    for variable in variables:
        if configured := os.environ.get(variable):
            path = Path(configured)
            if not path.exists():
                pytest.fail(f"{variable}={configured} does not exist")
            return path
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _missing_input(name: str) -> str:
    variables, candidates = _EXTERNAL_INPUTS[name]
    if name in _OPT_IN_INPUTS:
        return f"{name} unavailable; set {variables[0]}"
    looked = ", ".join(str(path) for path in dict.fromkeys(candidates)) or "none"
    return (
        f"{name} unavailable: set {' or '.join(variables)}; default locations "
        f"looked at: {looked} ({_WORKSPACE_ORIGIN})"
    )


class MissingLoopInput(pytest.skip.Exception):
    """A missing external input (other than an opt-in one) or staged native
    runtime: other tests skip with it, a loop test fails with it
    (``pytest_runtest_makereport``), whether its fixture raised it or a
    session fixture's cached result re-raised it."""


def _external_input(name: str) -> Path:
    path = _resolved_input(name)
    if path is None:
        if name in _OPT_IN_INPUTS:
            pytest.skip(_missing_input(name))
        raise MissingLoopInput(_missing_input(name))
    return path


def _input_fixture(name: str):
    def fixture() -> Path:
        return _external_input(name)

    fixture.__name__ = name
    fixture.__doc__ = f"Path of the external {name} input."
    return pytest.fixture(scope="session", name=name)(fixture)


for _name in _EXTERNAL_INPUTS:
    globals()[_name] = _input_fixture(_name)


@pytest.fixture(scope="session")
def loop_sm_compiled_model(loop_sm_ufo, tmp_path_factory):
    """loop_sm compiled once per session with the checked scalar-arity view."""

    from dataclasses import replace

    from pyamplicol.models.loading import compile_model_source
    from tools.loop_reference.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )

    root = tmp_path_factory.mktemp("loop-sm-model")
    model = compile_model_source(loop_sm_ufo, cache_dir=root / "model-cache")
    model = replace(
        model,
        raw_loop_model=prepare_loop_sm_scalar_declarations(
            model.raw_loop_model
        ).derived,
        _serialized_path=None,
        _prepared_bundle=None,
    )
    return model.write(root / "loop-sm.json")


@pytest.fixture(scope="session")
def loop_sm_colour_model(loop_sm_ufo, tmp_path_factory):
    """loop_sm with the committed explicit QCD loop/colour declarations."""

    import importlib.util

    script = TESTS.parent / "examples/loop/source_colour/prepare_model.py"
    spec = importlib.util.spec_from_file_location("_loop_colour_example", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path_factory.mktemp("loop-sm-colour-model")
    return module.prepare(loop_sm_ufo, root / "loop-sm-colour.json")


@cache
def _staged_loop_native() -> tuple[object | None, str | None]:
    """The staged loop core, or why it is unavailable (once per session)."""

    from pyamplicol.api.errors import DependencyError
    from pyamplicol.loops.native import load_loop_native

    try:
        return load_loop_native(), None
    except DependencyError as error:
        return None, f"staged native loop runtime unavailable: {error}"


@pytest.fixture(scope="session")
def loop_native():
    """The loop core registered in the staged ``pyamplicol._rusticol``."""

    native, missing = _staged_loop_native()
    if native is None:
        raise MissingLoopInput(missing)
    return native


@pytest.fixture(scope="session")
def loop_reference_path():
    """Path of a committed provider reference; a missing file is a failure."""

    def resolve(name: str) -> Path:
        path = LOOP_REFERENCES / f"{name}.json"
        if not path.is_file():
            pytest.fail(f"committed loop reference {path} is missing")
        return path

    return resolve


@pytest.fixture(scope="session")
def loop_reference(loop_reference_path):
    """Load a committed provider reference from ``tests/fixtures/loop``."""

    return lambda name: json.loads(loop_reference_path(name).read_text())


@pytest.fixture(scope="session")
def saved_loop_fixture(saved_loop_fixtures):
    """Resolve one saved PACBIN/artifact control below the fixture root."""

    def resolve(relative: str) -> Path:
        path = saved_loop_fixtures / relative
        if not path.exists():
            pytest.fail(f"saved loop fixture {path} is missing")
        return path

    return resolve


def pytest_configure(config):
    """A persistent model cache per test tier, never the user's ~/.cache.

    Unless ``PYAMPLICOL_CACHE_DIR`` is set, it points to a directory of
    pytest's cache in this checkout, one per marker expression (the tier):
    compiled models and CT-lowered model IR (``models.loading``) and the
    lowered one-loop model programs (``loops.preparation_cache``, its
    ``models/loop-preparations``) are reused across the runs of one tier.
    The cache keys hold the code identity, so a code change pays a cold
    preparation once per tier.
    Without pytest's cache provider (``-p no:cacheprovider``) the directory
    is a fresh temporary one, removed when the session ends.
    """

    if os.environ.get("PYAMPLICOL_CACHE_DIR"):
        return
    import re
    import shutil
    import tempfile

    expression = config.getoption("markexpr", "") or ""
    tier = re.sub(r"[^A-Za-z0-9_]+", "-", expression).strip("-")[:80] or "all"
    cache = getattr(config, "cache", None)
    if cache is not None:
        root = Path(cache.mkdir(f"pyamplicol-{tier}"))
    else:
        root = Path(tempfile.mkdtemp(prefix="pyamplicol-cache-"))

        def remove(root: Path = root) -> None:
            os.environ.pop("PYAMPLICOL_CACHE_DIR", None)
            shutil.rmtree(root, ignore_errors=True)

        config.add_cleanup(remove)
    os.environ["PYAMPLICOL_CACHE_DIR"] = str(root)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    """A loop test (``LOOP_MARKERS``) whose external input (other than an
    opt-in one) or staged native runtime is missing fails, naming what is
    missing and where it was looked for (``MissingLoopInput``,
    ``_missing_input``): as a setup error when one of its fixtures needs the
    input, as a failure when the test asks for it at run time
    (``request.getfixturevalue``), also when an earlier test's skip on the
    same session fixture is re-raised from pytest's fixture cache. A checkout
    at another location whose inputs cannot be derived must not pass with
    its loop tests skipped. Loop tests skipped at collection (no
    ``SYMBOLICA_LICENSE``, ``pytest_collection_modifyitems``) raise pytest's
    own skip and stay skipped (with ``REQUIRE_LICENSE`` they fail instead,
    ``pytest_runtest_setup``)."""

    report = yield
    if (
        call.excinfo is not None
        and call.excinfo.errisinstance(MissingLoopInput)
        and any(marker in item.keywords for marker in LOOP_MARKERS)
    ):
        report.outcome = "failed"
        report.longrepr = call.excinfo.value.msg
    if call.excinfo is not None and call.excinfo.errisinstance(ImportError):
        # MadGraph7: a test needing upstream-only tooling skips (see below).
        name = _upstream_only_module(call.excinfo.value)
        if name is not None:
            path, lineno, _ = item.reportinfo()
            report.outcome = "skipped"
            report.longrepr = (
                str(path),
                (lineno or 0) + 1,
                f"Skipped: needs {name}, kept in the upstream pyAmpliCol repository",
            )
    return report


def _license_required() -> bool:
    """Whether a missing ``SYMBOLICA_LICENSE`` fails the loop tests instead of
    skipping them (``REQUIRE_LICENSE`` = ``1``; unset, empty or ``0``: skip).
    A licensed CI job sets it, so a missing, rotated or expired secret cannot
    turn the job green with its loop tests skipped."""

    value = os.environ.get(REQUIRE_LICENSE, "")
    if value not in ("", "0", "1"):
        raise pytest.UsageError(f"{REQUIRE_LICENSE}={value!r}: expected 0 or 1")
    return value == "1"


_UNLICENSED = pytest.StashKey[str]()


@cache
def _library_key_accepted() -> bool:
    """MadGraph7: whether Symbolica accepts pyAmpliCol's own library key, so
    that the loop tests need no ``SYMBOLICA_LICENSE``. Checked once, in a
    child interpreter without ``SYMBOLICA_LICENSE`` (an unlicensed Symbolica
    may abort the interpreter that asks)."""

    import subprocess
    import sys

    code = (
        "import sys; sys.path[:0] = sys.argv[1:]; import pyamplicol; "
        "from pyamplicol.licensing import detect_symbolica_license; "
        "print(detect_symbolica_license(suggest=False).licensed)"
    )
    environment = dict(os.environ)
    environment.pop("SYMBOLICA_LICENSE", None)
    try:
        result = subprocess.run(
            [sys.executable, "-c", code, *sys.path],
            capture_output=True,
            text=True,
            env=environment,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    lines = result.stdout.strip().splitlines()
    return result.returncode == 0 and bool(lines) and lines[-1] == "True"


# MadGraph7: the one-loop reference and benchmark tooling (the code under
# tools/loop_reference and tools/loop_benchmark; their JSON data is kept),
# tools/loop_v2_ward.py, and the upstream test modules built on that tooling
# stay in the upstream pyAmpliCol repository
# (.github/scripts/import_pyamplicol.sh). A test that imports one of exactly
# these modules at run time skips with that reason; any other missing module
# is an error. The helpers restored under tools/loop_reference (listed in
# import_pyamplicol.sh) are not upstream-only: a failure to import them is an
# error too.
_UPSTREAM_ONLY_PACKAGES = ("tools.loop_reference", "tools.loop_benchmark")
_RESTORED_HELPERS = frozenset(
    {
        "tools.loop_reference.exact_points",
        "tools.loop_reference.feynkit_loop_graphs",
        "tools.loop_reference.legacy_capture",
        "tools.loop_reference.levi_civita_aloha",
        "tools.loop_reference.model_compatibility",
        "tools.loop_reference.njet",
        "tools.loop_reference.njet.njet_fixture",
        "tools.loop_reference.ufo_source_oracle",
    }
)
_UPSTREAM_ONLY_TOOLS = frozenset({"tools.loop_v2_ward"})
_UPSTREAM_ONLY_TESTS = frozenset(
    {
        "loop_mqcd_tiers",
        "test_loop_benchmark",
        "test_loop_reference_mqcd",
        "test_loop_reference_msm",
        "test_loop_v2_2to6",
        "test_loop_v2_companion",
        "test_loop_v2_corr_recola",
        "test_loop_v2_dec_checks",
        "test_loop_v2_dec_default",
        "test_loop_v2_dec_manifest",
        "test_loop_v2_dec_nightly",
        "test_loop_v2_li_campaign",
        "test_loop_v2_li_manifest",
        "test_loop_v2_master_census",
        "test_loop_v2_mqcd_default",
        "test_loop_v2_mqcd_manifest",
        "test_loop_v2_mqcd_nightly",
        *(f"test_loop_v2_mqcd_nightly_{index}" for index in range(2, 8)),
        "test_loop_v2_msm_default",
        "test_loop_v2_msm_manifest",
        "test_loop_v2_msm_nightly",
        *(f"test_loop_v2_msm_nightly_{index}" for index in range(2, 24)),
        "test_loop_v2_msm_smokes",
        *(f"test_loop_v2_msm_smokes_{index}" for index in range(2, 5)),
    }
)


def _upstream_only_module(error: BaseException) -> str | None:
    name = getattr(error, "name", None) or ""
    if name in _UPSTREAM_ONLY_TOOLS:
        return name
    if (
        name.startswith(tuple(f"{package}." for package in _UPSTREAM_ONLY_PACKAGES))
        and name not in _RESTORED_HELPERS
    ):
        return name
    if name.rpartition(".")[2] in _UPSTREAM_ONLY_TESTS:
        return name
    return None


def pytest_collection_modifyitems(config, items):
    expression = config.getoption("markexpr", "") or ""
    if DEFERRED_MARKER not in expression:
        deferred = [item for item in items if DEFERRED_MARKER in item.keywords]
        if deferred:
            config.hook.pytest_deselected(items=deferred)
            items[:] = [item for item in items if DEFERRED_MARKER not in item.keywords]
    required = _license_required()
    if os.environ.get("SYMBOLICA_LICENSE"):
        return
    if (
        not any(marker in item.keywords for item in items for marker in LOOP_MARKERS)
        or _library_key_accepted()
    ):
        return
    # An unlicensed Symbolica aborts the interpreter on this class of host, so
    # loop tests are skipped (or, with REQUIRE_LICENSE, failed in
    # pytest_runtest_setup) before any of them can touch it.
    reason = "loop tests require SYMBOLICA_LICENSE"
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if any(marker in item.keywords for marker in LOOP_MARKERS):
            if required:
                item.stash[_UNLICENSED] = f"{reason} ({REQUIRE_LICENSE}=1)"
            else:
                item.add_marker(skip)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """A loop test without ``SYMBOLICA_LICENSE`` under ``REQUIRE_LICENSE``
    fails before any of its fixtures is set up (so it cannot touch an
    unlicensed Symbolica)."""

    if (reason := item.stash.get(_UNLICENSED, None)) is not None:
        pytest.fail(reason, pytrace=False)
