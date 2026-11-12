# SPDX-License-Identifier: 0BSD
"""A user UFO model's own Python warnings (ux-clean): Python >= 3.12 raises a
``SyntaxWarning`` (3.11: a ``DeprecationWarning``, hidden by default) for an
invalid escape such as ``'\\lambda'`` in a non-raw literal when pyAmpliCol
imports or compiles the model. Such warnings are returned as data
(``ResultWarning(source="ufo-model")``) and never printed; warnings of other
files raised in the same scope pass through unchanged."""

from __future__ import annotations

import importlib
import logging
import shutil
import sys
import warnings
from importlib import resources
from pathlib import Path

import pytest

from pyamplicol._internal import symbolica_log
from pyamplicol.api.result_warnings import ResultWarning

ESCAPE_CATEGORY = SyntaxWarning if sys.version_info >= (3, 12) else DeprecationWarning
# The literal as a model file spells it: a lone backslash before 'l'.
BAD_LITERAL = "texname = '" + "\\" + "lambda_{HV}'\n"


def _model(tmp_path: Path, name: str) -> Path:
    root = tmp_path / name
    root.mkdir()
    (root / "__init__.py").write_text("from . import parameters\n", encoding="utf-8")
    (root / "parameters.py").write_text("x = 1\n" + BAD_LITERAL, encoding="utf-8")
    return root


def _import(root: Path) -> None:
    """Import ``root`` as the UFO loader does (no bytecode: compiled from
    source on every import)."""

    saved_path, saved_bytecode = sys.path[:], sys.dont_write_bytecode
    sys.path.insert(0, str(root.parent))
    sys.dont_write_bytecode = True
    try:
        importlib.import_module(root.name)
    finally:
        for name in [n for n in sys.modules if n.split(".")[0] == root.name]:
            del sys.modules[name]
        sys.path[:] = saved_path
        sys.dont_write_bytecode = saved_bytecode


def test_the_escape_warning_is_raised_by_the_import(tmp_path: Path) -> None:
    """Under an ``error`` filter (the caller's filters stay in force inside
    the scope) the synthetic model's import fails: the warning is real."""

    root = _model(tmp_path, "escmodel_error")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with (
            pytest.raises((SyntaxError, ESCAPE_CATEGORY)),
            symbolica_log.ufo_source_warnings(root),
        ):
            _import(root)


def test_model_warnings_are_captured_and_others_re_emitted(tmp_path: Path) -> None:
    root = _model(tmp_path, "escmodel_capture")
    with warnings.catch_warnings(record=True) as shown:
        # 'always' so that 3.11's DeprecationWarning is not filtered out.
        warnings.simplefilter("always")
        with (
            symbolica_log.capture("generate") as sink,
            symbolica_log.ufo_source_warnings(root),
        ):
            _import(root)
            warnings.warn("not the model's", UserWarning, stacklevel=1)
    assert [str(item.message) for item in shown] == ["not the model's"]
    (captured,) = sink.warnings()
    assert isinstance(captured, ResultWarning)
    assert captured.source == "ufo-model"
    assert captured.kind == ESCAPE_CATEGORY.__name__
    assert captured.where == "generate"
    assert captured.message.startswith("parameters.py:2: ")
    assert "invalid escape sequence" in captured.message
    assert captured.to_json()["source"] == "ufo-model"
    assert ResultWarning.from_json(captured.to_json()) == captured


@pytest.mark.skipif(
    sys.version_info < (3, 12), reason="Python 3.11 hides the warning by default"
)
def test_default_filters_print_nothing(tmp_path: Path, capfd) -> None:
    root = _model(tmp_path, "escmodel_default")
    with (
        symbolica_log.capture("check") as sink,
        symbolica_log.ufo_source_warnings(root),
    ):
        _import(root)
    captured = capfd.readouterr()
    assert "SyntaxWarning" not in captured.err + captured.out
    assert [item.kind for item in sink.warnings()] == ["SyntaxWarning"]


def test_outside_a_capture_they_are_logged_at_debug(tmp_path: Path, caplog) -> None:
    root = _model(tmp_path, "escmodel_debug")
    caplog.set_level(logging.DEBUG, logger="pyamplicol.models")
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("always")
        with symbolica_log.ufo_source_warnings(root):
            _import(root)
    assert shown == []
    assert any(
        "escmodel_debug" in record.getMessage()
        and "invalid escape sequence" in record.getMessage()
        for record in caplog.records
    )


def test_the_loader_path_returns_them(tmp_path: Path) -> None:
    """Through pyAmpliCol's own UFO loading (a copy of the bundled
    ``scalars`` UFO with one TeX name made an invalid escape)."""

    pytest.importorskip("ufo_model_loader")
    from pyamplicol.models.loop_import import load_raw_loop_model

    shipped = Path(
        str(
            resources.files("pyamplicol").joinpath("assets", "models", "ufo", "scalars")
        )
    )
    root = tmp_path / "scalars_escape"
    shutil.copytree(shipped, root)
    parameters = root / "parameters.py"
    text = parameters.read_text(encoding="utf-8")
    assert text.count("texname='\\\\lambda'") == 1
    parameters.write_text(
        text.replace("texname='\\\\lambda'", "texname='\\lambda'"), encoding="utf-8"
    )
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("always")
        with symbolica_log.capture("generate") as sink:
            load_raw_loop_model(root)
    assert not [w for w in shown if "invalid escape" in str(w.message)]
    found = [w for w in sink.warnings() if w.source == "ufo-model"]
    assert len(found) == 1
    assert found[0].message.startswith("parameters.py:23: ")
    assert "invalid escape sequence" in found[0].message
