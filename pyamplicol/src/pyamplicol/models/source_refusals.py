# SPDX-License-Identifier: 0BSD
"""Listed refusals of model sources (tree and one-loop alike).

A UFO directory the loader cannot import for a listed reason is refused with
a typed error (``ModelSourceUnsupported``) whose ``key`` names the rule
(``MODEL_SOURCE_REFUSALS``), instead of the Python exception the UFO's own
code raised. The rules read the source's code, never a model name:

* ``ufo-python2``: the UFO is Python-2 code (MG5_aMC's older model-database
  downloads, e.g. the G_mu electroweak loop model): one of its modules is not
  Python-3 source (``print`` statements, ``except E, e``, ...), or its import
  failed on a Python-2-only name (``dict.iteritems``, ``xrange``,
  ``unicode``, ...). MG5_aMC 3 converts such a model when it imports it;
  pyAmpliCol does not rewrite a user's model.
* ``ufo-value-writes``: the UFO writes a declared value after its constructor
  (e.g. the module-level ``eCoup_a0_UV_EW.value[-1] += '...'`` of the
  UFOmodeldb issue-1 patch of loop_qcd_qed_sm_Gmu). ufo-model-loader then
  certifies none of the model's values as its exact source, and the one-loop
  engine reads exact values only (raised where it first needs one).
"""

from __future__ import annotations

import builtins
from dataclasses import dataclass
from pathlib import Path

from pyamplicol.api.errors import ModelError


@dataclass(frozen=True)
class ModelSourceRefusal:
    """A listed model source the product refuses: ``key`` names it,
    ``feature`` says what is refused and ``remedy`` what to use instead."""

    key: str
    feature: str
    remedy: str


MODEL_SOURCE_REFUSALS = (
    ModelSourceRefusal(
        "ufo-python2",
        "a UFO model written in Python 2",
        "convert the model to Python 3 (MG5_aMC 3: 'convert model <path>', "
        "which edits it in place) or select a UFO model pyAmpliCol ships by "
        "its name (model.source)",
    ),
    ModelSourceRefusal(
        "ufo-value-writes",
        "a UFO model that writes a declared value after its constructor, for "
        "one-loop requests",
        "write the value in its constructor (tools/developer/"
        "fold_ufo_value_appends.py folds module-level string appends to "
        "Laurent terms, as in the shipped loop_qcd_qed_sm_Gmu) or select a UFO "
        "model pyAmpliCol ships by its name (model.source)",
    ),
)
_REFUSALS = {refusal.key: refusal for refusal in MODEL_SOURCE_REFUSALS}


class ModelSourceUnsupported(ModelError, ValueError):
    """A listed model source the product refuses (``MODEL_SOURCE_REFUSALS``);
    ``key`` is its key. The message starts with the key and names the
    remedy. A ``ModelError`` (and a ``ValueError``, as the import errors it
    replaces)."""

    def __init__(self, key: str, message: str) -> None:
        refusal = _REFUSALS.get(key)
        if refusal is None:
            raise ValueError(f"unknown model source refusal {key!r}")
        super().__init__(f"{key}: {message}. Instead: {refusal.remedy}")
        self.key = key


def listed_model_source_refusal_keys() -> tuple[str, ...]:
    """The keys of ``MODEL_SOURCE_REFUSALS``."""

    return tuple(refusal.key for refusal in MODEL_SOURCE_REFUSALS)


# Names Python 3 removed: dict's iterator and membership methods, and
# builtins. An AttributeError or NameError of a UFO import naming one of them
# is Python-2 code.
_PYTHON2_ATTRIBUTES = frozenset(
    {"iteritems", "iterkeys", "itervalues", "has_key", "viewitems", "viewkeys"}
)
_PYTHON2_BUILTINS = frozenset(
    {
        "basestring",
        "cmp",
        "execfile",
        "long",
        "raw_input",
        "reduce",
        "unichr",
        "unicode",
        "xrange",
    }
) - frozenset(dir(builtins))


def python2_source_evidence(source: Path) -> str | None:
    """The first module of the UFO directory ``source`` that is not Python-3
    source, as evidence that the UFO is Python 2, or None. Compiles, never
    imports: the dry run uses it on a source it does not load."""

    if not source.is_dir():
        return None
    # The modules an import can load: AppleDouble (``._x.py``) and other
    # files whose stem is no identifier are not modules.
    for path in sorted(source.glob("*.py")):
        if not path.stem.isidentifier():
            continue
        try:
            compile(path.read_bytes(), path.name, "exec", dont_inherit=True)
        except SyntaxError as syntax:
            return f"{path.name} is not Python 3 source ({syntax.msg})"
    return None


def python2_evidence(source: Path, error: BaseException) -> str | None:
    """Why the UFO directory ``source``, whose import raised ``error``, is
    Python-2 code, or None when nothing says so: the first of its modules
    that is not Python-3 source (``python2_source_evidence``), else ``error``
    itself when it names a Python-2-only attribute or builtin."""

    evidence = python2_source_evidence(source)
    if evidence is not None:
        return evidence
    name = getattr(error, "name", None)
    if (isinstance(error, AttributeError) and name in _PYTHON2_ATTRIBUTES) or (
        isinstance(error, NameError)
        and not isinstance(error, UnboundLocalError)
        and name in _PYTHON2_BUILTINS
    ):
        return f"its import failed on the Python-2 name {name!r} ({error})"
    return None


def _refusal(source: Path, evidence: str) -> ModelSourceUnsupported:
    from pyamplicol._internal.packaged_models import packaged_ufo_models

    shipped = ", ".join(packaged_ufo_models()) or "none"
    return ModelSourceUnsupported(
        "ufo-python2",
        f"the UFO model {source.name!r} is Python 2: {evidence} (UFO models "
        f"shipped with pyAmpliCol: {shipped})",
    )


def refuse_python2_source(source: Path) -> None:
    """Raise ``ufo-python2`` when a module of the UFO directory ``source`` is
    not Python-3 source (``python2_source_evidence``; no import)."""

    evidence = python2_source_evidence(source)
    if evidence is not None:
        raise _refusal(source, evidence)


# ufo-model-loader's reason for a value it does not certify because the model
# writes values after their constructors (``ufo_model_loader.raw``).
_VALUE_WRITES = "model contains potentially dynamic value writes"


def refuse_value_writes(model: str, source: dict) -> None:
    """Raise ``ufo-value-writes`` when ``source`` (a raw record's
    ``value_source`` or ``expression_source``) is uncertified because the UFO
    ``model`` writes values after their constructors; return otherwise."""

    if not source.get("exact", False) and source.get("reason") == _VALUE_WRITES:
        raise ModelSourceUnsupported(
            "ufo-value-writes",
            f"the UFO model {model!r} writes declared values after their "
            "constructors, so none of its values is certified as its exact "
            "source, which one-loop requests need",
        )


def refuse_python2_ufo(source: Path, error: BaseException) -> None:
    """Raise ``ufo-python2`` (from ``error``) when the UFO directory
    ``source``, whose import raised ``error``, is Python-2 code; return
    otherwise (the caller re-raises ``error``)."""

    evidence = python2_evidence(source, error)
    if evidence is not None:
        raise _refusal(source, evidence) from error


__all__ = [
    "MODEL_SOURCE_REFUSALS",
    "ModelSourceRefusal",
    "ModelSourceUnsupported",
    "listed_model_source_refusal_keys",
    "python2_evidence",
    "python2_source_evidence",
    "refuse_python2_source",
    "refuse_python2_ufo",
    "refuse_value_writes",
]
