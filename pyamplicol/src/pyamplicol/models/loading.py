# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import sys
import threading
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from copy import copy
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .external import CompiledUFOModel
    from .loop_import import RawLoopModel
    from .prepared import PreparedModelBundle

from .._internal.versions import (
    COMPILED_MODEL_SCHEMA_VERSION,
    SYMBOLICA_SERIALIZATION_ABI,
    package_version,
)
from .contact_decomposition import (
    CONTACT_DECOMPOSITION_ALGORITHM,
    CONTACT_DECOMPOSITION_ALGORITHM_VERSION,
)
from .contracts import (
    DEFAULT_FEYNMAN_PROPAGATOR_SOURCE,
    MODEL_GAUGES,
    MODEL_SUPPLIED_PROPAGATOR_SOURCE,
    PROPAGATOR_SOURCE_FIELD,
    TENSOR_ORDERING_CONTRACT_VERSION,
    CompiledModelIR,
    ModelGauge,
    TreeKernelVariant,
)

COMPILED_MODEL_KIND = "pyamplicol-compiled-model"
MODEL_COMPILER_VERSION = 16
BUILTIN_SM_ALIASES = frozenset(("builtin_sm", "built-in-sm"))
DEFAULT_MODEL_RESTRICTION = "default"
NO_MODEL_RESTRICTION = "none"
SANITIZED_MODEL_ENVIRONMENT_PREFIXES = (
    "UFO_SCALARS_MODEL_",
    "UFO_GRAVITY_MODEL_",
)
_MODEL_LOAD_LOCK = threading.RLock()

SUPPORTED_FUNCTION_ARITIES = {
    name: frozenset({1})
    for name in (
        "Theta",
        "abs",
        "acos",
        "acosh",
        "acsc",
        "asec",
        "asin",
        "asinh",
        "atan",
        "atanh",
        "complexconjugate",
        "conj",
        "cos",
        "cosh",
        "csc",
        "exp",
        "im",
        "log",
        "log10",
        "re",
        "reglog",
        "reglogm",
        "reglogp",
        "sec",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
    )
}
SUPPORTED_FUNCTION_ARITIES.update(
    {
        "complex": frozenset({2}),
        "cond": frozenset({3}),
        "if": frozenset({3}),
        "pow": frozenset({2}),
        # MG5_aMC's NLO UFO scalar helpers next to cond/reglog/reglogp/reglogm
        # (the NLOCT function library of the QCD+QED loop models): the
        # complex-mass-scheme selectors recms(cms, z) / crecms(cms, z), the
        # principal argument arg(z), the principal square root regsqrt(z) and
        # the branch-continued logarithm grreglog(logswitch, z1, z2). The
        # QCD+QED loop models declare them and call them in counterterm
        # expressions only, which the loop counterterm program lowers
        # (loops.v2.ct_recursion); their tree models never call them.
        "arg": frozenset({1}),
        "crecms": frozenset({2}),
        "grreglog": frozenset({3}),
        "recms": frozenset({2}),
        "regsqrt": frozenset({1}),
    }
)
SUPPORTED_FUNCTIONS = frozenset(SUPPORTED_FUNCTION_ARITIES)

UFO_TENSOR_HEADS = frozenset(
    {
        "C",
        "Epsilon",
        "EpsilonBar",
        "Gamma",
        "Gamma5",
        "Identity",
        "IdentityL",
        "K6",
        "K6Bar",
        "Metric",
        "P",
        "PSlash",
        "ProjM",
        "ProjP",
        "Sigma",
        "T",
        "T6",
        "d",
        "dummy",
        "f",
        "idx",
    }
)


def _require_ufo_model_loader() -> None:
    """Reject loaders predating the supported model serialization API."""

    installed = _distribution_version("ufo-model-loader", "not installed")
    # The loader publishes three-component release versions. Development and
    # prerelease builds are not substitutes for the required release interface.
    release = re.fullmatch(
        r"(\d+)\.(\d+)\.(\d+)(?:\.post\d+)?(?:\+[\w.-]+)?", installed
    )
    if release is None or tuple(map(int, release.groups())) < (1, 0, 0):
        raise RuntimeError(
            "UFO model import requires ufo-model-loader release 1.0.0 or newer "
            f"for the supported model serialization API; found {installed}. "
            "Install the updated loader before importing a UFO model."
        )


@dataclass(frozen=True, slots=True)
class ModelCompileOptions:
    """How a model source is compiled.

    ``gauge`` is the gauge of the massive vectors (``contracts.ModelGauge``):
    ``"unitary"`` (the default) or ``"feynman"``, the 't Hooft-Feynman gauge
    with explicit Goldstones, admissible only for a UFO model whose ``gauge``
    declaration lists 1 (``declared_model_gauges``). The source is loaded and
    compiled without it (``gauge_independent``: the two gauges share every
    compiled record but the Goldstone dispositions, so the model cache holds
    one compile) and the gauge is then applied (``with_model_gauge``), which
    records it as ``canonical_payload`` does.
    """

    restriction: str = DEFAULT_MODEL_RESTRICTION
    simplify: bool = True
    gauge: ModelGauge = "unitary"

    def __post_init__(self) -> None:
        restriction_path = Path(self.restriction).expanduser()
        if not self.restriction or (
            any(character.isspace() for character in self.restriction)
            and not restriction_path.is_file()
        ):
            raise ValueError(
                "model restriction must be default, none, or a restriction name"
            )
        if not isinstance(self.simplify, bool):
            raise TypeError("model simplification must be a boolean")
        if self.gauge not in MODEL_GAUGES:
            raise ValueError(
                f"model gauge must be one of {', '.join(MODEL_GAUGES)}, "
                f"not {self.gauge!r}"
            )

    def gauge_independent(self) -> ModelCompileOptions:
        """These options without the gauge: the compile the model cache
        holds, to which ``with_model_gauge`` applies ``gauge``."""

        return replace(self, gauge="unitary")

    def canonical_payload(self) -> dict[str, object]:
        restriction_path = Path(self.restriction).expanduser()
        if restriction_path.is_file():
            restriction: object = {
                "kind": "file",
                "sha256": hashlib.sha256(restriction_path.read_bytes()).hexdigest(),
            }
        else:
            restriction = {"kind": "name", "value": self.restriction}
        return {
            "restriction": restriction,
            "simplify": self.simplify,
            # Only off the default, so unitary-gauge compiles keep their record
            # and cache key.
            **({} if self.gauge == "unitary" else {"gauge": self.gauge}),
        }


@dataclass(frozen=True)
class ModelCompatibilityIssue:
    severity: str
    code: str
    message: str
    context: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "context": self.context,
        }

    @staticmethod
    def from_dict(payload: Mapping[str, object]) -> ModelCompatibilityIssue:
        return ModelCompatibilityIssue(
            severity=str(payload["severity"]),
            code=str(payload["code"]),
            message=str(payload["message"]),
            context=str(payload.get("context", "")),
        )


@dataclass(frozen=True)
class CompiledModel:
    """A compiled model.

    ``phase_timings`` and ``conversion_seconds`` are the wall-clock durations
    of the compile that produced this object in this process; they are not
    part of the payload (``to_dict``), so equal compiles serialise, digest
    and cache to equal bytes, and a model loaded from a file has none
    (``{}`` and ``0.0``)."""

    source: Mapping[str, object]
    producer: Mapping[str, object]
    model: Mapping[str, object]
    ir: CompiledModelIR
    parameter_defaults: Mapping[str, tuple[float, float]]
    capabilities: Mapping[str, object]
    issues: tuple[ModelCompatibilityIssue, ...]
    phase_timings: Mapping[str, float]
    conversion_seconds: float
    raw_loop_model: RawLoopModel | None = None
    _serialized_path: Path | None = field(default=None, compare=False, repr=False)
    _prepared_bundle: PreparedModelBundle | None = field(
        default=None,
        compare=False,
        repr=False,
    )

    @property
    def name(self) -> str:
        return str(self.model.get("name", "unnamed-model"))

    @property
    def schema_version(self) -> int:
        return COMPILED_MODEL_SCHEMA_VERSION

    @property
    def model_compiler_version(self) -> int:
        return MODEL_COMPILER_VERSION

    @property
    def supported(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def prepared_bundle(self) -> PreparedModelBundle | None:
        return self._prepared_bundle

    @property
    def prepared_backend(self) -> str | None:
        return None if self._prepared_bundle is None else self._prepared_bundle.backend

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "kind": COMPILED_MODEL_KIND,
            "schema_version": COMPILED_MODEL_SCHEMA_VERSION,
            "model_compiler_version": MODEL_COMPILER_VERSION,
            "source": dict(self.source),
            "producer": dict(self.producer),
            "model": dict(self.model),
            "ir": self.ir.to_dict(),
            "parameter_defaults": {
                name: [value[0], value[1]]
                for name, value in sorted(self.parameter_defaults.items())
            },
            "capabilities": dict(self.capabilities),
            "issues": [issue.to_dict() for issue in self.issues],
        }
        if self.raw_loop_model is not None:
            payload["raw_loop_model"] = self.raw_loop_model.to_dict()
        return payload

    def write(self, path: Path) -> Path:
        output = _compiled_model_output_path(path)
        if self._serialized_path is None or not self._serialized_path.is_file():
            _atomic_write_json(output, self.to_dict(), compact=True)
        else:
            _atomic_copy(self._serialized_path, output)
        return output

    def write_parameter_card(self, path: Path) -> Path:
        _atomic_write_json(
            path,
            {
                name: [value[0], value[1]]
                for name, value in sorted(self.parameter_defaults.items())
            },
        )
        return path

    @staticmethod
    def from_dict(
        payload: Mapping[str, object],
        *,
        validate_fingerprint: bool = True,
        serialized_path: Path | None = None,
    ) -> CompiledModel:
        if payload.get("kind") != COMPILED_MODEL_KIND:
            raise ValueError("file is not a pyAmpliCol compiled model")
        if int(payload.get("schema_version", -1)) != COMPILED_MODEL_SCHEMA_VERSION:
            raise ValueError("compiled model schema mismatch; regenerate the model")
        if int(payload.get("model_compiler_version", -1)) != MODEL_COMPILER_VERSION:
            raise ValueError("compiled model compiler mismatch; regenerate the model")
        producer = _mapping(payload.get("producer"), "producer")
        if validate_fingerprint and producer != compiler_fingerprint():
            raise ValueError(
                "compiled model dependency fingerprint mismatch; regenerate the model"
            )
        parameter_payload = _mapping(
            payload.get("parameter_defaults"),
            "parameter_defaults",
        )
        parameters = {
            str(name): _complex_pair(value, context=f"parameter {name}")
            for name, value in parameter_payload.items()
        }
        raw_issues = payload.get("issues", ())
        if not isinstance(raw_issues, list):
            raise ValueError("compiled model issues must be a list")
        raw_loop_model = None
        if payload.get("raw_loop_model") is not None:
            from .loop_import import raw_loop_model_from_dict

            raw_loop_model = raw_loop_model_from_dict(
                _mapping(payload["raw_loop_model"], "raw_loop_model")
            )
        return CompiledModel(
            source=_mapping(payload.get("source"), "source"),
            producer=producer,
            model=_mapping(payload.get("model"), "model"),
            ir=CompiledModelIR.from_dict(_mapping(payload.get("ir"), "ir")),
            parameter_defaults=parameters,
            capabilities=_mapping(payload.get("capabilities"), "capabilities"),
            issues=tuple(
                ModelCompatibilityIssue.from_dict(_mapping(issue, "issue"))
                for issue in raw_issues
            ),
            phase_timings={},
            conversion_seconds=0.0,
            raw_loop_model=raw_loop_model,
            _serialized_path=(
                None if serialized_path is None else Path(serialized_path).resolve()
            ),
        )


def compiler_fingerprint() -> dict[str, object]:
    return {"pyamplicol": package_version(), **_compiler_code_identity()}


def _compiler_code_identity() -> dict[str, object]:
    """``compiler_fingerprint`` without the package version: what a compile's
    result depends on (the model-compiler sources, Symbolica and the
    contracts). The version check of a source checkout (``package_version``
    verifies the staged native runtime) is about a fifth of a second."""

    return {
        "ufo_model_loader": _distribution_version("ufo-model-loader", "missing"),
        "symbolica": _distribution_version("symbolica", "missing"),
        "compiled_model_schema_version": COMPILED_MODEL_SCHEMA_VERSION,
        "model_compiler_version": MODEL_COMPILER_VERSION,
        "model_compiler_sha256": _model_compiler_digest(),
        "symbolica_serialization_abi": SYMBOLICA_SERIALIZATION_ABI,
        "contact_decomposition_policy": (
            f"{CONTACT_DECOMPOSITION_ALGORITHM}-v"
            f"{CONTACT_DECOMPOSITION_ALGORITHM_VERSION}"
        ),
        "tensor_ordering_contract": (
            f"explicit-canonical-component-order-v{TENSOR_ORDERING_CONTRACT_VERSION}"
        ),
        "model_environment_policy": "sanitize-historical-scalar-options-v1",
        "symbol_namespace_policy": "model-name-and-pyamplicol-registry-v1",
    }


def detect_model_source(source: str | Path) -> tuple[str, str | Path]:
    source_kind, resolved, _payload = _detect_model_source_payload(source)
    return source_kind, resolved


def _detect_model_source_payload(
    source: str | Path,
) -> tuple[str, str | Path, dict[str, object] | None]:
    text = str(source)
    if text.lower() in BUILTIN_SM_ALIASES:
        return "built-in-sm", "built-in-sm", None
    from .._internal.packaged_models import resolve_model_path

    path = resolve_model_path(text)
    if path.is_dir():
        if not (path / "__init__.py").is_file():
            raise ValueError(f"UFO model directory has no __init__.py: {path}")
        return "ufo", path, None
    if not path.is_file():
        raise ValueError(f"model source does not exist: {path}")
    from .prepared import PREPARED_MODEL_BUNDLE_SUFFIX

    if path.name.lower().endswith(PREPARED_MODEL_BUNDLE_SUFFIX):
        return "prepared", path, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"model JSON could not be read: {path}: {exc}") from exc
    if isinstance(payload, dict) and payload.get("kind") == COMPILED_MODEL_KIND:
        return "pyamplicol", path, payload
    return "json", path, payload if isinstance(payload, dict) else None


def compile_model_source(
    source: str | Path = "BUILTIN_SM",
    *,
    restriction: str = "default",
    simplify: bool = True,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    require_supported: bool = True,
    gauge: ModelGauge | None = None,
) -> CompiledModel:
    """Compile (or load from the model cache) one model source.

    ``gauge`` selects the gauge of the massive vectors (``with_model_gauge``;
    ``None`` keeps the source's: unitary for a UFO or JSON source, the
    recorded one for a compiled model). The two gauges share every compiled
    record but the Goldstone dispositions, so the model cache holds the
    gauge-independent compile and the gauge is applied to it.
    """

    options = ModelCompileOptions(
        restriction=restriction, simplify=simplify, gauge=gauge or "unitary"
    )
    compiled = _compile_model_source(
        source,
        options=options.gauge_independent(),
        cache_dir=cache_dir,
        use_cache=use_cache,
        require_supported=require_supported,
    )
    return compiled if gauge is None else with_model_gauge(compiled, options.gauge)


def _compile_model_source(
    source: str | Path = "BUILTIN_SM",
    *,
    options: ModelCompileOptions,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    require_supported: bool = True,
) -> CompiledModel:
    """The gauge-independent compile of ``source`` (``options`` without a
    gauge, ``ModelCompileOptions.gauge_independent``)."""

    if options.gauge != "unitary":
        raise ValueError("the model source is compiled without a gauge")
    restriction, simplify = options.restriction, options.simplify
    source_kind, resolved, detected_payload = _detect_model_source_payload(source)
    if source_kind in {"pyamplicol", "prepared"}:
        if restriction != DEFAULT_MODEL_RESTRICTION or not simplify:
            raise ValueError(
                "restriction and simplification options cannot be applied to an "
                "already compiled pyAmpliCol model"
            )
        if source_kind == "prepared":
            from .prepared import load_prepared_model_bundle

            bundle = load_prepared_model_bundle(Path(resolved))
            compiled = CompiledModel.from_dict(
                bundle.compiled_model_payload(),
                validate_fingerprint=False,
            )
            compiled = replace(compiled, _prepared_bundle=bundle)
            _raise_for_unsupported(compiled, require_supported=require_supported)
            return compiled
        if detected_payload is None:
            raise RuntimeError(
                "compiled model detection did not retain its JSON payload"
            )
        return CompiledModel.from_dict(
            detected_payload,
            serialized_path=Path(resolved),
        )
    source_digest, fingerprint, cache_path = _compilation_cache_identity(
        source_kind,
        resolved,
        options=options,
        cache_dir=cache_dir,
    )
    if use_cache and cache_path.is_file():
        compiled = load_compiled_model(cache_path)
        _raise_for_unsupported(compiled, require_supported=require_supported)
        return compiled

    started = time.perf_counter()
    phase_timings: dict[str, float] = {}
    phase_started = time.perf_counter()
    raw_loop_model = None
    raw_loop_import_error = None
    if source_kind == "built-in-sm":
        from .builtin.adapters import build_model_payload

        model_payload, parameter_defaults = build_model_payload()
    else:
        worker_payload = _load_external_model(
            Path(resolved),
            options=options,
        )
        model_payload = _mapping(worker_payload.get("model"), "worker model")
        card = _mapping(worker_payload.get("parameter_card"), "parameter card")
        parameter_defaults = {
            str(name): _complex_pair(value, context=f"parameter {name}")
            for name, value in card.items()
        }
        if worker_payload.get("raw_loop_model") is not None:
            from .loop_import import raw_loop_model_from_dict

            raw_loop_model = raw_loop_model_from_dict(
                _mapping(worker_payload["raw_loop_model"], "raw_loop_model")
            )
        raw_loop_import_error = worker_payload.get("raw_loop_import_error")
    phase_timings["model_loading"] = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    issues, capabilities = preflight_model(model_payload)
    phase_timings["preflight"] = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    from .compiler import compile_builtin_model_ir, compile_ufo_model_ir

    model_ir = (
        compile_builtin_model_ir(model_payload)
        if source_kind == "built-in-sm"
        else compile_ufo_model_ir(model_payload)
    )
    phase_timings["tensor_lowering"] = time.perf_counter() - phase_started
    contact_term_ids = {term.id for term in model_ir.vertex_terms if term.valence > 3}
    lowered_contact_term_ids = {
        term_id
        for kernel in model_ir.oriented_kernels
        if "::contact-" in kernel.vertex and "final" in kernel.vertex
        for term_id in (kernel.term_ids or (kernel.term_id,))
    }
    unlowered_contact_term_ids = contact_term_ids - lowered_contact_term_ids
    if unlowered_contact_term_ids:
        issues = (
            *issues,
            ModelCompatibilityIssue(
                "error",
                "unsupported-contact-color-lowering",
                "one or more higher-point color tensors could not be lowered",
                ", ".join(
                    str(term_id) for term_id in sorted(unlowered_contact_term_ids)
                ),
            ),
        )
    evaluation_class_sizes: dict[str, int] = {}
    for kernel in model_ir.oriented_kernels:
        if not kernel.evaluation_equivalence_verified:
            continue
        evaluation_class_sizes[kernel.evaluation_class] = (
            evaluation_class_sizes.get(kernel.evaluation_class, 0) + 1
        )
    capabilities = {
        **capabilities,
        "compiled_vertex_term_count": len(model_ir.vertex_terms),
        "compiled_contact_term_count": len(lowered_contact_term_ids),
        "unlowered_contact_term_count": len(unlowered_contact_term_ids),
        "compiled_propagator_count": len(model_ir.propagators),
        "verified_kernel_evaluation_class_count": len(evaluation_class_sizes),
        "reusable_kernel_evaluation_class_count": sum(
            size > 1 for size in evaluation_class_sizes.values()
        ),
        "signed_kernel_evaluation_relation_count": sum(
            kernel.evaluation_factor != (1.0, 0.0)
            for kernel in model_ir.oriented_kernels
            if kernel.evaluation_equivalence_verified
        ),
        "swapped_kernel_evaluation_relation_count": sum(
            kernel.evaluation_input_order == (1, 0)
            for kernel in model_ir.oriented_kernels
            if kernel.evaluation_equivalence_verified
        ),
        "exchange_symmetric_kernel_evaluation_relation_count": sum(
            kernel.evaluation_input_exchange_factor is not None
            for kernel in model_ir.oriented_kernels
            if kernel.evaluation_equivalence_verified
        ),
    }
    if raw_loop_model is not None:
        capabilities["loop_model_import"] = dict(raw_loop_model.capabilities)
    elif raw_loop_import_error is not None:
        capabilities["loop_model_import"] = {
            "stage": "raw-import-incomplete",
            "error": str(raw_loop_import_error),
            "loop_runtime_supported": False,
        }
    conversion_seconds = time.perf_counter() - started
    phase_timings["total"] = conversion_seconds
    compiled = CompiledModel(
        source={
            "kind": source_kind,
            "source_name": (
                None if source_kind == "built-in-sm" else Path(resolved).name
            ),
            "digest": source_digest,
            "options": options.canonical_payload(),
        },
        producer=fingerprint,
        model=model_payload,
        ir=model_ir,
        parameter_defaults=parameter_defaults,
        capabilities=capabilities,
        issues=issues,
        phase_timings=phase_timings,
        conversion_seconds=conversion_seconds,
        raw_loop_model=raw_loop_model,
    )
    if use_cache:
        compiled.write(cache_path)
        compiled = replace(compiled, _serialized_path=cache_path.resolve())
    _raise_for_unsupported(compiled, require_supported=require_supported)
    return compiled


def load_cached_model_source(
    source: str | Path,
    *,
    restriction: str = DEFAULT_MODEL_RESTRICTION,
    simplify: bool = True,
    cache_dir: Path | None = None,
    require_supported: bool = True,
    gauge: ModelGauge | None = None,
) -> CompiledModel | None:
    """Load an already compiled source without compiling or writing a cache;
    ``gauge`` as in ``compile_model_source``."""

    options = ModelCompileOptions(
        restriction=restriction, simplify=simplify, gauge=gauge or "unitary"
    )
    compiled = _load_cached_model_source(
        source,
        options=options.gauge_independent(),
        cache_dir=cache_dir,
        require_supported=require_supported,
    )
    if compiled is None or gauge is None:
        return compiled
    return with_model_gauge(compiled, options.gauge)


def _load_cached_model_source(
    source: str | Path,
    *,
    options: ModelCompileOptions,
    cache_dir: Path | None = None,
    require_supported: bool = True,
) -> CompiledModel | None:
    """Load an already compiled source without compiling or writing a cache
    (``options`` without a gauge, as ``_compile_model_source``)."""

    if options.gauge != "unitary":
        raise ValueError("the model source is compiled without a gauge")
    restriction, simplify = options.restriction, options.simplify
    source_kind, resolved, detected_payload = _detect_model_source_payload(source)
    if source_kind in {"pyamplicol", "prepared"}:
        if restriction != DEFAULT_MODEL_RESTRICTION or not simplify:
            raise ValueError(
                "restriction and simplification options cannot be applied to an "
                "already compiled pyAmpliCol model"
            )
        if source_kind == "prepared":
            from .prepared import load_prepared_model_bundle

            bundle = load_prepared_model_bundle(Path(resolved))
            compiled = CompiledModel.from_dict(
                bundle.compiled_model_payload(),
                validate_fingerprint=False,
            )
            compiled = replace(compiled, _prepared_bundle=bundle)
            _raise_for_unsupported(compiled, require_supported=require_supported)
            return compiled
        if detected_payload is None:
            raise RuntimeError(
                "compiled model detection did not retain its JSON payload"
            )
        compiled = CompiledModel.from_dict(
            detected_payload,
            serialized_path=Path(resolved),
        )
        _raise_for_unsupported(compiled, require_supported=require_supported)
        return compiled

    _, _, cache_path = _compilation_cache_identity(
        source_kind,
        resolved,
        options=options,
        cache_dir=cache_dir,
    )
    if not cache_path.is_file():
        return None
    compiled = load_compiled_model(cache_path)
    _raise_for_unsupported(compiled, require_supported=require_supported)
    return compiled


def declared_model_gauges(compiled: CompiledModel) -> tuple[int, ...] | None:
    """The gauges the source UFO of ``compiled`` declares (its module-level
    ``gauge`` list, MG5's convention: 0 unitary, 1 Feynman), from the raw
    source records the compile keeps; ``None`` when it declares none or no
    raw source is kept (JSON and built-in models)."""

    raw = compiled.raw_loop_model
    declared = None if raw is None else raw.metadata.get("gauge")
    if not isinstance(declared, (list, tuple)) or not all(
        type(item) is int for item in declared
    ):
        return None
    return tuple(declared)


def with_model_gauge(compiled: CompiledModel, gauge: ModelGauge) -> CompiledModel:
    """``compiled`` with its massive vectors in ``gauge``
    (``contracts.ModelGauge``).

    The 't Hooft-Feynman gauge is admissible only for a model whose source
    declares gauge 1 (``declared_model_gauges``): its Goldstone vertices and
    counterterms are written for it. Only the Goldstone dispositions
    (``compiler_gauge``) and the recorded gauge change: every vertex, kernel,
    propagator and parameter record is the compile's own, and the gauge is
    recorded in the compile options (``ModelCompileOptions.gauge``). A
    prepared bundle carries the gauge it was prepared in and is refused.
    """

    if gauge not in MODEL_GAUGES:
        raise ValueError(
            f"model gauge must be one of {', '.join(MODEL_GAUGES)}, not {gauge!r}"
        )
    if compiled.ir.gauge == gauge:
        return compiled
    if compiled.prepared_bundle is not None:
        raise ValueError(
            "a prepared model bundle carries the gauge it was prepared in "
            f"({compiled.ir.gauge}); prepare the model in {gauge} gauge instead"
        )
    if gauge == "feynman":
        declared = declared_model_gauges(compiled)
        if declared is None or 1 not in declared:
            raise ValueError(
                f"model gauge {gauge!r} needs a UFO model that declares gauge 1 "
                f"('t Hooft-Feynman); {compiled.name!r} declares "
                f"{'none' if declared is None else list(declared)}"
            )
    from .compiler_gauge import compile_goldstone_partner_records

    ir = compiled.ir
    options = {
        key: value
        for key, value in dict(
            _mapping(compiled.source.get("options", {}), "source options")
        ).items()
        if key != "gauge"
    }
    return replace(
        compiled,
        source={
            **compiled.source,
            # As ``ModelCompileOptions.canonical_payload`` records it.
            "options": {**options, **({} if gauge == "unitary" else {"gauge": gauge})},
        },
        ir=replace(
            ir,
            gauge=gauge,
            goldstone_partners=compile_goldstone_partner_records(
                ir.particles, ir.parameters, ir.propagators, gauge=gauge
            ),
        ),
        _serialized_path=None,
        _prepared_bundle=None,
    )


def load_compiled_model(path: Path) -> CompiledModel:
    from .prepared import PREPARED_MODEL_BUNDLE_SUFFIX, load_prepared_model_bundle

    if Path(path).name.lower().endswith(PREPARED_MODEL_BUNDLE_SUFFIX):
        bundle = load_prepared_model_bundle(Path(path))
        return replace(
            CompiledModel.from_dict(
                bundle.compiled_model_payload(),
                validate_fingerprint=False,
            ),
            _prepared_bundle=bundle,
        )
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load compiled model {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"compiled model root must be an object: {path}")
    return CompiledModel.from_dict(payload, serialized_path=Path(path))


def preflight_model(
    model: Mapping[str, object],
) -> tuple[tuple[ModelCompatibilityIssue, ...], dict[str, object]]:
    particles = _list_of_mappings(model.get("particles"), "particles")
    propagators = _list_of_mappings(model.get("propagators", []), "propagators")
    vertices = _list_of_mappings(model.get("vertex_rules"), "vertex_rules")
    functions = _list_of_mappings(model.get("functions", []), "functions")
    form_factors = _list_of_mappings(model.get("form_factors", []), "form_factors")
    issues: list[ModelCompatibilityIssue] = []
    spins = sorted({int(particle.get("spin", 0)) for particle in particles})
    colors = sorted({int(particle.get("color", 0)) for particle in particles})
    unsupported_spins = sorted(set(spins) - {-1, 1, 2, 3, 5})
    if unsupported_spins:
        issues.append(
            ModelCompatibilityIssue(
                "error",
                "unsupported-spin",
                f"unsupported UFO spin codes: {unsupported_spins}",
            )
        )
    unsupported_colors = sorted(set(colors) - {-3, 1, 3, 8})
    if unsupported_colors:
        issues.append(
            ModelCompatibilityIssue(
                "error",
                "unsupported-color-representation",
                f"unsupported UFO color representations: {unsupported_colors}",
            )
        )
    majorana = sorted(
        str(particle.get("name"))
        for particle in particles
        if int(particle.get("spin", 0)) == 2
        and particle.get("name") == particle.get("antiname")
    )
    if majorana:
        issues.append(
            ModelCompatibilityIssue(
                "error",
                "majorana-fermion",
                "Majorana/FNV fermion flow is not implemented",
                ", ".join(majorana),
            )
        )
    if form_factors:
        issues.append(
            ModelCompatibilityIssue(
                "error",
                "form-factors",
                "UFO form factors are not supported in this milestone",
                ", ".join(sorted(str(item.get("name")) for item in form_factors)),
            )
        )
    declared_functions = {str(function.get("name")) for function in functions}
    invalid_function_arities = []
    for function in functions:
        name = str(function.get("name"))
        if name not in SUPPORTED_FUNCTION_ARITIES:
            continue
        arity = len(_sequence(function.get("arguments")))
        if arity not in SUPPORTED_FUNCTION_ARITIES[name]:
            invalid_function_arities.append(f"{name}/{arity}")
    if invalid_function_arities:
        issues.append(
            ModelCompatibilityIssue(
                "error",
                "function-arity",
                "model function declarations do not match pyAmpliCol's registry",
                ", ".join(sorted(invalid_function_arities)),
            )
        )
    expression_functions = _model_expression_functions(model)
    unknown_functions = sorted(
        (declared_functions | expression_functions)
        - SUPPORTED_FUNCTIONS
        - UFO_TENSOR_HEADS
    )
    if unknown_functions:
        issues.append(
            ModelCompatibilityIssue(
                "error",
                "unknown-functions",
                "model uses functions that are not in pyAmpliCol's hard-coded registry",
                ", ".join(unknown_functions),
            )
        )
    massive_tensors = [
        particle
        for particle in particles
        if int(particle.get("spin", 0)) == 5
        and str(particle.get("mass", "ZERO")).upper() != "ZERO"
    ]
    if massive_tensors:
        issues.append(
            ModelCompatibilityIssue(
                "warning",
                "experimental-massive-spin-2",
                "massive spin-2 support is experimental",
                ", ".join(str(item.get("name")) for item in massive_tensors),
            )
        )
    max_valence = max(
        (len(_sequence(vertex.get("particles"))) for vertex in vertices), default=0
    )
    capabilities = {
        "supported": not any(issue.severity == "error" for issue in issues),
        "particle_count": len(particles),
        "parameter_count": len(_sequence(model.get("parameters"))),
        "vertex_count": len(vertices),
        "max_vertex_valence": max_valence,
        "spins": spins,
        "color_representations": colors,
        "declared_functions": sorted(declared_functions),
        "expression_functions": sorted(expression_functions),
        "form_factor_count": len(form_factors),
        "has_custom_propagators": any(
            propagator.get(PROPAGATOR_SOURCE_FIELD) == MODEL_SUPPLIED_PROPAGATOR_SOURCE
            for propagator in propagators
        ),
        "color_accuracy_modes": ["lc", "nlc", "full"],
    }
    return tuple(issues), capabilities


@contextmanager
def _sanitized_model_environment() -> Iterator[None]:
    """Hide historical untyped model switches while trusted UFO code executes."""

    with _MODEL_LOAD_LOCK:
        previous_dont_write_bytecode = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        removed = {
            name: value
            for name, value in os.environ.items()
            if name.startswith(SANITIZED_MODEL_ENVIRONMENT_PREFIXES)
        }
        for name in removed:
            os.environ.pop(name, None)
        try:
            yield
        finally:
            sys.dont_write_bytecode = previous_dont_write_bytecode
            for name in tuple(os.environ):
                if name.startswith(SANITIZED_MODEL_ENVIRONMENT_PREFIXES):
                    os.environ.pop(name, None)
            os.environ.update(removed)


def _load_external_model(
    source: Path,
    *,
    options: ModelCompileOptions,
) -> dict[str, object]:
    # These imports intentionally live inside the execution boundary so package
    # import, metadata inspection, and built-in-SM planning stay dependency-light.
    _require_ufo_model_loader()
    with _sanitized_model_environment():
        from ufo_model_loader.commands import load_model
        from ufo_model_loader.common import JSONLook

        restriction = _loader_restriction_name(source, options.restriction)
        try:
            model, parameter_card = load_model(
                str(source),
                restriction,
                options.simplify,
                wrap_indices_in_lorentz_structures=False,
                # Loop counterterms belong to the exact source program. A tree
                # projection need not parse or evaluate their finite
                # coefficients.
                include_counterterm_finite_parts=not source.is_dir(),
            )
        except Exception as error:
            from .source_refusals import refuse_python2_ufo

            refuse_python2_ufo(source, error)
            raise
        _canonical_loader_values(model)
        from .loop_import import adapt_raw_ufo_model

        raw_loop_model = None
        raw_loop_import_error = model.raw_ufo_error
        if model.raw_ufo is not None:
            try:
                raw_loop_model = adapt_raw_ufo_model(model.raw_ufo)
            except (TypeError, ValueError) as error:
                # Invalid loop declarations do not reject a usable tree view,
                # but cannot silently acquire a complete loop source.
                raw_loop_import_error = str(error)
        propagator_sources = _classify_external_propagators(model)
        model.wrap_indices_in_lorentz_structures()
        model_payload = json.loads(model.to_json(JSONLook.COMPACT))
        # The existing compiled-model field owns the source once. The loader's
        # JSON carries it for interchange; the normalized tree payload does not
        # duplicate it in a pyAmpliCol compiled artifact.
        model_payload.pop("raw_ufo", None)
        model_payload.pop("raw_ufo_error", None)
        raw_propagators = model_payload.get("propagators", [])
        if not isinstance(raw_propagators, list):
            raise ValueError("serialized model propagators must be a list")
        for propagator in raw_propagators:
            if not isinstance(propagator, dict):
                raise ValueError("serialized model propagator must be an object")
            particle = str(propagator["particle"])
            propagator[PROPAGATOR_SOURCE_FIELD] = propagator_sources[particle]
        result: dict[str, object] = {
            "loader_version": _distribution_version("ufo-model-loader", "missing"),
            "model": model_payload,
            "parameter_card": {
                name: [value.real, value.imag]
                for name, value in sorted(parameter_card.items())
            },
        }
        if raw_loop_model is not None:
            result["raw_loop_model"] = raw_loop_model.to_dict()
        if raw_loop_import_error is not None:
            result["raw_loop_import_error"] = raw_loop_import_error
        return result


def _canonical_loader_values(model: Any) -> None:
    """Re-evaluate the loader's internal parameters and couplings on
    canonical symbols.

    ufo_model_loader evaluates them in binary64 in Symbolica's normal-form
    order, which depends on the process history (``canonical_order``): the
    same card gave loop_sm's ``yb = sqrt(2) ymb / v`` one ulp apart in two
    processes, and with it every coupling built from it. Each value is
    evaluated again by the loader's own function, with its model functions
    and the values of the parameters it depends on, on the expression whose
    variables are renamed to canonical symbols (function heads, which select
    the model functions, keep theirs; their calls are evaluated first and
    in a fixed order). Values in the loader's dependency order, as the loader
    computes them; external and constant parameters keep the card's values.
    """

    from ufo_model_loader.symbolica_processing import (
        evaluate_symbolica_expression_safe,
    )

    from .._internal.physics.canonical_order import canonical_expression
    from .._internal.physics.symbols import symbols

    functions = model.get_model_functions()
    values: dict[Any, complex] = model.get_model_variables()

    def evaluate(expression: Any) -> complex:
        # Symbolica's own constants (pi) keep their symbols, as the heads.
        variables = set(expression.get_all_symbols(False)).intersection(values)
        canonical, renaming = canonical_expression(
            expression, keep=lambda symbol: symbol not in variables
        )
        pool = {
            renaming.pool[index]: values[symbol]
            for index, symbol in enumerate(renaming.symbols)
        }
        return evaluate_symbolica_expression_safe(canonical, pool, functions)

    pending = []
    heads = set()
    for parameter in model.parameters:
        head = symbols.qualified(f"UFO::{parameter.name}")
        heads.add(head)
        if parameter.nature == "external" or parameter.expression is None:
            if parameter.value is not None:
                values[head] = parameter.value
            continue
        pending.append((parameter, head))
    while pending:
        remaining = []
        for parameter, head in pending:
            dependencies = heads.intersection(
                parameter.expression.get_all_symbols(False)
            )
            if dependencies <= values.keys():
                parameter.value = evaluate(parameter.expression)
                values[head] = parameter.value
            else:
                remaining.append((parameter, head))
        if len(remaining) == len(pending):
            names = ", ".join(parameter.name for parameter, _ in remaining)
            raise ValueError(f"UFO parameters depend on unknown symbols: {names}")
        pending = remaining
    for coupling in model.couplings:
        if coupling.expression is not None and coupling.value is not None:
            coupling.value = evaluate(coupling.expression)


def _classify_external_propagators(model: Any) -> dict[str, str]:
    """Classify propagators by expression, independently of UFO object names."""

    from ufo_model_loader.model import Propagator

    result: dict[str, str] = {}
    for propagator in model.propagators:
        actual = (
            propagator.numerator.to_canonical_string(),
            propagator.denominator.to_canonical_string(),
        )
        defaults = {
            (
                default.numerator.to_canonical_string(),
                default.denominator.to_canonical_string(),
            )
            for default in _default_feynman_propagator_candidates(
                Propagator,
                propagator.particle,
            )
        }
        result[str(propagator.particle.name)] = (
            DEFAULT_FEYNMAN_PROPAGATOR_SOURCE
            if actual in defaults
            else MODEL_SUPPLIED_PROPAGATOR_SOURCE
        )
    return result


def _default_feynman_propagator_candidates(
    propagator_type: Any,
    particle: Any,
) -> tuple[Any, ...]:
    """Cover loader defaults created before a restriction changes a mass value."""

    current = propagator_type.from_particle(particle, "Feynman")
    if particle.spin not in {1, 2, 3, -1}:
        return (current,)
    alternate_particle = copy(particle)
    alternate_particle.mass = copy(particle.mass)
    alternate_particle.mass.value = 0j if particle.is_massive() else 1.0 + 0j
    alternate = propagator_type.from_particle(alternate_particle, "Feynman")
    return current, alternate


def _loader_restriction_name(source: Path, restriction: str) -> str | None:
    """Translate the public restriction-file contract to the loader's name API."""

    if restriction == DEFAULT_MODEL_RESTRICTION:
        return None
    if restriction == NO_MODEL_RESTRICTION:
        return "full"

    path = Path(restriction).expanduser()
    if not path.is_absolute():
        return restriction
    path = path.resolve(strict=False)
    if not path.is_file():
        raise ValueError(f"model restriction file does not exist: {path}")

    model_root = source if source.is_dir() else source.parent
    if path.parent != model_root.resolve(strict=False):
        raise ValueError(
            "model restriction files must be stored next to the selected model"
        )

    suffix = ".dat" if source.is_dir() else ".json"
    prefix = "restrict_"
    if path.suffix.lower() != suffix or not path.name.startswith(prefix):
        legacy_default = (
            not source.is_dir() and path.name == f"{source.stem}_default.json"
        )
        if legacy_default:
            return DEFAULT_MODEL_RESTRICTION
        raise ValueError(
            f"model restriction file must be named {prefix}<name>{suffix}: {path}"
        )
    name = path.name[len(prefix) : -len(suffix)]
    if not name:
        raise ValueError(f"model restriction filename has no restriction name: {path}")
    return name


def _model_expression_functions(model: Mapping[str, object]) -> set[str]:
    expressions: list[str] = []
    for key in ("parameters", "couplings", "propagators", "lorentz_structures"):
        for item in _list_of_mappings(model.get(key, []), key):
            for value in item.values():
                if isinstance(value, str):
                    expressions.append(value)
    for vertex in _list_of_mappings(model.get("vertex_rules", []), "vertex_rules"):
        expressions.extend(
            str(value) for value in _sequence(vertex.get("color_structures"))
        )
    heads = set()
    for expression in expressions:
        for match in re.finditer(
            r"(?:(?:UFO|spenso)::(?:\{\}::)?)?([A-Za-z][A-Za-z0-9_]*)\(",
            expression,
        ):
            heads.add(match.group(1))
    return heads


def _source_digest(kind: str, source: str | Path) -> str:
    digest = hashlib.sha256()
    digest.update(kind.encode("utf-8") + b"\0")
    if kind == "built-in-sm":
        from .builtin.adapters import source_digest

        digest.update(source_digest().encode("ascii"))
        return digest.hexdigest()
    path = Path(source)
    root = path.parent if path.is_file() else path
    files = (
        [path]
        if path.is_file()
        else sorted(
            item
            for item in path.rglob("*")
            if item.is_file()
            and item.suffix != ".pyc"
            and "__pycache__" not in item.parts
        )
    )
    for item in files:
        relative = item.name if path.is_file() else item.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _compilation_cache_identity(
    source_kind: str,
    source: str | Path,
    *,
    options: ModelCompileOptions,
    cache_dir: Path | None,
) -> tuple[str, dict[str, object], Path]:
    source_digest = _source_digest(source_kind, source)
    fingerprint = compiler_fingerprint()
    cache_key = hashlib.sha256(
        json.dumps(
            {
                "source_digest": source_digest,
                "options": options.canonical_payload(),
                "producer": fingerprint,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    active_cache = cache_dir or _default_model_cache_dir()
    return (
        source_digest,
        fingerprint,
        active_cache / f"{cache_key}.pyAmplicol-model.json",
    )


def _model_compiler_digest() -> str:
    digest = hashlib.sha256()
    package_root = Path(__file__).resolve().parents[1]
    for path in _model_compiler_source_paths():
        relative = path.relative_to(package_root).as_posix()
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _model_compiler_source_paths() -> tuple[Path, ...]:
    """Return every first-party source file that can change compiled model IR."""
    package_root = Path(__file__).resolve().parents[1]
    model_sources = Path(__file__).resolve().parent.glob("*.py")
    physics_sources = (package_root / "_internal" / "physics").glob("*.py")
    process_sources = (package_root / "processes" / "core_syntax.py",)
    return tuple(sorted((*model_sources, *physics_sources, *process_sources)))


def _default_model_cache_dir() -> Path:
    configured = os.environ.get("PYAMPLICOL_CACHE_DIR")
    if configured:
        root = Path(configured).expanduser()
    else:
        from platformdirs import user_cache_path

        root = user_cache_path("pyamplicol")
    return root / "models"


def _compiled_model_output_path(path: Path) -> Path:
    text = str(path)
    suffix = ".pyAmplicol-model.json"
    return Path(text if text.endswith(suffix) else text + suffix)


def _atomic_write_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    compact: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    serialized = (
        json.dumps(payload, separators=(",", ":"), sort_keys=True)
        if compact
        else json.dumps(payload, indent=2, sort_keys=True)
    )
    temporary.write_text(serialized + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_copy(source: Path, path: Path) -> None:
    source = source.resolve()
    path = path.resolve()
    if source == path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(path)


def _distribution_version(name: str, default: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return default


# ----------------------------------------------------------- counterterm lowering

# Pseudo-order of a counterterm (CT) insertion. One insertion carries it
# COUNTERTERM_INSERTION_WEIGHT times: a CT vertex twice, a two-point insertion
# once on each of the two vertices around its clone line
# (``models.ct_line_insertions``). The order limit 2 therefore interns the
# currents without and with one insertion in one tree recursion.
COUNTERTERM_INSERTION_ORDER = "PYAMPLICOL_CT_INSERTION"
COUNTERTERM_INSERTION_WEIGHT = 2
COUNTERTERM_COMPONENTS = ("r2", "uv_local")
COUNTERTERM_POWERS = (0, -1, -2)
_COUNTERTERM_PARAMETER_BLOCK = "PYAMPLICOL_CT"


def counterterm_slot_order(component: str, power: int, grade: int = 0) -> str:
    """Pseudo-order of one CT slot (component x epsilon power, design §6.1).

    Each insertion carries the order of its slot (``COUNTERTERM_INSERTION_WEIGHT``
    times, like the insertion order), so currents and roots with one
    insertion stay separate per slot and the squared-order partials of the
    tree runtime split the Born x CT interference into components and
    Laurent coefficients. ``grade`` is the N_c power of a coupling graded by
    a formal-N_c continuation (design §6.7); grade 0 keeps the plain name.
    """

    if component not in COUNTERTERM_COMPONENTS or power not in COUNTERTERM_POWERS:
        raise ValueError(f"unknown counterterm slot ({component!r}, {power})")
    name = f"{_COUNTERTERM_PARAMETER_BLOCK}_{component.upper()}_EPS{-power}"
    if grade:
        name += f"_NC{grade}" if grade > 0 else f"_NCM{-grade}"
    return name


@dataclass(frozen=True, slots=True, order=True)
class CountertermStructure:
    """One CT vertex term of one slot: what the tree recursion lowers.

    ``particles`` are the UFO particle names in vertex-leg order, ``colour``
    the CT entry's UFO colour (``models.ct_color_flow``), ``lorentz`` the name
    of a Lorentz structure of the model, ``orders`` the physical coupling
    orders of the entry, ``component`` ``"r2"`` or ``"uv_local"`` and
    ``power`` the epsilon power of the slot and ``grade`` the N_c power of a
    coupling graded by a formal-N_c continuation (0 without one).
    """

    particles: tuple[str, ...]
    colour: str
    lorentz: str
    orders: tuple[tuple[str, int], ...]
    component: str
    power: int
    grade: int = 0

    @property
    def slot_order(self) -> str:
        return counterterm_slot_order(self.component, self.power, self.grade)


class CountertermLoweringError(ValueError):
    """CT vertex terms the tree compiler compiled but cannot decompose.

    ``refused`` maps each such ``CountertermStructure`` to the reason.
    """

    def __init__(self, refused: Mapping[CountertermStructure, str]):
        self.refused = dict(refused)
        super().__init__(
            "counterterm vertices the tree compiler cannot lower: "
            + "; ".join(sorted({reason for reason in self.refused.values()}))
        )


@dataclass(frozen=True)
class CountertermLoweredModel:
    """A compiled model whose tree vertices include lowered CT vertices.

    ``couplings`` names the model coupling of every lowered CT vertex term
    (arity >= 3) and ``insertions`` the clone particle of every lowered
    two-point term (``models.ct_line_insertions``); ``line_fields`` maps
    every clone particle to the field whose current it is read as and
    ``line_factors`` the clone particle whose vertex copies carry a Weyl
    line's factor ``kappa`` to ``kappa`` (``LineInsertions.factors``);
    ``dirac_fields`` are the Weyl fields of the model that the lowered model
    propagates as Dirac fields (``LineInsertions.dirac_fields``);
    ``parameters`` are the real external runtime parameters the coupling
    expressions introduced. CT vertices carry the physical orders of their
    entry and ``COUNTERTERM_INSERTION_ORDER`` and the order of their slot
    ``COUNTERTERM_INSERTION_WEIGHT`` times. ``refused`` lists terms the tree
    compiler could not lower (``CountertermLoweringError``).
    """

    compiled: CompiledModel
    couplings: Mapping[CountertermStructure, str]
    parameters: tuple[str, ...]
    insertions: Mapping[CountertermStructure, str] = field(default_factory=dict)
    refused: Mapping[CountertermStructure, str] = field(default_factory=dict)
    line_factors: Mapping[str, str] = field(default_factory=dict)
    line_fields: Mapping[str, str] = field(default_factory=dict)
    dirac_fields: frozenset[str] = frozenset()
    # The lowering-cache entry of ``compiled`` (file, key), which also keeps
    # its Weyl-projection support (``tree_model``); None: not cached.
    cache_entry: tuple[Path, str] | None = field(
        default=None, compare=False, repr=False
    )
    _weyl_support: list[Mapping[tuple[int, int, int, int], bool]] = field(
        default_factory=list, compare=False, repr=False
    )

    @property
    def slot_orders(self) -> tuple[str, ...]:
        return tuple(
            sorted({s.slot_order for s in (*self.couplings, *self.insertions)})
        )

    def tree_model(self) -> CompiledUFOModel:
        """The tree model of ``compiled`` (a new ``CompiledUFOModel``) with
        its Weyl-projection support decided (``counterterm_weyl_support``:
        from the lowering cache, or probed once and stored there), so that
        no tree recursion over the lowered model repeats those probes."""

        from .external import CompiledUFOModel

        model = CompiledUFOModel(self.compiled)
        if not self._weyl_support:
            self._weyl_support.append(counterterm_weyl_support(model, self.cache_entry))
        model.seed_weyl_projection_support(self._weyl_support[0])
        return model


_METRIC_LORENTZ_PREFIX = f"{_COUNTERTERM_PARAMETER_BLOCK}_METRIC"


def counterterm_metric_lorentz_name(pairs: Iterable[tuple[int, int]]) -> str:
    """Name of the Lorentz structure ``prod Metric(a, b)`` over 1-based legs
    ``pairs`` that ``lower_counterterm_vertices`` adds to the payload."""

    canonical = sorted(tuple(sorted(pair)) for pair in pairs)
    return _METRIC_LORENTZ_PREFIX + "".join(f"_{a}{b}" for a, b in canonical)


def counterterm_metric_lorentz(name: str) -> dict[str, Any] | None:
    """The payload record of a ``counterterm_metric_lorentz_name``, or None."""

    if not name.startswith(_METRIC_LORENTZ_PREFIX + "_"):
        return None
    pairs = [
        (int(text[0]), int(text[1]))
        for text in name.removeprefix(_METRIC_LORENTZ_PREFIX + "_").split("_")
    ]
    legs = sorted(leg for pair in pairs for leg in pair)
    if legs != list(range(1, len(legs) + 1)):
        return None
    return {
        "name": name,
        "spins": [3] * len(legs),
        "structure": "*".join(
            f"UFO::{{}}::Metric(UFO::{{}}::idx(1,{a}),UFO::{{}}::idx(1,{b}))"
            for a, b in pairs
        ),
    }


def counterterm_lorentz_error(compiled: CompiledModel, name: str) -> str | None:
    """Why the tree compiler cannot normalise Lorentz structure ``name``.

    ``None`` when it can, the Levi-Civita tensor of some CT vertices (e.g. the
    gluon-gluon-Z R2) included (``counterterm_lorentz_levi_civita``).
    """

    from .._internal.physics.symbols import symbols
    from .tensors import normalize_lorentz_expression

    (record,) = (
        item for item in compiled.model["lorentz_structures"] if item["name"] == name
    )
    try:
        normalize_lorentz_expression(
            str(record["structure"]),
            tuple(int(value) for value in record["spins"]),
            model_symbols=symbols.model(compiled.name),
        )
    except (NotImplementedError, ValueError) as error:
        return str(error)
    return None


def counterterm_lorentz_levi_civita(compiled: CompiledModel, name: str) -> bool:
    """Whether the normalised Lorentz structure ``name`` contracts the
    Levi-Civita tensor (``tensors.LEVI_CIVITA``, UFO ``Epsilon``)."""

    from .._internal.physics.symbols import symbols
    from .tensors import LEVI_CIVITA, normalize_lorentz_expression

    (record,) = (
        item for item in compiled.model["lorentz_structures"] if item["name"] == name
    )
    heads = normalize_lorentz_expression(
        str(record["structure"]),
        tuple(int(value) for value in record["spins"]),
        model_symbols=symbols.model(compiled.name),
    ).tensor_heads
    return LEVI_CIVITA.rsplit("::", 1)[-1] in heads


def _orders_within(
    orders: Iterable[tuple[str, int]], limits: Mapping[str, int]
) -> bool:
    return all(
        int(power) <= limits.get(str(name), int(power)) for name, power in orders
    )


def _vertex_rules_within(
    rules: Sequence[Mapping[str, Any]],
    couplings: Sequence[Mapping[str, Any]],
    limits: Mapping[str, int],
) -> list[dict[str, Any]]:
    """The vertex rules of a payload restricted to couplings within ``limits``:
    cells above them are emptied, colour rows and Lorentz columns left
    without a cell are removed, and so are rules left without any."""

    within = {
        str(item["name"]): _orders_within(
            (tuple(pair) for pair in item.get("orders", ())), limits
        )
        for item in couplings
    }
    kept: list[dict[str, Any]] = []
    for rule in rules:
        cells = [
            [
                name if name is not None and within.get(str(name), True) else None
                for name in row
            ]
            for row in rule["couplings"]
        ]
        rows = [i for i, row in enumerate(cells) if any(n is not None for n in row)]
        columns = [
            j
            for j in range(len(rule["lorentz_structures"]))
            if any(cells[i][j] is not None for i in rows)
        ]
        if not rows:
            continue
        if len(rows) == len(cells) and len(columns) == len(rule["lorentz_structures"]):
            kept.append(rule)
            continue
        kept.append(
            {
                **rule,
                "color_structures": [rule["color_structures"][i] for i in rows],
                "lorentz_structures": [rule["lorentz_structures"][j] for j in columns],
                "couplings": [[cells[i][j] for j in columns] for i in rows],
            }
        )
    return kept


def lower_counterterm_vertices(
    compiled: CompiledModel,
    couplings: Mapping[CountertermStructure, str],
    *,
    parameters: Mapping[str, float] = MappingProxyType({}),
    excluded_particles: Iterable[str] = (),
    tree_kernel_variant: TreeKernelVariant = "full",
    multi_trace: bool = False,
    dirac_fields: Iterable[str] = (),
    order_limits: Mapping[str, int] | None = None,
) -> CountertermLoweredModel:
    """Lower CT vertex terms into the tree model of ``compiled`` (design §6.1).

    The CT vertices become ordinary UFO vertex rules of a copy of the compiled
    payload: one rule per (particles, orders, slot) with a coupling matrix
    over its colour and Lorentz structures. They go through the ordinary
    tree compiler (``compile_ufo_model_ir``), so Lorentz normalisation,
    oriented kernels and contact decompositions are those of tree vertices,
    in the model's kernel variant. Two-point terms become clone lines
    (``models.ct_line_insertions``): a clone of the field with the
    propagator ``P Pi P`` and copies of the field's vertices (a Goldstone
    and its vector mix through copies of the Goldstone's vertices that carry
    the line momentum; the propagators are those the engine applies in the
    model's gauge, and terms on Goldstones the unitary-gauge tree model
    absorbs are refused). Colours
    outside the tree colour flow need the multi-trace colour state and are
    refused here. Terms the tree compiler cannot lower (contact vertices
    without a colour-flow decomposition, e.g. two gluons and two
    colour-singlet bosons, and the two-point terms
    ``ct_line_insertions`` refuses) raise ``CountertermLoweringError``
    naming them, so that a caller can close their rules as sites.

    ``couplings`` maps every vertex term to its coupling expression in the
    payload's canonical syntax; terms of one slot and orders with the same
    expression share one model coupling, so that cells with a common
    coupling (the three colour-Lorentz cells of the four-gluon UV
    counterterm) fuse into one contact kernel as the ordinary quartic vertex
    does. The expressions are derived model couplings: the runtime evaluates
    them from the model parameters in every carrier, exactly in arbitrary
    precision. ``parameters`` declares the real external parameters they
    introduce, with their defaults. A coupling that vanishes at the defaults
    is a structural zero to the tree compiler, so defaults must not make a
    lowered coupling vanish.

    ``excluded_particles`` (UFO names, both charge states) removes every
    vertex, ordinary or CT, with such a leg: the tree model of a process that
    forbids those particles (MadGraph's ``/ z``), which the tree generator
    has no option for. Terms whose coupling the model's restriction makes
    vanish are dropped (absent from ``couplings`` of the result).

    The tree kernels are compiled in the ``"full"`` variant by default: a
    CT = 1 current is not conserved, so the transverse Yang-Mills shortcut,
    exact only for conserved inputs, does not apply to it (design §3.2).

    ``multi_trace`` admits the colours whose currents carry closed traces
    (``ct_color_flow``): only loop-request blocks carry those, so a model
    lowered with it serves the counterterm closure (``loops.v2.ct_closure``),
    not complete tree programs.

    ``dirac_fields`` (UFO names) are Weyl fields to propagate as Dirac fields
    in the lowered model, besides those its two-point terms require (a
    chiral or chirality-flipping two-point term, or mixing with a Dirac
    field; ``ct_line_insertions``): their standard propagators become
    model-supplied, so that their sources, currents and clone lines are
    Dirac spinors.

    ``order_limits`` (physical coupling order -> largest power; an unlisted
    order is unbounded) are the order limits of the tree program the model
    serves: a coupling above them can be in no tree of that program, so the
    ordinary vertex cells it holds are dropped (a vertex with none left is
    dropped), and a CT term above them is refused as a ``ValueError`` (the
    caller bounds its rules first). With ``None`` the whole model is kept.
    """

    from .ct_color_flow import tree_colour_source
    from .ct_line_insertions import lower_line_insertions

    dirac_fields = frozenset(dirac_fields)
    payload = json.loads(json.dumps(compiled.model))
    particles = {str(item["name"]): item for item in payload["particles"]}
    excluded = set(excluded_particles)
    excluded |= {str(particles[name]["antiname"]) for name in excluded}
    payload["vertex_rules"] = [
        rule
        for rule in payload["vertex_rules"]
        if not excluded & set(rule["particles"])
    ]
    if order_limits is not None:
        payload["vertex_rules"] = _vertex_rules_within(
            payload["vertex_rules"], payload["couplings"], order_limits
        )
        beyond = [s for s in couplings if not _orders_within(s.orders, order_limits)]
        if beyond:
            raise ValueError(
                f"counterterm terms beyond the order limits {dict(order_limits)}: "
                + ", ".join(sorted(str(s.particles) for s in beyond))
            )
    lorentz_names = {str(item["name"]) for item in payload["lorentz_structures"]}
    groups: dict[tuple[object, ...], list[CountertermStructure]] = {}
    structures = tuple(sorted(s for s in couplings if not excluded & set(s.particles)))
    for structure in structures:
        if structure.lorentz not in lorentz_names:
            record = counterterm_metric_lorentz(structure.lorentz)
            if record is None:
                raise ValueError(f"unknown Lorentz structure {structure.lorentz!r}")
            payload["lorentz_structures"].append(record)
            lorentz_names.add(structure.lorentz)
        if len(structure.particles) < 3:
            continue
        groups.setdefault(
            (
                structure.particles,
                structure.orders,
                structure.component,
                structure.power,
                structure.grade,
            ),
            [],
        ).append(structure)
    defaults = dict(compiled.parameter_defaults)
    known = {str(item["name"]) for item in payload["parameters"]}
    for index, (name, default) in enumerate(sorted(parameters.items())):
        if name in known:
            raise ValueError(f"counterterm parameter {name!r} already exists")
        payload["parameters"].append(
            {
                "expression": None,
                "lhablock": _COUNTERTERM_PARAMETER_BLOCK,
                "lhacode": [index + 1],
                "name": name,
                "nature": "external",
                "parameter_type": "real",
                "value": [float(default), 0.0],
            }
        )
        defaults[name] = (float(default), 0.0)
    lines = lower_line_insertions(
        payload,
        {s: couplings[s] for s in structures if len(s.particles) == 2},
        prefix=_COUNTERTERM_PARAMETER_BLOCK,
        insertion_order=COUNTERTERM_INSERTION_ORDER,
        slot_order=lambda structure: structure.slot_order,
        weyl_fields=_weyl_fields(compiled),
        dirac_fields=dirac_fields,
        gauge=compiled.ir.gauge,
        absorbed=frozenset(
            record.goldstone
            for record in compiled.ir.goldstone_partners
            if record.policy == "absorbed"
        ),
    )
    if lines.refused:
        raise CountertermLoweringError(lines.refused)
    # Weyl fields the clone lines need as Dirac fields: a model-supplied
    # standard propagator makes every current of the field a Dirac spinor.
    for propagator in payload["propagators"]:
        if propagator.get("particle") in lines.dirac_fields:
            propagator[PROPAGATOR_SOURCE_FIELD] = MODEL_SUPPLIED_PROPAGATOR_SOURCE
    classes: dict[tuple[object, ...], list[CountertermStructure]] = {}
    for structure in (s for s in structures if len(s.particles) > 2):
        classes.setdefault(
            (
                structure.orders,
                structure.component,
                structure.power,
                structure.grade,
                couplings[structure],
            ),
            [],
        ).append(structure)
    names: dict[CountertermStructure, str] = {}
    for index, ((orders, *_slot, expression), members) in enumerate(classes.items()):
        coupling = f"{_COUNTERTERM_PARAMETER_BLOCK}C_{index}"
        for member in members:
            names[member] = coupling
        payload["couplings"].append(
            {
                "expression": expression,
                "name": coupling,
                "orders": [
                    *([name, value] for name, value in orders),
                    [COUNTERTERM_INSERTION_ORDER, COUNTERTERM_INSERTION_WEIGHT],
                    [members[0].slot_order, COUNTERTERM_INSERTION_WEIGHT],
                ],
                "value": [1.0, 0.0],
            }
        )
    vertex_members: dict[str, list[CountertermStructure]] = {}
    for number, ((legs, *_slot), members) in enumerate(sorted(groups.items())):
        representations = [int(particles[name]["color"]) for name in legs]
        colours = sorted({member.colour for member in members})
        lorentz = sorted({member.lorentz for member in members})
        cells = {(member.colour, member.lorentz): names[member] for member in members}
        vertex_members[f"{_COUNTERTERM_PARAMETER_BLOCK}_V{number}"] = members
        payload["vertex_rules"].append(
            {
                "name": f"{_COUNTERTERM_PARAMETER_BLOCK}_V{number}",
                "particles": list(legs),
                "color_structures": [
                    tree_colour_source(colour, representations, multi_trace=multi_trace)
                    for colour in colours
                ],
                "lorentz_structures": lorentz,
                "couplings": [
                    [cells.get((colour, name)) for name in lorentz]
                    for colour in colours
                ],
            }
        )
    payload["parameters"].extend(lines.parameters)
    payload["lorentz_structures"].extend(lines.lorentz_structures)
    payload["particles"].extend(lines.particles)
    payload["propagators"].extend(lines.propagators)
    payload["couplings"].extend(lines.couplings)
    payload["vertex_rules"].extend(lines.vertex_rules)
    vertex_members.update(lines.members)
    hierarchy = max((int(item["hierarchy"]) for item in payload["orders"]), default=0)
    for offset, name in enumerate(
        sorted(
            {COUNTERTERM_INSERTION_ORDER}
            | {structure.slot_order for structure in structures}
        ),
        start=1,
    ):
        payload["orders"].append(
            {"expansion_order": -1, "hierarchy": hierarchy + offset, "name": name}
        )
    started = time.perf_counter()
    # The multi-trace currents project the gluon legs of the counterterm
    # vertices onto SU(N) (U(1)-subtraction auxiliaries, design §6.1).
    ir, entry = _counterterm_ir(
        payload,
        tree_kernel_variant=tree_kernel_variant,
        u1_projected_vertices=frozenset(vertex_members) if multi_trace else frozenset(),
        gauge=compiled.ir.gauge,
    )
    # Couplings the model's restriction makes vanish (e.g. off-diagonal CKM
    # entries) are structural zeros: their terms are dropped, not lowered.
    vanishing = {
        term.coupling
        for term in ir.vertex_terms
        if term.coupling.startswith(f"{_COUNTERTERM_PARAMETER_BLOCK}C_")
        and term.coupling_expression == "0"
    }
    if vanishing:
        return lower_counterterm_vertices(
            compiled,
            {s: e for s, e in couplings.items() if names.get(s) not in vanishing},
            parameters=parameters,
            excluded_particles=excluded,
            tree_kernel_variant=tree_kernel_variant,
            multi_trace=multi_trace,
            dirac_fields=dirac_fields,
            order_limits=order_limits,
        )
    lowered_vertices = {
        rule["name"]
        for rule in payload["vertex_rules"]
        if str(rule["name"]).startswith(_COUNTERTERM_PARAMETER_BLOCK)
    }
    contact = {
        term.id
        for term in ir.vertex_terms
        if term.valence > 3 and term.vertex in lowered_vertices
    }
    lowered_contact = {
        term_id
        for kernel in ir.oriented_kernels
        if "::contact-" in kernel.vertex and "final" in kernel.vertex
        for term_id in (kernel.term_ids or (kernel.term_id,))
    }
    if contact - lowered_contact:
        failed = {
            term.vertex
            for term in ir.vertex_terms
            if term.id in contact - lowered_contact
        }
        legs = {
            str(rule["name"]): "-".join(rule["particles"])
            for rule in payload["vertex_rules"]
        }
        raise CountertermLoweringError(
            {
                structure: f"no colour-flow contact decomposition of {legs[vertex]}"
                for vertex, members in vertex_members.items()
                if vertex in failed
                for structure in members
            }
        )
    digest = hashlib.sha256(
        json.dumps(
            [
                sorted(excluded),
                multi_trace,
                sorted(lines.dirac_fields),
                sorted(parameters.items()),
                [
                    [
                        list(s.particles),
                        s.colour,
                        s.lorentz,
                        s.orders,
                        s.component,
                        s.power,
                        s.grade,
                        couplings[s],
                    ]
                    for s in structures
                ],
            ],
            sort_keys=True,
        ).encode()
    ).hexdigest()
    lowered = CompiledModel(
        source={**compiled.source, "counterterm_lowering": digest},
        producer=compiled.producer,
        model=payload,
        ir=ir,
        parameter_defaults=defaults,
        capabilities={
            **compiled.capabilities,
            "counterterm_vertex_term_count": len(structures),
        },
        issues=compiled.issues,
        phase_timings={
            **compiled.phase_timings,
            "counterterm_lowering": time.perf_counter() - started,
        },
        conversion_seconds=compiled.conversion_seconds,
    )
    return CountertermLoweredModel(
        lowered,
        names,
        tuple(sorted(parameters)),
        insertions=lines.clones,
        line_factors=lines.factors,
        line_fields=lines.reads_as,
        dirac_fields=lines.dirac_fields,
        cache_entry=entry,
    )


COUNTERTERM_IR_CACHE_SUFFIX = ".pyAmplicol-counterterm-ir.json"
_COUNTERTERM_IR_CACHE_KIND = "pyamplicol-counterterm-model-ir"
COUNTERTERM_WEYL_CACHE_SUFFIX = ".pyAmplicol-counterterm-weyl.json"
_COUNTERTERM_WEYL_CACHE_KIND = "pyamplicol-counterterm-weyl-support"
_COUNTERTERM_IR_FINGERPRINT: list[dict[str, object]] = []
# Lowered models one cache directory keeps, least recently used out first:
# an entry is 14-19 MB for loop_sm's six-leg QCD scopes, and every change of
# the compiler's code identity writes new ones (``_counterterm_ir``).
COUNTERTERM_IR_CACHE_KEPT = 64


def compile_counterterm_model_ir(
    payload: Mapping[str, object],
    *,
    tree_kernel_variant: TreeKernelVariant,
    u1_projected_vertices: frozenset[str],
    cache_dir: Path | None = None,
    gauge: ModelGauge = "unitary",
) -> CompiledModelIR:
    """``compile_ufo_model_ir`` of a CT-lowered payload, content-addressed.

    The lowered model of a request scope is the same for every process of
    that scope (``lower_counterterm_vertices``), so its compile (most of the
    lowering: oriented kernels, contact decompositions and their Symbolica
    proofs) is cached like a compiled model (``compile_model_source``): in
    ``cache_dir`` (default: the model cache, ``$PYAMPLICOL_CACHE_DIR/models``
    or the platform cache) under the SHA-256 of the payload (in its own key
    order), the variant, the projected vertices, the gauge and the compiler's
    code identity (``compiler_fingerprint`` without the package version:
    Symbolica, the model-compiler sources and contracts).
    The IR round-trips exactly (``CompiledModelIR.to_dict``/``from_dict``); an
    unreadable or foreign entry is recompiled, and a cache that cannot be
    written leaves the result uncached.
    """

    return _counterterm_ir(
        payload,
        tree_kernel_variant=tree_kernel_variant,
        u1_projected_vertices=u1_projected_vertices,
        cache_dir=cache_dir,
        gauge=gauge,
    )[0]


def _counterterm_ir(
    payload: Mapping[str, object],
    *,
    tree_kernel_variant: TreeKernelVariant,
    u1_projected_vertices: frozenset[str],
    cache_dir: Path | None = None,
    gauge: ModelGauge = "unitary",
) -> tuple[CompiledModelIR, tuple[Path, str]]:
    """``compile_counterterm_model_ir`` and its cache entry (file, key)."""

    from .compiler_entry import compile_ufo_model_ir

    if not _COUNTERTERM_IR_FINGERPRINT:
        _COUNTERTERM_IR_FINGERPRINT.append(_compiler_code_identity())
    key = hashlib.sha256(
        json.dumps(
            [
                payload,
                str(tree_kernel_variant),
                sorted(u1_projected_vertices),
                # Only off the default (unitary keys are unchanged).
                *(() if gauge == "unitary" else (str(gauge),)),
                _COUNTERTERM_IR_FINGERPRINT[0],
            ],
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    path = (cache_dir or _default_model_cache_dir()) / (
        f"{key}{COUNTERTERM_IR_CACHE_SUFFIX}"
    )
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if (
            stored.get("kind") == _COUNTERTERM_IR_CACHE_KIND
            and stored.get("key") == key
        ):
            ir = CompiledModelIR.from_dict(_mapping(stored["ir"], "ir"))
            with suppress(OSError):
                os.utime(path)  # recently used (``_evict_counterterm_entries``)
            return ir, (path, key)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    ir = compile_ufo_model_ir(
        payload,
        tree_kernel_variant=tree_kernel_variant,
        u1_projected_vertices=u1_projected_vertices,
        gauge=gauge,
    )
    with suppress(OSError):
        _atomic_write_json(
            path,
            {"kind": _COUNTERTERM_IR_CACHE_KIND, "key": key, "ir": ir.to_dict()},
            compact=True,
        )
        _evict_counterterm_entries(path.parent)
    return ir, (path, key)


def _evict_counterterm_entries(
    directory: Path, kept: int = COUNTERTERM_IR_CACHE_KEPT
) -> None:
    """Remove the CT lowering entries of ``directory`` (with their Weyl
    tables, ``counterterm_weyl_support``) beyond the ``kept`` most recently
    used. Only these two kinds of file are touched; an entry another process
    loses is recompiled on its next miss, never trusted half-read."""

    entries = []
    for path in directory.glob(f"*{COUNTERTERM_IR_CACHE_SUFFIX}"):
        with suppress(OSError):
            entries.append((path.stat().st_mtime_ns, path.name, path))
    entries.sort(reverse=True)
    for _used, name, path in entries[kept:]:
        key = name[: -len(COUNTERTERM_IR_CACHE_SUFFIX)]
        for stale in (path, path.with_name(f"{key}{COUNTERTERM_WEYL_CACHE_SUFFIX}")):
            with suppress(OSError):
                stale.unlink()


def counterterm_weyl_support(
    model: CompiledUFOModel, entry: tuple[Path, str] | None
) -> dict[tuple[int, int, int, int], bool]:
    """``model.weyl_projection_support()`` of a CT-lowered model (a
    ``CompiledUFOModel`` without runtime parameters), kept next to its
    lowering-cache ``entry`` (``compile_counterterm_model_ir``: file, key) as
    ``<key>.pyAmplicol-counterterm-weyl.json``.

    The probes are symbolic zero tests of the projected kernels at the
    model's parameter defaults, so the stored table is keyed by the entry's
    key (payload and compiler code) and the defaults. An unreadable, foreign
    or malformed table is probed again; without an entry nothing is stored.
    """

    compiled = model.compiled
    if compiled is None:
        raise ValueError("the Weyl support of a model without its compiled model")
    defaults = hashlib.sha256(
        json.dumps(
            sorted(
                (str(name), [float(part) for part in value])
                for name, value in compiled.parameter_defaults.items()
            ),
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if entry is None:
        return model.weyl_projection_support()
    file, key = entry
    path = file.with_name(f"{key}{COUNTERTERM_WEYL_CACHE_SUFFIX}")
    with suppress(OSError, ValueError, KeyError, TypeError):
        stored = json.loads(path.read_text(encoding="utf-8"))
        if (
            stored["kind"] == _COUNTERTERM_WEYL_CACHE_KIND
            and stored["key"] == key
            and stored["defaults"] == defaults
        ):
            return dict(_weyl_support_rows(stored["support"]))
    support = model.weyl_projection_support()
    with suppress(OSError):
        _atomic_write_json(
            path,
            {
                "kind": _COUNTERTERM_WEYL_CACHE_KIND,
                "key": key,
                "defaults": defaults,
                "support": [
                    [*probe, value] for probe, value in sorted(support.items())
                ],
            },
            compact=True,
        )
    return support


def _weyl_support_rows(
    rows: object,
) -> Iterator[tuple[tuple[int, int, int, int], bool]]:
    if not isinstance(rows, list):
        raise ValueError("Weyl support rows must be a list")
    for row in rows:
        if (
            not isinstance(row, list)
            or len(row) != 5
            or not all(type(item) is int for item in row[:4])
            or type(row[4]) is not bool
        ):
            raise ValueError(f"malformed Weyl support row {row!r}")
        yield (row[0], row[1], row[2], row[3]), row[4]


def _weyl_fields(compiled: CompiledModel) -> frozenset[str]:
    """Fermions whose tree currents are Weyl currents (``is_chiral_eligible``)."""

    from .external import CompiledUFOModel

    model = CompiledUFOModel(compiled)
    return frozenset(
        particle.name
        for particle in compiled.ir.particles
        if model.is_chiral_eligible(particle.pdg_code)
    )


def _raise_for_unsupported(
    model: CompiledModel,
    *,
    require_supported: bool,
) -> None:
    if require_supported and not model.supported:
        details = "; ".join(
            f"{issue.code}: {issue.message}"
            + (f" ({issue.context})" if issue.context else "")
            for issue in model.issues
            if issue.severity == "error"
        )
        raise ValueError(f"model {model.name!r} is not supported: {details}")


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return {str(key): item for key, item in value.items()}


def _sequence(value: object) -> list[object]:
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _list_of_mappings(value: object, context: str) -> list[dict[str, object]]:
    return [_mapping(item, context) for item in _sequence(value)]


def _complex_pair(value: object, *, context: str) -> tuple[float, float]:
    pair = _sequence(value)
    if len(pair) != 2:
        raise ValueError(f"{context} must be [real, imaginary]")
    return float(pair[0]), float(pair[1])
