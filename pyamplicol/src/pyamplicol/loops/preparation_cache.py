# SPDX-License-Identifier: 0BSD
"""The one-loop request-model preparation, shared within a generation.

A one-loop generation prepares the request model of its model
(``loops.v2.assemble.prepare_request_model``: the restricted declarations,
their scalar program, the covariant structure, the loop model and the
counterterm model) for the loop-order inference of a loop-induced process,
for every process slice, for the dry run and, with
``generation.loop_induced_probes``, for the probe and Goldstone slices. A
``PreparationMemo`` is created per ``Generator.generate`` / ``Generator.plan``
call, passed explicitly to those consumers and dropped with the call:

* the restricted declarations and their lowered scalar program
  (``model_restrictions.prepare_loop_restrictions``, which hands its program
  to ``covariant_model.build_covariant_structure``) are formed once per raw
  model and simplification: they read neither the order bounds nor the
  fixed parameters, so every preparation of the generation shares them;
* the request model of ``loops.compiler.prepare_loop_request_model`` (and
  its declared source-colour continuation) is formed once per recipe: the
  raw model, the simplification, the order bounds, the fixed parameters
  (with the source-colour ones and a probe's zeroed widths), the
  compatibility profile and the colour declaration, everything it reads.
  The refusals that read a request's process, veto or compiled model
  (``prepare_request_model``) run again for every request.

Never a model, process or name key: the keys are the inputs themselves.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from typing import Any


class PreparationMemo:
    """The request-model preparations of one generation (module docstring).

    ``counts`` records what was formed and what was reused (``restrictions``
    / ``restriction_hits``, ``lowerings``, ``requests`` / ``request_hits``),
    for the generation's report and the tests."""

    def __init__(self) -> None:
        # id(raw) -> raw: the memo holds every raw model it keys, so an
        # object id is never reused for another model while it lives.
        self._raws: dict[int, Any] = {}
        self._sources: dict[tuple[int, bool], Any] = {}
        self._requests: dict[tuple[Any, ...], tuple[Any, Any]] = {}
        self.counts: Counter[str] = Counter()

    def _raw_key(self, raw: Any) -> int:
        self._raws.setdefault(id(raw), raw)
        return id(raw)

    # -- the restricted declarations and their scalar program --------------

    def restrictions(self, raw: Any, *, simplify: bool = True) -> Any:
        """``prepare_loop_restrictions(raw, simplify=simplify)``, formed once
        per raw model and simplification."""

        from .model_restrictions import prepare_loop_restrictions

        key = (self._raw_key(raw), simplify)
        found = self._sources.get(key)
        if found is not None:
            self.counts["restriction_hits"] += 1
            return found
        self.counts["restrictions"] += 1
        found = self._sources[key] = prepare_loop_restrictions(
            raw, simplify=simplify, lower=self._lower
        )
        return found

    def _lower(self, raw: Any) -> Any:
        from .scalar_frontend import lower_scalar_program

        self.counts["lowerings"] += 1
        return lower_scalar_program(raw)

    def remember_program(self, raw: Any, simplify: bool, program: Any) -> None:
        """Keep the scalar program ``build_covariant_structure`` lowered for
        restricted declarations that carry none (no restriction card, or
        no simplification), for the next preparation of ``raw``."""

        key = (self._raw_key(raw), simplify)
        found = self._sources.get(key)
        if found is not None and found.program is None:
            if program.library.raw is not found.derived:
                raise ValueError("scalar program belongs to another raw model")
            self.counts["lowerings"] += 1
            self._sources[key] = replace(found, program=program)

    # -- the request model --------------------------------------------------

    def request_model(
        self,
        compiled: Any,
        recipe: dict[str, Any],
        prepare: Any,
    ) -> tuple[Any, Any]:
        """``prepare()`` (the prepared request model and its source-colour
        continuation, or None) once per recipe: ``compiled``'s raw model and
        simplification and ``recipe``, every other input of the preparation
        (JSON). A reused model is handed over to ``compiled`` (its owner)."""

        raw = compiled.raw_loop_model
        simplify = compiled.source.get("options", {}).get("simplify", True)
        try:
            text = json.dumps(recipe, sort_keys=True)
        except TypeError:
            # An input without a JSON form is never keyed: prepared afresh.
            self.counts["requests"] += 1
            return prepare()
        key = (self._raw_key(raw), simplify, text)
        found = self._requests.get(key)
        if found is None:
            self.counts["requests"] += 1
            found = self._requests[key] = prepare()
            return found
        self.counts["request_hits"] += 1
        prepared, continuation = found
        if (
            prepared.compiled_model is not None
            and prepared.compiled_model is not compiled
        ):
            prepared = replace(prepared, compiled_model=compiled)
        return prepared, continuation
