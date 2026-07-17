# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from time import sleep

import pytest
import symbolica

from pyamplicol._internal.physics.symbols import SymbolRegistry


def test_symbol_construction_is_serialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    participants = 24
    start = Barrier(participants)
    observations_lock = Lock()
    calls: list[tuple[str, bool]] = []
    active = 0
    max_active = 0

    def fake_symbol(name: str, **attributes: object) -> object:
        nonlocal active, max_active
        with observations_lock:
            calls.append((name, attributes.get("is_real") is True))
            active += 1
            max_active = max(max_active, active)
        sleep(0.01)
        value = object()
        with observations_lock:
            active -= 1
        return value

    monkeypatch.setattr(symbolica, "S", fake_symbol)
    registry = SymbolRegistry(namespace="thread_safety_test")
    model_registry = registry.model("thread-safety-test")

    def construct(index: int) -> object:
        start.wait(timeout=5)
        operation = index % 3
        if operation == 0:
            return registry.symbol("shared")
        if operation == 1:
            return registry.real_symbol("shared")
        return model_registry.symbol("shared")

    with ThreadPoolExecutor(max_workers=participants) as executor:
        results = list(executor.map(construct, range(participants)))

    assert max_active == 1
    assert Counter(calls) == Counter(
        {
            (registry.qualified_name("shared"), False): participants // 3,
            (registry.qualified_name("shared"), True): participants // 3,
            (model_registry.qualified_name("shared"), False): participants // 3,
        }
    )
    assert len({id(result) for result in results}) == participants
