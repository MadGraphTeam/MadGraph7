# SPDX-License-Identifier: 0BSD
"""Spenso's symbolic threads and pyAmpliCol's package-scoped Symbolica key.

The library key that ``pyamplicol`` registers unlocks Symbolica only for frames
of the package. Spenso's Rayon workers carry no such frame and ask for it
through the GIL, which the calling thread holds while it waits for them: a
deadlock (``LoopSlice`` generation of g g > g g g g, design §6.1). The package
therefore keeps Spenso's symbolic work serial unless a key that holds on every
thread licenses Symbolica, asked lazily (``licensing.ensure_spenso_thread_policy``,
called by the model-compilation Symbolica namespaces) on a thread without
frames of the package. Importing the package checks no license
(``test_public_import_laziness``). The end-to-end check without a personal key
is ``test_loop_v2_ct_closure``'s
``test_six_gluon_generation_completes_without_a_personal_key``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

from pyamplicol import licensing

_SOURCE = Path(__file__).resolve().parents[2] / "src"


def _policy_calls(monkeypatch, licensed: bool) -> tuple[list, list]:
    calls: list = []
    frames: list = []

    def is_licensed() -> bool:
        frame = sys._getframe(1)
        while frame is not None:
            frames.append(frame.f_globals.get("__name__"))
            frame = frame.f_back
        return licensed

    tensor = SimpleNamespace(
        SymbolicParallelism=SimpleNamespace(Serial="serial"),
        set_symbolica_rayon_enabled=calls.append,
    )
    monkeypatch.setitem(sys.modules, "symbolica.community.tensor", tensor)
    licensing._apply_spenso_thread_policy(SimpleNamespace(is_licensed=is_licensed))
    return calls, frames


def test_spenso_stays_serial_under_the_package_key_alone(monkeypatch):
    calls, frames = _policy_calls(monkeypatch, licensed=False)
    assert calls == ["serial"]
    # The license is asked on a thread that carries no frame of the package.
    assert frames
    assert not any(str(name).startswith("pyamplicol") for name in frames)


def test_spenso_keeps_its_policy_under_a_thread_wide_key(monkeypatch):
    calls, _frames = _policy_calls(monkeypatch, licensed=True)
    assert calls == []


def test_the_policy_is_applied_by_model_compilation_not_by_the_import():
    """Without a personal key, the import leaves Spenso alone; the first
    model-compilation namespace sets it serial, once."""

    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in ("SYMBOLICA_LICENSE", "SYMBOLICA_MASTER_LICENSE")
    }
    environment["PYTHONPATH"] = os.pathsep.join(
        entry for entry in (str(_SOURCE), environment.get("PYTHONPATH")) if entry
    )
    environment["SYMBOLICA_HIDE_BANNER"] = "1"
    source = textwrap.dedent(
        """
        import symbolica.community.tensor as tensor
        calls = []
        original = tensor.set_symbolica_rayon_enabled
        tensor.set_symbolica_rayon_enabled = lambda policy: calls.append(policy)
        import pyamplicol
        from pyamplicol import licensing
        assert calls == [] and not licensing._SPENSO_POLICY_APPLIED
        from pyamplicol.models import compiler_symbolica, tensors
        compiler_symbolica._ensure_symbolica()
        tensors._ensure_symbolica()
        assert calls == [tensor.SymbolicParallelism.Serial], calls
        assert licensing._SPENSO_POLICY_APPLIED
        print("policy applied lazily")
        """
    )
    completed = subprocess.run(
        (sys.executable, "-c", source),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "policy applied lazily" in completed.stdout
