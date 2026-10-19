# SPDX-License-Identifier: 0BSD
from unittest.mock import patch

import pytest

from pyamplicol.api import CompatibilityError, Runtime, services
from pyamplicol.runtime import backend as adapters

from .test_api_services import _RuntimeBackend


def test_load_defaults_forward_exact_inputs_and_only_explicit_keywords(tmp_path):
    backend = _RuntimeBackend()
    received = []

    def loader(path, **kwargs):
        received.append(kwargs)
        return backend

    with patch.object(services, "_runtime_loader", loader):
        Runtime.load(tmp_path)
        Runtime.load(
            tmp_path,
            model_parameters={"aS": "1e400"},
            arithmetic="arbitrary",
            precision=70,
            r2_strategy="model",
        )
    assert received[0] == {
        "process": None,
        "model_parameters": None,
        "mute_warnings": False,
    }
    assert received[1] == {
        "process": None,
        "model_parameters": {"aS": "1e400"},
        "mute_warnings": False,
        "arithmetic": "arbitrary",
        "precision": 70,
        "r2_strategy": "model",
    }


def test_legacy_loader_works_without_options_and_rejects_before_binding(tmp_path):
    received = []

    def loader(path, *, process, model_parameters, mute_warnings):
        received.append(path)
        return _RuntimeBackend()

    with patch.object(services, "_runtime_loader", loader):
        Runtime.load(tmp_path)
        with pytest.raises(CompatibilityError, match="load-time arithmetic"):
            Runtime.load(tmp_path, arithmetic="arbitrary")
    assert len(received) == 1


def test_evaluation_default_precision_and_explicit_precision_wrappers():
    class Backend(_RuntimeBackend):
        def evaluate_resolved(self, momenta, **selectors):
            self.selectors = selectors
            return super().evaluate_resolved(momenta, **selectors)

    backend = Backend()
    runtime = Runtime(backend)
    runtime.evaluate([])
    assert backend.selectors["precision"] == 16
    backend.default_precision = 70
    for evaluate in (runtime.evaluate, runtime.evaluate_resolved):
        evaluate([])
        assert backend.selectors["precision"] == 70
        evaluate([], precision=16)
        assert backend.selectors["precision"] == 16
    runtime.evaluate_with_prec([], 40)
    assert backend.selectors["precision"] == 40
    runtime.evaluate_resolved_with_prec([], 40)
    assert backend.selectors["precision"] == 40
    backend.default_precision = False
    with pytest.raises(TypeError, match="positive integer"):
        runtime.evaluate([])


@pytest.mark.parametrize(
    "options",
    [dict(precision=70), dict(arithmetic="arbitrary"), dict(r2_strategy="model")],
)
def test_tree_load_options_rejected_before_native_initialization(tmp_path, options):
    with (
        patch.object(adapters, "_load_native_module", side_effect=AssertionError),
        pytest.raises(CompatibilityError, match="only for virtual artifacts"),
    ):
        adapters.load_runtime_backend(tmp_path, **options)
