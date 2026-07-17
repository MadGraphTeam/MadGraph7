# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"


def _run_isolated(source: str) -> None:
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        entry for entry in (str(SOURCE_ROOT), existing_pythonpath) if entry
    )
    completed = subprocess.run(
        (sys.executable, "-c", textwrap.dedent(source)),
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_root_import_is_minimal_in_a_fresh_interpreter() -> None:
    _run_isolated(
        """
        import sys
        import pyamplicol

        assert pyamplicol.__version__
        assert set(pyamplicol.__all__) == {
            *pyamplicol._PUBLIC_EXPORTS,
            "__version__",
        }
        assert "Generator" in dir(pyamplicol)
        assert "pyamplicol.api" not in sys.modules
        assert "pyamplicol.config" not in sys.modules
        assert not any(
            name == "pyamplicol.models" or name.startswith("pyamplicol.models.")
            for name in sys.modules
        )
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in sys.modules
            for prefix in ("symbolica", "ufo_model_loader")
        )
        """
    )


def test_lightweight_public_exports_do_not_load_model_tooling() -> None:
    _run_isolated(
        """
        import sys
        from pyamplicol import (
            Generator,
            ModelSource,
            ProcessRequest,
            ProcessSet,
            Runtime,
        )
        import pyamplicol.api as public_api
        from pyamplicol.api import ModelSource as ApiModelSource
        from pyamplicol.api.requests import ProcessRequest as DirectProcessRequest

        assert set(public_api.__all__) == set(public_api._PUBLIC_EXPORTS)
        request = ProcessRequest.parse("d d~ > z")
        process_set = ProcessSet((request,))
        assert process_set.requests == (request,)
        assert ModelSource is ApiModelSource
        assert ProcessRequest is DirectProcessRequest
        assert ModelSource.built_in_sm().kind == "built-in-sm"
        assert Generator.__module__ == "pyamplicol.api.services"
        assert Runtime.__module__ == "pyamplicol.api.services"
        assert "pyamplicol.models.loading" not in sys.modules
        assert not any(
            name.startswith("pyamplicol.models.compiler")
            for name in sys.modules
        )
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in sys.modules
            for prefix in ("symbolica", "ufo_model_loader")
        )
        """
    )


def test_compiled_model_export_resolves_to_the_canonical_type() -> None:
    _run_isolated(
        """
        from typing import get_type_hints

        from pyamplicol import ModelSource
        from pyamplicol import CompiledModel as RootCompiledModel
        from pyamplicol.api import CompiledModel as ApiCompiledModel
        from pyamplicol.api.models import CompiledModel as ModelCompiledModel

        assert RootCompiledModel is ModelCompiledModel
        assert ApiCompiledModel is ModelCompiledModel
        assert get_type_hints(ModelSource.compile)["return"] is ModelCompiledModel
        assert "pyamplicol.models.loading" not in __import__("sys").modules
        """
    )


def test_cli_help_construction_does_not_load_model_tooling() -> None:
    _run_isolated(
        """
        import sys
        from pyamplicol.cli.parser import build_parser

        help_text = build_parser().format_help()
        assert "generate" in help_text
        assert "model" in help_text
        assert "pyamplicol.models.loading" not in sys.modules
        assert not any(
            name.startswith("pyamplicol.models.compiler")
            for name in sys.modules
        )
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in sys.modules
            for prefix in ("symbolica", "ufo_model_loader")
        )
        """
    )
