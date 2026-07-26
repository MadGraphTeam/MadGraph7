# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "script_relative",
    (
        Path("docs/result_tables.py"),
        Path("tools/developer/compiled_all_jit_arena_gate.py"),
        Path("tools/developer/four_quark_compiled_gate.py"),
    ),
)
def test_direct_entrypoint_imports_repo_code_only_after_isolated_reexec(
    tmp_path: Path,
    script_relative: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    script = tmp_path / script_relative
    script.parent.mkdir(parents=True)
    shutil.copy2(repository / script_relative, script)

    package = tmp_path / "tools" / "performance_report"
    package.mkdir(parents=True, exist_ok=True)
    if script_relative == Path("docs/result_tables.py"):
        (tmp_path / "src/pyamplicol").mkdir(parents=True)
    sentinel = tmp_path / "repo-imports.txt"
    (package / "runtime_evidence.py").write_text(
        "\n".join(
            (
                "import os",
                "import sys",
                "from pathlib import Path",
                "sentinel = Path(os.environ['PYAMPLICOL_BOOTSTRAP_SENTINEL'])",
                "prefix = Path(sys.pycache_prefix or '.')",
                "with sentinel.open('a', encoding='ascii') as stream:",
                "    stream.write(",
                "        f'{sys.flags.isolated},{sys.flags.no_site},'",
                "        f'{sys.flags.ignore_environment},'",
                "        f'{sys.flags.dont_write_bytecode},'",
                "        f'{int(prefix.exists())}\\n'",
                "    )",
                "class RuntimeEvidenceError(RuntimeError):",
                "    pass",
                "def source_only_bytecode_policy():",
                "    raise RuntimeEvidenceError('stop after authenticated import')",
                "def established_preimport_runtime_identity():",
                "    return {}",
                "def loaded_pyamplicol_origin_policy(*args, **kwargs):",
                "    return {}",
                "def native_extension_in_package(*args, **kwargs):",
                "    return Path('/missing-native')",
                "def preimport_python_runtime_identity(*args, **kwargs):",
                "    return {}",
                "def python_package_tree_identity(*args, **kwargs):",
                "    return {}",
                "",
            )
        ),
        encoding="ascii",
    )

    environment = os.environ.copy()
    environment["PYAMPLICOL_BOOTSTRAP_SENTINEL"] = str(sentinel)
    for name in (
        "PYAMPLICOL_EXACT_PYTHON_REEXEC",
        "PYAMPLICOL_EXACT_IMPORT_PATHS",
        "PYTHONPYCACHEPREFIX",
        "PYTHONDONTWRITEBYTECODE",
    ):
        environment.pop(name, None)
    completed = subprocess.run(
        (sys.executable, str(script), "--help"),
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )

    assert completed.returncode != 0
    assert sentinel.read_text(encoding="ascii").splitlines() == ["1,1,1,1,0"]
