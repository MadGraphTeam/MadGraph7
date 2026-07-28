# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "script_relative",
    (
        Path("docs/arxiv/result_tables.py"),
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
    if script_relative == Path("docs/arxiv/result_tables.py"):
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


def test_result_tables_preauthenticates_staged_source_native(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    script = tmp_path / "docs/arxiv/result_tables.py"
    script.parent.mkdir(parents=True)
    shutil.copy2(repository / "docs/arxiv/result_tables.py", script)

    source_package = tmp_path / "src/pyamplicol"
    source_package.mkdir(parents=True)
    native_name = f"_rusticol{EXTENSION_SUFFIXES[0]}"
    (source_package / native_name).write_bytes(b"source native")
    installed_site = tmp_path / "installed"
    installed_package = installed_site / "pyamplicol"
    installed_package.mkdir(parents=True)
    (installed_package / native_name).write_bytes(b"installed native")

    package = tmp_path / "tools/performance_report"
    package.mkdir(parents=True)
    sentinel = tmp_path / "preimport.json"
    (package / "runtime_evidence.py").write_text(
        "\n".join(
            (
                "import json",
                "import os",
                "from pathlib import Path",
                "class RuntimeEvidenceError(RuntimeError):",
                "    pass",
                "def source_only_bytecode_policy():",
                "    return {}",
                "def native_extension_in_package(package):",
                f"    return Path(package) / {native_name!r}",
                "def preimport_python_runtime_identity(roots, *, native_extension):",
                "    payload = {",
                "        'roots': [str(Path(root)) for root in roots],",
                "        'native_extension': str(Path(native_extension)),",
                "    }",
                "    Path(os.environ['PYAMPLICOL_PREIMPORT_SENTINEL']).write_text(",
                "        json.dumps(payload), encoding='ascii'",
                "    )",
                "    raise RuntimeError('stop after preimport capture')",
                "",
            )
        ),
        encoding="ascii",
    )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(installed_site)
    environment["PYAMPLICOL_PREIMPORT_SENTINEL"] = str(sentinel)
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
    assert json.loads(sentinel.read_text(encoding="ascii")) == {
        "roots": [str(source_package)],
        "native_extension": str(source_package / native_name),
    }


@pytest.mark.parametrize(
    "script_relative",
    (
        Path("docs/arxiv/result_tables.py"),
        Path("docs/performance_reports/macbook_M3/result_tables.py"),
        Path("docs/performance_reports/x86_EPYC/result_tables.py"),
    ),
)
def test_class_c_prepare_uses_ancestor_package_with_descendant_tools(
    tmp_path: Path,
    script_relative: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    script = tmp_path / script_relative
    script.parent.mkdir(parents=True)
    shutil.copy2(repository / script_relative, script)

    descendant_package = tmp_path / "src/pyamplicol"
    descendant_package.mkdir(parents=True)
    (descendant_package / "__init__.py").write_text(
        "raise RuntimeError('descendant package was imported')\n",
        encoding="ascii",
    )
    ancestor = tmp_path / "ancestor-runtime"
    ancestor_package = ancestor / "src/pyamplicol"
    ancestor_package.mkdir(parents=True)
    package_sentinel = tmp_path / "ancestor-package.txt"
    (ancestor_package / "__init__.py").write_text(
        "\n".join(
            (
                "import os",
                "from pathlib import Path",
                "Path(os.environ['PYAMPLICOL_ANCESTOR_PACKAGE_SENTINEL']).write_text(",
                "    __file__, encoding='ascii'",
                ")",
                "",
            )
        ),
        encoding="ascii",
    )
    native_name = f"_rusticol{EXTENSION_SUFFIXES[0]}"
    (ancestor_package / native_name).write_bytes(b"ancestor native")

    tools_package = tmp_path / "tools/performance_report"
    tools_package.mkdir(parents=True)
    (tools_package / "__init__.py").write_text("", encoding="ascii")
    preimport_sentinel = tmp_path / "preimport.json"
    (tools_package / "runtime_evidence.py").write_text(
        "\n".join(
            (
                "import json",
                "import os",
                "from pathlib import Path",
                "class RuntimeEvidenceError(RuntimeError):",
                "    pass",
                "def source_only_bytecode_policy():",
                "    return {}",
                "def native_extension_in_package(package):",
                f"    return Path(package) / {native_name!r}",
                "def preimport_python_runtime_identity(roots, *, native_extension):",
                "    payload = {",
                "        'roots': [str(Path(root)) for root in roots],",
                "        'native_extension': str(Path(native_extension)),",
                "    }",
                "    Path(os.environ['PYAMPLICOL_PREIMPORT_SENTINEL']).write_text(",
                "        json.dumps(payload), encoding='ascii'",
                "    )",
                "    return payload",
                "",
            )
        ),
        encoding="ascii",
    )
    cli_sentinel = tmp_path / "descendant-cli.json"
    (tools_package / "cli.py").write_text(
        "\n".join(
            (
                "import json",
                "import os",
                "from pathlib import Path",
                "def main(arguments):",
                "    sentinel = os.environ['PYAMPLICOL_DESCENDANT_CLI_SENTINEL']",
                "    Path(sentinel).write_text(",
                "        json.dumps({'arguments': arguments, 'origin': __file__}),",
                "        encoding='ascii',",
                "    )",
                "    return 0",
                "",
            )
        ),
        encoding="ascii",
    )

    environment = os.environ.copy()
    environment.update(
        {
            "PYAMPLICOL_ANCESTOR_PACKAGE_SENTINEL": str(package_sentinel),
            "PYAMPLICOL_DESCENDANT_CLI_SENTINEL": str(cli_sentinel),
            "PYAMPLICOL_PREIMPORT_SENTINEL": str(preimport_sentinel),
        }
    )
    for name in (
        "PYAMPLICOL_EXACT_PYTHON_REEXEC",
        "PYAMPLICOL_EXACT_IMPORT_PATHS",
        "PYTHONPYCACHEPREFIX",
        "PYTHONDONTWRITEBYTECODE",
    ):
        environment.pop(name, None)
    completed = subprocess.run(
        (
            sys.executable,
            str(script),
            "--class-c-ancestor-runtime-root",
            str(ancestor),
            "prepare-class-c-bridge",
            "--ancestor-revision",
            "a" * 40,
            "--descendant-revision",
            "d" * 40,
            "--impact",
            "hzz-orientation-v1",
        ),
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert Path(package_sentinel.read_text(encoding="ascii")).parent == ancestor_package
    assert json.loads(preimport_sentinel.read_text(encoding="ascii")) == {
        "roots": [str(ancestor_package)],
        "native_extension": str(ancestor_package / native_name),
    }
    cli = json.loads(cli_sentinel.read_text(encoding="ascii"))
    assert Path(cli["origin"]).parent == tools_package
    assert "--class-c-ancestor-runtime-root" in cli["arguments"]


def test_class_c_ancestor_runtime_option_is_prepare_only(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    script = tmp_path / "docs/arxiv/result_tables.py"
    script.parent.mkdir(parents=True)
    shutil.copy2(repository / "docs/arxiv/result_tables.py", script)
    (tmp_path / "tools/performance_report").mkdir(parents=True)
    (tmp_path / "src/pyamplicol").mkdir(parents=True)
    ancestor = tmp_path / "ancestor-runtime"
    (ancestor / "src/pyamplicol").mkdir(parents=True)

    completed = subprocess.run(
        (
            sys.executable,
            str(script),
            "--class-c-ancestor-runtime-root",
            str(ancestor),
            "audit",
        ),
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode != 0
    assert "restricted to prepare-class-c-bridge" in completed.stderr
