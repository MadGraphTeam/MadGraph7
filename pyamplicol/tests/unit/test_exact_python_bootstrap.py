# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import pytest

REPORT_ENTRYPOINT = Path("src/pyamplicol/_profiling_campaign/result_tables.py")


def test_report_worker_bootstrap_recovers_repository_venv_under_no_site(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    script = repository / REPORT_ENTRYPOINT
    namespace = runpy.run_path(
        str(script),
        run_name="result_tables_worker_bootstrap_test",
    )
    bootstrap = namespace["_bootstrap_exact_python"]
    globals_ = bootstrap.__globals__
    globals_["COMMAND"] = "_prepare"
    globals_["MEASUREMENT_SOURCE_ROOT"] = None
    globals_["REPOSITORY_ROOT"] = repository
    globals_["ENTRYPOINT"] = script
    for name in (
        "PYAMPLICOL_EXACT_PYTHON_REEXEC",
        "PYAMPLICOL_EXACT_IMPORT_PATHS",
        "PYTHONPYCACHEPREFIX",
        "PYTHONDONTWRITEBYTECODE",
    ):
        monkeypatch.delenv(name, raising=False)

    observed: dict[str, object] = {}

    def capture_execve(
        executable: str,
        arguments: tuple[str, ...],
        environment: dict[str, str],
    ) -> None:
        observed.update(
            {
                "executable": executable,
                "arguments": arguments,
                "environment": environment,
            }
        )
        raise RuntimeError("captured exact worker re-exec")

    monkeypatch.setattr(os, "execve", capture_execve)
    with pytest.raises(RuntimeError, match="captured exact worker re-exec"):
        bootstrap(["_prepare"])

    environment = observed["environment"]
    assert isinstance(environment, dict)
    import_paths = json.loads(environment["PYAMPLICOL_EXACT_IMPORT_PATHS"])
    assert import_paths
    assert all(Path(path).is_relative_to(repository / ".venv") for path in import_paths)
    assert any((Path(path) / "symbolica").is_dir() for path in import_paths)
    assert observed["arguments"][:4] == (
        sys.executable,
        "-I",
        "-S",
        "-B",
    )


@pytest.mark.parametrize(
    "script_relative",
    (
        REPORT_ENTRYPOINT,
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
    if script_relative == REPORT_ENTRYPOINT:
        (tmp_path / "src/pyamplicol").mkdir(parents=True, exist_ok=True)
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
    script = tmp_path / REPORT_ENTRYPOINT
    script.parent.mkdir(parents=True)
    shutil.copy2(repository / REPORT_ENTRYPOINT, script)

    source_package = tmp_path / "src/pyamplicol"
    source_package.mkdir(parents=True, exist_ok=True)
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


def test_class_c_prepare_uses_ancestor_package_with_descendant_tools(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    script_relative = REPORT_ENTRYPOINT
    script = tmp_path / script_relative
    script.parent.mkdir(parents=True)
    shutil.copy2(repository / script_relative, script)

    descendant_package = tmp_path / "src/pyamplicol"
    descendant_package.mkdir(parents=True, exist_ok=True)
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
    script = tmp_path / REPORT_ENTRYPOINT
    script.parent.mkdir(parents=True)
    shutil.copy2(repository / REPORT_ENTRYPOINT, script)
    (tmp_path / "tools/performance_report").mkdir(parents=True)
    (tmp_path / "src/pyamplicol").mkdir(parents=True, exist_ok=True)
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


def test_split_worker_uses_wrapper_tools_and_measured_source_venv(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    wrapper = tmp_path / "wrapper"
    measured = tmp_path / "measured"
    output = tmp_path / "output"
    output.mkdir()

    def initialize_git(root: Path) -> None:
        subprocess.run(("git", "init", "-q"), cwd=root, check=True)
        subprocess.run(
            ("git", "config", "user.email", "test@example.invalid"),
            cwd=root,
            check=True,
        )
        subprocess.run(
            ("git", "config", "user.name", "Test"),
            cwd=root,
            check=True,
        )

    wrapper_entrypoint = wrapper / REPORT_ENTRYPOINT
    wrapper_entrypoint.parent.mkdir(parents=True)
    shutil.copy2(
        repository / REPORT_ENTRYPOINT,
        wrapper_entrypoint,
    )
    wrapper_source = wrapper / "src/pyamplicol"
    wrapper_source.mkdir(parents=True, exist_ok=True)
    (wrapper_source / "__init__.py").write_text(
        "raise RuntimeError('wrapper pyamplicol escaped split routing')\n",
        encoding="ascii",
    )
    wrapper_tools = wrapper / "tools/performance_report"
    wrapper_tools.mkdir(parents=True)
    (wrapper / "tools/__init__.py").write_text("", encoding="ascii")
    (wrapper_tools / "__init__.py").write_text(
        "ORIGIN = 'wrapper-tools'\n",
        encoding="ascii",
    )
    native_name = f"_rusticol{EXTENSION_SUFFIXES[0]}"
    preimport_sentinel = output / "preimport.json"
    (wrapper_tools / "runtime_evidence.py").write_text(
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
                "    Path(os.environ['SPLIT_PREIMPORT_SENTINEL']).write_text(",
                "        json.dumps(payload), encoding='ascii'",
                "    )",
                "    return payload",
                "",
            )
        ),
        encoding="ascii",
    )
    wrapper_legacy = wrapper_tools / "legacy.py"
    wrapper_legacy.write_text(
        "ORIGIN = 'wrapper-legacy-bootstrap-exclusion'\n",
        encoding="ascii",
    )
    (wrapper_tools / "cli.py").write_text(
        "\n".join(
            (
                "import json",
                "import os",
                "import sys",
                "from pathlib import Path",
                "import pyamplicol",
                "import tools.performance_report as report_package",
                "from tools.performance_report import legacy",
                "def main(arguments):",
                "    command = next(",
                "        value for value in arguments",
                "        if value in {'_prepare', '_worker'}",
                "    )",
                "    result_path = Path(",
                "        arguments[arguments.index('--result-json') + 1]",
                "    )",
                "    if command == '_prepare':",
                "        prepared = result_path.parent / 'prepared-model'",
                "        prepared.mkdir(exist_ok=True)",
                "        result_path.write_text(",
                "            json.dumps({'path': str(prepared), 'reused': False}),",
                "            encoding='ascii',",
                "        )",
                "    else:",
                "        result_path.write_text(",
                "            json.dumps({'status': 'ok'}), encoding='ascii'",
                "        )",
                "    payload = {",
                "        'arguments': arguments,",
                "        'cli_origin': __file__,",
                "        'tools_origin': report_package.__file__,",
                "        'legacy_origin': legacy.__file__,",
                "        'legacy_marker': legacy.ORIGIN,",
                "        'pyamplicol_origin': pyamplicol.__file__,",
                "        'pyamplicol_path': list(pyamplicol.__path__),",
                "        'sys_path': list(sys.path),",
                "    }",
                "    sentinel = Path(os.environ['SPLIT_CLI_SENTINEL_ROOT'])",
                "    (sentinel / f'{command}.json').write_text(",
                "        json.dumps(payload), encoding='ascii'",
                "    )",
                "    return 0",
                "",
            )
        ),
        encoding="ascii",
    )
    initialize_git(wrapper)
    subprocess.run(("git", "add", "."), cwd=wrapper, check=True)
    subprocess.run(
        ("git", "commit", "-qm", "authenticated wrapper"),
        cwd=wrapper,
        check=True,
    )

    measured_package = measured / "src/pyamplicol"
    measured_package.mkdir(parents=True)
    (measured / ".gitignore").write_text(".venv/\n", encoding="ascii")
    (measured_package / "__init__.py").write_text(
        "ORIGIN = 'measured-pyamplicol'\n",
        encoding="ascii",
    )
    native_payload = b"authenticated measured native fixture"
    (measured_package / native_name).write_bytes(native_payload)
    measured_decoy = measured / "src/tools/performance_report"
    measured_decoy.mkdir(parents=True)
    (measured / "src/tools/__init__.py").write_text("", encoding="ascii")
    (measured_decoy / "__init__.py").write_text(
        "raise RuntimeError('measured tools escaped wrapper routing')\n",
        encoding="ascii",
    )
    (measured_decoy / "legacy.py").write_text(
        "ORIGIN = 'measured-legacy-without-bootstrap-fix'\n",
        encoding="ascii",
    )
    initialize_git(measured)
    subprocess.run(("git", "add", "."), cwd=measured, check=True)
    subprocess.run(
        ("git", "commit", "-qm", "measured source"),
        cwd=measured,
        check=True,
    )
    subprocess.run(
        (sys.executable, "-m", "venv", "--without-pip", measured / ".venv"),
        check=True,
        timeout=30,
    )
    site_paths = tuple(
        (measured / ".venv").glob("lib/python*/site-packages")
    )
    assert len(site_paths) == 1
    installed_package = site_paths[0] / "pyamplicol"
    installed_package.mkdir()
    (installed_package / "__init__.py").write_text(
        "ORIGIN = 'measured-venv-pyamplicol'\n",
        encoding="ascii",
    )
    (installed_package / native_name).write_bytes(native_payload)

    attacker = tmp_path / "attacker"
    attacker_package = attacker / "tools/performance_report"
    attacker_package.mkdir(parents=True)
    (attacker / "tools/__init__.py").write_text("", encoding="ascii")
    (attacker_package / "__init__.py").write_text(
        "raise RuntimeError('attacker PYTHONPATH was imported')\n",
        encoding="ascii",
    )
    (attacker / "pyamplicol").mkdir()
    (attacker / "pyamplicol/__init__.py").write_text(
        "raise RuntimeError('attacker pyamplicol was imported')\n",
        encoding="ascii",
    )

    def git_value(root: Path, expression: str) -> str:
        return subprocess.run(
            ("git", "rev-parse", expression),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    common_arguments = (
        "--repo-root",
        str(measured),
        "--measurement-source-root",
        str(measured),
        "--expected-measurement-source-revision",
        git_value(measured, "HEAD^{commit}"),
        "--expected-measurement-source-tree",
        git_value(measured, "HEAD^{tree}"),
        "--expected-policy-wrapper-revision",
        git_value(wrapper, "HEAD^{commit}"),
        "--expected-policy-wrapper-tree",
        git_value(wrapper, "HEAD^{tree}"),
        "--expected-policy-entrypoint-sha256",
        digest(wrapper_entrypoint),
        "--expected-legacy-adapter-sha256",
        digest(wrapper_legacy),
        "--study-contract-sha256",
        "a" * 64,
    )
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(attacker),
            "PYTHONHOME": str(attacker),
            "PYTHONSTARTUP": str(attacker / "startup.py"),
            "PYTHONUSERBASE": str(attacker),
            "SPLIT_CLI_SENTINEL_ROOT": str(output),
            "SPLIT_PREIMPORT_SENTINEL": str(preimport_sentinel),
        }
    )
    for command in ("_prepare", "_worker"):
        result_path = output / f"{command}-result.json"
        completed = subprocess.run(
            (
                measured / ".venv/bin/python",
                "-I",
                "-S",
                "-B",
                wrapper_entrypoint,
                *common_arguments,
                command,
                "--result-json",
                result_path,
            ),
            cwd=measured,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=30,
        )
        assert completed.returncode == 0, completed.stderr
        captured = json.loads(
            (output / f"{command}.json").read_text(encoding="ascii")
        )
        assert Path(captured["cli_origin"]).is_relative_to(wrapper_tools)
        assert Path(captured["tools_origin"]).is_relative_to(wrapper_tools)
        assert Path(captured["legacy_origin"]) == wrapper_legacy
        assert captured["legacy_marker"] == (
            "wrapper-legacy-bootstrap-exclusion"
        )
        assert Path(captured["pyamplicol_origin"]).is_relative_to(
            measured_package
        )
        assert {
            Path(path).resolve()
            for path in captured["pyamplicol_path"]
        } == {measured_package.resolve()}
        assert str(attacker) not in captured["sys_path"]
        assert str(wrapper / "src") not in captured["sys_path"]
        assert {
            Path(path).resolve()
            for path in captured["sys_path"]
            if "site-packages" in path
        } == {site_paths[0].resolve()}
        assert result_path.is_file()

    assert json.loads(preimport_sentinel.read_text(encoding="ascii")) == {
        "roots": [str(measured_package)],
        "native_extension": str(measured_package / native_name),
    }

    preimport_sentinel.unlink()
    (output / "_worker.json").unlink()
    (measured_package / "__init__.py").write_text(
        "ORIGIN = 'dirty-measured-pyamplicol'\n",
        encoding="ascii",
    )
    dirty_result = subprocess.run(
        (
            measured / ".venv/bin/python",
            "-I",
            "-S",
            "-B",
            wrapper_entrypoint,
            *common_arguments,
            "_worker",
            "--result-json",
            output / "dirty-result.json",
        ),
        cwd=measured,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )

    assert dirty_result.returncode != 0
    assert (
        "measured-source checkout has tracked changes"
        in dirty_result.stderr
    )
    assert not preimport_sentinel.exists()
    assert not (output / "_worker.json").exists()

    (measured_package / "__init__.py").write_text(
        "ORIGIN = 'measured-pyamplicol'\n",
        encoding="ascii",
    )
    (measured_package / "injected.py").write_text(
        "raise RuntimeError('untracked measured source was imported')\n",
        encoding="ascii",
    )
    untracked_result = subprocess.run(
        (
            measured / ".venv/bin/python",
            "-I",
            "-S",
            "-B",
            wrapper_entrypoint,
            *common_arguments,
            "_worker",
            "--result-json",
            output / "untracked-result.json",
        ),
        cwd=measured,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )

    assert untracked_result.returncode != 0
    assert (
        "measured-source checkout has untracked files in imported "
        "source roots"
        in untracked_result.stderr
    )
    assert not preimport_sentinel.exists()
    assert not (output / "_worker.json").exists()
