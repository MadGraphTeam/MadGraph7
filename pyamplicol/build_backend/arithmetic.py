# SPDX-License-Identifier: 0BSD
"""Shared GMP, MPFR and MPC libraries for distribution wheels.

GMP, MPFR and MPC are LGPL-3.0-or-later. Distribution wheels link them
dynamically (``shared-arithmetic`` Cargo feature: ``gmp-mpfr-sys``'s
``use-system-libs`` mode) and carry them as separate shared libraries that a
user can replace: the extension's copies are bundled by Maturin's standard
wheel repair (``pyamplicol.libs/`` on Linux, ``pyamplicol.dylibs/`` on macOS)
and the C SDK's copies sit next to its static archive in ``_sdk/lib/``.

The libraries are built here from the very sources ``gmp-mpfr-sys`` (pinned by
``Cargo.lock``) bundles and would otherwise compile statically, with that
crate's configure options and shared instead of static libraries, so every
value is the one of a static developer build. On Linux the libraries get
private sonames (``libgmp-pyamplicol.so.10``, ...): neither the build (where
the system compiler itself loads the system ``libgmp.so.10`` and
``libmpc.so.3``) nor a consumer can pick up another copy by soname.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

CARGO_FEATURE = "shared-arithmetic"
SOURCE_CRATE = "gmp-mpfr-sys"
SONAME_TAG = "pyamplicol"
# Linker order: dependents before their dependencies.
LINK_ORDER = ("mpc", "mpfr", "gmp")
_SHARED = ("--enable-shared", "--disable-static", "--with-pic")
# gmp-mpfr-sys 1.7's build.rs configure options (static there, shared here).
_CONFIGURE: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("gmp", ("--enable-fat",)),
    (
        "mpfr",
        (
            "--enable-thread-safe",
            "--disable-decimal-float",
            "--disable-float128",
            "--with-gmp={prefix}",
        ),
    ),
    ("mpc", ("--with-gmp={prefix}", "--with-mpfr={prefix}")),
)


@dataclass(frozen=True)
class ArithmeticRuntime:
    """Installed shared libraries, keyed by library name (``gmp``, ...)."""

    prefix: Path
    libraries: Mapping[str, Path]

    @property
    def library_directory(self) -> Path:
        return self.prefix / "lib"

    def link_files(self) -> tuple[Path, ...]:
        return tuple(self.libraries[name] for name in LINK_ORDER)

    def cargo_environment(self, encoded_rustflags: str) -> dict[str, str]:
        """Point ``gmp-mpfr-sys``'s system-library mode and the link at us."""

        library = str(self.library_directory)
        flags = [flag for flag in encoded_rustflags.split("\x1f") if flag]
        flags.extend(("-L", f"native={library}"))
        environment = {
            "C_INCLUDE_PATH": str(self.prefix / "include"),
            "CARGO_ENCODED_RUSTFLAGS": "\x1f".join(flags),
            "LIBRARY_PATH": library,
        }
        if sys.platform.startswith("linux"):
            # gmp-mpfr-sys runs version probes linked against the libraries,
            # and the wheel repair resolves the extension's needs here. The
            # private sonames cannot shadow the compiler's own GMP/MPC.
            environment["LD_LIBRARY_PATH"] = library
        return environment


def _jobs() -> str:
    affinity = getattr(os, "sched_getaffinity", None)
    count = len(affinity(0)) if affinity is not None else os.cpu_count()
    return str(max(1, count or 1))


def _run(command: Sequence[str | Path], *, cwd: Path, env: Mapping[str, str]) -> None:
    rendered = [str(part) for part in command]
    completed = subprocess.run(rendered, cwd=cwd, env=dict(env), check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"arithmetic runtime build step failed ({completed.returncode}): "
            + " ".join(rendered)
        )


def _output(command: Sequence[str | Path], *, env: Mapping[str, str]) -> str:
    completed = subprocess.run(
        [str(part) for part in command],
        env=dict(env),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "arithmetic runtime inspection failed: "
            + " ".join(map(str, command))
            + f": {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def locked_source_version(root: Path) -> str:
    """Return the ``gmp-mpfr-sys`` version that ``Cargo.lock`` pins."""

    lock = tomllib.loads((root / "Cargo.lock").read_text(encoding="utf-8"))
    versions = {
        str(package["version"])
        for package in lock.get("package", [])
        if package.get("name") == SOURCE_CRATE
    }
    if len(versions) != 1:
        raise RuntimeError(
            f"Cargo.lock must pin exactly one {SOURCE_CRATE} version, found "
            f"{sorted(versions)}"
        )
    return versions.pop()


def locate_sources(root: Path, environment: Mapping[str, str], target: str) -> Path:
    """Fetch the locked crate graph and return the ``gmp-mpfr-sys`` sources."""

    version = locked_source_version(root)
    _run(["cargo", "fetch", "--locked", "--target", target], cwd=root, env=environment)
    cargo_home = environment.get("CARGO_HOME") or str(Path.home() / ".cargo")
    candidates = sorted(
        path
        for path in (Path(cargo_home) / "registry" / "src").glob(
            f"*/{SOURCE_CRATE}-{version}"
        )
        if (path / "Cargo.toml").is_file()
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one fetched {SOURCE_CRATE} {version} source tree, found "
            f"{[str(path) for path in candidates]}"
        )
    return candidates[0]


def _library_sources(crate: Path) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for name, _ in _CONFIGURE:
        matches = sorted(
            path for path in crate.glob(f"{name}-*-c") if (path / "configure").is_file()
        )
        if len(matches) != 1:
            raise RuntimeError(
                f"{SOURCE_CRATE} must bundle one {name} source tree, found "
                f"{[path.name for path in matches]}"
            )
        sources[name] = matches[0]
    return sources


def _private_soname(soname: str) -> str:
    stem, separator, suffix = soname.partition(".so")
    if not separator or not stem:
        raise RuntimeError(f"unexpected shared-library soname: {soname}")
    return f"{stem}-{SONAME_TAG}.so{suffix}"


def _real_file(link: Path) -> Path:
    """Follow ``libtool``'s relative symlinks without resolving the prefix.

    The install names recorded on macOS use the configured prefix string, so
    the returned path must not canonicalise symlinked parents (``/var``).
    """

    path = link
    seen: set[Path] = set()
    while path.is_symlink():
        if path in seen:
            raise RuntimeError(f"symlink loop at installed library: {link}")
        seen.add(path)
        path = path.parent / os.readlink(path)
    if not path.is_file():
        raise RuntimeError(f"missing installed library: {link}")
    return path


def _install_private_soname(
    library: Path, name: str, environment: Mapping[str, str]
) -> Path:
    """Give an installed Linux library its private soname; return its file."""

    real = _real_file(library / f"lib{name}.so")
    soname = _output(["patchelf", "--print-soname", real], env=environment)
    private = library / _private_soname(soname)
    _run(["patchelf", "--set-soname", private.name, real], cwd=library, env=environment)
    if name == "gmp":
        _run(["patchelf", "--remove-rpath", real], cwd=library, env=environment)
    else:
        _run(["patchelf", "--set-rpath", "$ORIGIN", real], cwd=library, env=environment)
    for path in sorted(library.glob(f"lib{name}.so*")):
        if path != real:
            path.unlink()
    real.rename(private)
    # Linker name only (never loaded by soname, never shipped).
    (library / f"lib{name}.so").symlink_to(private.name)
    return private


def build(
    root: Path, work: Path, environment: Mapping[str, str], *, target: str
) -> ArithmeticRuntime:
    """Build GMP, MPFR and MPC as shared libraries below ``work``."""

    linux = sys.platform.startswith("linux")
    if not linux and sys.platform != "darwin":
        raise RuntimeError(
            "shared arithmetic libraries are built only on Linux and macOS"
        )
    if linux and shutil.which("patchelf", path=environment.get("PATH")) is None:
        raise RuntimeError(
            "distribution wheels on Linux need patchelf (Maturin's wheel repair "
            "uses it as well)"
        )
    sources = _library_sources(locate_sources(root, environment, target))
    if work.exists():
        shutil.rmtree(work)
    prefix = work / "prefix"
    library = prefix / "lib"
    library.mkdir(parents=True)
    build_environment = {
        key: value
        for key, value in environment.items()
        if key not in {"CFLAGS", "CPPFLAGS", "LDFLAGS"}
    }
    if linux:
        build_environment["LD_LIBRARY_PATH"] = str(library)
    jobs = _jobs()
    libraries: dict[str, Path] = {}
    for name, options in _CONFIGURE:
        tree = work / sources[name].name
        shutil.copytree(sources[name], tree, symlinks=True)
        step_environment = dict(build_environment)
        if name != "gmp":
            # As gmp-mpfr-sys: MPFR and MPC take CC and CFLAGS from gmp.h.
            step_environment.pop("CC", None)
        configure = [
            "sh",
            "configure",
            f"--prefix={prefix}",
            *_SHARED,
            *(option.format(prefix=prefix) for option in options),
        ]
        print(f"* Building shared {name} ({sources[name].name})", flush=True)
        _run(configure, cwd=tree, env=step_environment)
        _run(["make", f"-j{jobs}"], cwd=tree, env=step_environment)
        _run(["make", "check", f"-j{jobs}"], cwd=tree, env=step_environment)
        _run(["make", "install"], cwd=tree, env=step_environment)
        for archive in library.glob("*.la"):
            # Without libtool archives, dependants link -l<name> plainly and
            # libtool hardcodes no build-prefix run path into them.
            archive.unlink()
        libraries[name] = (
            _install_private_soname(library, name, build_environment)
            if linux
            else _real_file(library / f"lib{name}.dylib")
        )
        shutil.rmtree(tree)
    return ArithmeticRuntime(prefix=prefix, libraries=libraries)


def stage_sdk_libraries(
    runtime: ArithmeticRuntime, destination: Path, environment: Mapping[str, str]
) -> tuple[str, ...]:
    """Copy the libraries next to the SDK archive; return their file names.

    Linux copies keep their private sonames and ``$ORIGIN`` run paths. macOS
    copies are renamed to ``@rpath/`` install names that resolve through the
    consumer's run path (``rusticol-config --libs``) and are re-signed.
    """

    destination.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    copies: dict[str, Path] = {}
    for name in LINK_ORDER:
        source = runtime.libraries[name]
        copy = destination / source.name
        shutil.copyfile(source, copy)
        copy.chmod(0o755)
        copies[name] = copy
        names.append(copy.name)
    if sys.platform == "darwin":
        for name, copy in copies.items():
            command: list[str | Path] = [
                "install_name_tool",
                "-id",
                f"@rpath/{copy.name}",
            ]
            for dependency in LINK_ORDER:
                if dependency != name:
                    command.extend(
                        (
                            "-change",
                            str(runtime.libraries[dependency]),
                            f"@rpath/{runtime.libraries[dependency].name}",
                        )
                    )
            if name != "gmp":
                command.extend(("-add_rpath", "@loader_path"))
            command.append(copy)
            _run(command, cwd=destination, env=environment)
            _run(
                ["codesign", "--force", "--sign", "-", copy],
                cwd=destination,
                env=environment,
            )
    return tuple(names)
