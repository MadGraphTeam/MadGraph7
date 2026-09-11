"""One-off madspace bootstrap for the mg7 output.

The mg7 launcher needs the compiled ``madspace`` package. A git checkout ships
only the sources, so the first run has to build and install it. That bootstrap
used to live at the top of :mod:`launch.py`, executed as an import side effect.
That was fine while ``launch`` was only ever imported by ``bin/generate_events``
in a throw-away subprocess whose ``sys.argv``/``sys.stdin`` described the run
itself.

``launch`` is now also imported *in process* by MG5's ``launch`` command, where
neither of those is true: ``sys.argv`` is MG5's own command line (so a ``-f``
meant for, say, ``mg5_aMC -f`` would be misread as the run's force flag, and
conversely a ``launch -f`` would not be seen at all) and ``sys.stdin`` is MG5's,
which may be a terminal even when the run itself is scripted.

So the decision is made by the caller and passed in explicitly. Keeping it in a
module of its own -- one that must never import ``madspace`` -- lets MG5 run the
bootstrap *before* importing :mod:`launch`, which is what makes the heavy import
safe to do lazily.
"""

import os
import subprocess
import sys
from pathlib import Path

# Locate the madspace installation bundled alongside MadGraph.
# madgraph/__init__.py lives one level below the MadGraph root, so .parents[1]
# reaches the root and then "madspace/install" is the local install prefix.
import madgraph as _mg_pkg

MG_ROOT = Path(_mg_pkg.__file__).parents[1]
MADSPACE_DIR = MG_ROOT / "madspace"
INSTALL_DIR = MADSPACE_DIR / "install"


def madspace_is_installed() -> bool:
    """True when a compiled madspace is already available to import."""
    return (INSTALL_DIR / "madspace").is_dir()


def ensure_madspace(interactive=None) -> None:
    """Install madspace if needed, then put it on ``sys.path``.

    ``interactive`` says whether the installer may take over the terminal and
    ask questions. ``None`` falls back to auto-detection from ``sys.stdin``,
    which is only correct for ``bin/generate_events``; every in-process caller
    should pass the value explicitly.
    """
    if not madspace_is_installed():
        if interactive is None:
            interactive = sys.stdin.isatty()
        print()
        print("You don't have madspace installed for this madgraph instance")
        print("Running the madspace installation script")
        print()

        install_cmd = [sys.executable, str(MADSPACE_DIR / "install.py")]
        # When the run is non-interactive (scripted / piped), install
        # non-interactively with a source build and default options
        # (--source --yes), and keep the installer away from our stdin (which
        # may carry the run's scripted card-editing commands); when
        # interactive, let it share the terminal so the user can answer.
        install_stdin = None if interactive else subprocess.DEVNULL
        if not interactive:
            install_cmd += ["--source", "--yes"]
        # Expose madgraph on PYTHONPATH so the installer subprocess can import
        # cmd.ask for its prompts.
        install_env = os.environ.copy()
        install_env["PYTHONPATH"] = os.pathsep.join(
            [str(MG_ROOT)]
            + ([install_env["PYTHONPATH"]] if install_env.get("PYTHONPATH") else [])
        )
        result = subprocess.run(install_cmd, env=install_env, stdin=install_stdin)
        if result.returncode != 0:
            raise RuntimeError("madspace installation failed — see output above")

    if str(INSTALL_DIR) not in sys.path:
        sys.path.insert(0, str(INSTALL_DIR))


def drop_install_path() -> None:
    """Take the madspace install directory back off ``sys.path``.

    It is not only the madspace package: the installer puts madspace's build
    dependencies in the same target, so the directory also carries ``yaml``,
    ``packaging``, ``pathspec``, ``scikit_build_core`` and ``pybind11_stubgen``.
    Prepending it therefore shadows any of those the caller already had.

    That was harmless while the launcher only ever ran in a throw-away
    ``bin/generate_events`` process. Now that it is imported into MG5's own
    long-lived session, the shadowing would outlast the run, so drop the entry
    once madspace itself is imported. The madspace package keeps working: its
    submodules resolve through its own ``__path__``, not through ``sys.path``.
    """
    while str(INSTALL_DIR) in sys.path:
        sys.path.remove(str(INSTALL_DIR))
