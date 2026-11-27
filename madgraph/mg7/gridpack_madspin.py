"""MadSpin in an mg7 gridpack ([gridpack] include_madspin).

A gridpack that carries MadSpin decays the events of each of its runs itself,
without a MadGraph installation and without a compiler. When the gridpack is
made, MadSpin prepares an ms_dir for it -- the decay channels as mg7
gridpacks, the compiled density matrix elements, the maximum weights -- from a
production sample (MadSpin/gridpack_decay.py, ``prepare``), and the Python it
runs with (madgraph, MadSpin, madmatrix, aloha, the UFO model) is copied next
to it::

    <gridpack>/madspin/ms_dir/     the prepared ms_dir, with the card it runs
    <gridpack>/madspin/python/     what goes on PYTHONPATH to run it
    <gridpack>/Cards/madspin_card.dat   the card, for the record

bin/generate_events then runs ``MadSpin.gridpack_decay decay`` on its LHE
events (see template_files/mg7/gridpack.py).
"""

import logging
import os
import shutil
import subprocess
import sys
import tempfile

# the launcher's logger (madgraph.mg7.common), without importing madspace
logger = logging.getLogger("madgraph7")

MADSPIN_DIR = "madspin"
MS_DIR = os.path.join(MADSPIN_DIR, "ms_dir")
PYTHON_DIR = os.path.join(MADSPIN_DIR, "python")
CARD = os.path.join("Cards", "madspin_card.dat")
# events of the production sample MadSpin is prepared with, when the run made
# none (gridpack run_mode minimal/fix_max_weight, the cluster mode): enough for
# the maximum-weight probe (75 events by default) and its margin
SAMPLE_EVENTS = 1000

_CARD_HEADER = """\
# The madspin card this gridpack decays its events with.
# Modifying this file has no effect: MadSpin was prepared for it when the
# gridpack was made (decay channels, matrix elements, maximum weights) and runs
# the copy in madspin/ms_dir. The event file, the ms_dir (madspin/ms_dir,
# relative to the gridpack) and the seed are given by bin/generate_events.
# bin/generate_events --no_madspin skips the decays, --keep_undecayed keeps
# the undecayed events next to the decayed ones.
"""


class GridpackMadSpinError(Exception):
    """MadSpin cannot be included in the gridpack."""


def madspin_wanted(run_card, madspin_switch=None, cards_dir="Cards") -> bool:
    """Whether the gridpack carries MadSpin: [gridpack] include_madspin, whose
    'auto' follows the madspin switch of the launch (when known) or else the
    presence of the madspin card, as the madspin step of a run does."""
    setting = run_card["gridpack"]["include_madspin"]
    has_card = os.path.exists(os.path.join(cards_dir, "madspin_card.dat"))
    if setting == "auto":
        if madspin_switch is not None:
            return bool(madspin_switch) and has_card
        return has_card
    if setting and not has_card:
        raise GridpackMadSpinError(
            "[gridpack] include_madspin is on but there is no "
            "Cards/madspin_card.dat to prepare MadSpin with: switch madspin "
            "on (or write the card), or set include_madspin = false")
    return bool(setting)


def model_dir(cards_dir="Cards") -> str:
    """The directory of the UFO model of the process (Cards/proc_card_mg5.dat),
    which MadSpin imports."""
    from madgraph.various import banner
    from madgraph.models import import_ufo
    card = banner.ProcCard(os.path.join(cards_dir, "proc_card_mg5.dat"))
    name = card.info.get("model") or "sm"
    path, _, _ = import_ufo.get_path_restrict(name)
    return os.path.realpath(path)


def _python_env(*paths) -> dict:
    env = os.environ.copy()
    entries = [str(p) for p in paths if p]
    entries += [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def _run(command, cwd, env, log_path, what):
    """Run ``command``; on failure raise with the tail of its log, which the
    caller keeps (see add_madspin)."""
    with open(log_path, "w") as log:
        returncode = subprocess.call(command, cwd=cwd, env=env, stdout=log,
                                     stderr=subprocess.STDOUT)
    if returncode:
        try:
            with open(log_path) as f:
                tail = "\n".join(f.read().splitlines()[-30:])
        except OSError:
            tail = ""
        kept = os.path.join(os.path.dirname(os.path.dirname(log_path)),
                            "madspin_" + os.path.basename(log_path))
        raise GridpackMadSpinError(
            f"{what} failed (exit {returncode}), see {kept}:\n{tail}")


def production_sample(gridpack_path, work_dir, seed, madspace_path=None,
                      threads=-1, events=SAMPLE_EVENTS) -> str:
    """Generate ``events`` LHE events with the gridpack: the sample MadSpin is
    prepared with when the run that made the gridpack wrote no events."""
    output = os.path.join(work_dir, "sample")
    command = [sys.executable, os.path.join(gridpack_path, "bin", "generate_events"),
               "--events", str(events), "--seed", str(seed),
               "--output_dir", output, "--output_format", "lhe",
               "--verbosity", "log", "--temp_output_dir", "auto",
               "--no_madspin"]
    if threads and threads > 0:
        command += ["--cpu_thread_pool_size", str(threads)]
    logger.info("MadSpin in the gridpack: generating %d events with the gridpack "
                "to prepare the decays", events)
    _run(command, gridpack_path, _python_env(madspace_path),
         os.path.join(work_dir, "sample.log"), "generating the production sample")
    for name in ("events.lhe.gz", "events.lhe"):
        path = os.path.join(output, name)
        if os.path.exists(path):
            return path
    raise GridpackMadSpinError(f"the gridpack wrote no LHE events in {output}")


def add_madspin(gridpack_path, seed, events_path=None, madspace_path=None,
                threads=-1, cards_dir="Cards") -> None:
    """Prepare MadSpin in ``gridpack_path`` (see the module docstring).

    ``events_path``: LHE events of the process to prepare the decays with;
    None (or an npy file) makes a sample with the gridpack. ``seed``: the
    seed of the run that makes the gridpack. ``madspace_path``: the directory
    holding the madspace package, for the gridpack and the mg7 decay
    gridpacks when they do not carry one."""
    import madgraph
    from MadSpin import gridpack_decay
    from MadSpin.gridpack_decay import madspin_seed

    mg_root = os.path.dirname(os.path.dirname(os.path.abspath(madgraph.__file__)))
    card = os.path.abspath(os.path.join(cards_dir, "madspin_card.dat"))
    with open(card) as f:
        try:
            body = gridpack_decay.gridpack_card(f.read())
        except gridpack_decay.GridpackDecayError as error:
            raise GridpackMadSpinError(str(error)) from error

    gridpack_path = os.path.abspath(gridpack_path)
    ms_dir = os.path.join(gridpack_path, MS_DIR)
    if os.path.exists(os.path.join(gridpack_path, MADSPIN_DIR)):
        shutil.rmtree(os.path.join(gridpack_path, MADSPIN_DIR))
    work = tempfile.mkdtemp(prefix="madspin_gridpack_",
                            dir=os.path.dirname(gridpack_path))
    failed = True
    try:
        if not events_path or not events_path.endswith((".lhe", ".lhe.gz")):
            events_path = production_sample(gridpack_path, work, seed,
                                             madspace_path, threads)
        logger.info("MadSpin in the gridpack: preparing the decays in %s", ms_dir)
        command = [sys.executable, "-m", "MadSpin.gridpack_decay", "prepare",
                   "--card", card, "--events", os.path.abspath(events_path),
                   "--ms_dir", ms_dir, "--seed", str(madspin_seed(seed))]
        if threads and threads > 0:
            command += ["--nb_core", str(threads)]
        _run(command, os.getcwd(), _python_env(mg_root, madspace_path),
             os.path.join(work, "prepare.log"), "preparing MadSpin")
        # the prepare log is the record of what was built
        shutil.copy(os.path.join(work, "prepare.log"),
                    os.path.join(gridpack_path, MADSPIN_DIR, "prepare.log"))
        failed = False
    finally:
        if failed:
            # keep the logs of what failed next to the gridpack
            for name in ("sample.log", "prepare.log"):
                if os.path.exists(os.path.join(work, name)):
                    shutil.copy(os.path.join(work, name), os.path.join(
                        os.path.dirname(gridpack_path), "madspin_" + name))
        shutil.rmtree(work, ignore_errors=True)

    gridpack_decay.install_runtime(os.path.join(gridpack_path, PYTHON_DIR),
                                   model_dir(cards_dir), mg_root)
    with open(os.path.join(gridpack_path, CARD), "w") as f:
        f.write(_CARD_HEADER + "# set ms_dir madspin/ms_dir\n" + body)
    with open(os.path.join(gridpack_path, "Cards", "grid_run_card.toml"), "a") as f:
        f.write("\n[madspin]\n"
                "enable = true # decay the events with the prepared MadSpin "
                "(madspin/ms_dir); bin/generate_events --no_madspin skips it\n")
    logger.info("MadSpin in the gridpack: done, the gridpack decays its events "
                "(%s)", ", ".join(line.strip() for line in body.splitlines()
                                  if line.strip().startswith("decay")))
