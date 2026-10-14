"""Runs, scans and the entry point of the mg7 launcher."""

import gc
import logging
import os
import re
import shutil

from madgraph.iolibs.template_files.mg7.common import logger
from madgraph.various.banner import RunCardMG7
from madgraph.iolibs.template_files.mg7.model_widths import compute_auto_widths
from madgraph.iolibs.template_files.mg7.postprocess import run_lhe_postprocessing, run_selected_tools, selected_tools
from madgraph.iolibs.template_files.mg7.madgraph_process import MadgraphProcess

def _release_channel_generators(process) -> None:
    """Drop the per-channel event/weight generators so their intermediate
    .npy files are deleted right away. madspace only frees them when the
    owning Python objects are collected, and process.event_generator /
    process.phasespaces sit in a reference cycle that plain refcounting
    never breaks -- only an explicit gc.collect() does, promptly, instead of
    leaving it to whenever the cyclic GC next runs on its own."""
    process.event_generator = None
    process.phasespaces = None
    gc.collect()


def run_single(switch=None) -> "MadgraphProcess":
    """Run a single generation and return the process (for its result)."""
    compute_auto_widths()
    process = MadgraphProcess()
    process.survey()
    process.train_madnis()
    process.generate_events()
    if process.gridpack_run_mode != "regular":
        return process
    # run_card-driven LHE post-processing (displaced vertex + systematics)
    run_lhe_postprocessing(process)
    # run the post-processing tools (Pythia8/Delphes/MadSpin/reweight/analysis)
    # selected in the merged question above on the generated events.
    if switch:
        run_selected_tools(switch, process)
    return process


def detect_run_scan(run_card_path):
    """Return a banner.RunCardIterator if the run_card contains scan:[...]
    values, else None."""
    from madgraph.various import banner as banner_mod
    from madgraph.various import misc as _misc
    with _misc.TMP_variable(banner_mod.RunCard, 'allow_scan', True):
        rc = banner_mod.RunCard(run_card_path, consistency=False)
    if getattr(rc, 'scan_set', None):
        return banner_mod.RunCardIterator(run_card_path)
    return None


def detect_param_scan(param_card_path):
    """Return a ParamCardIterator if the param_card contains scan:[...] values,
    else None."""
    if not os.path.exists(param_card_path):
        return None
    from models import check_param_card as param_card_mod
    it = param_card_mod.ParamCardIterator(param_card_path)
    for block in it.order:
        for param in block:
            if isinstance(param.value, str) and param.value.strip().lower().startswith('scan'):
                return it
    return None


def run_scan(iterator, card_path, switch=None) -> None:
    """Iterate over all scan points, running a full generation for each and
    accumulating the results, then write the scan summary. Works for both the
    run_card (RunCardIterator) and the param_card (ParamCardIterator); their
    interface (__iter__/write/store_entry/get_next_name/write_summary) is the
    same. The scan card is restored afterwards."""
    import tomllib
    with open(os.path.join("Cards", "run_card.toml"), "rb") as f:
        run_name = tomllib.load(f).get("run", {}).get("run_name", "run")

    from models import check_param_card as param_card_mod
    is_param_scan = isinstance(iterator, param_card_mod.ParamCardIterator)

    backup = card_path + ".scan_bak"
    shutil.copy(card_path, backup)
    try:
        for i, point in enumerate(iterator):
            point.write(card_path)
            logger.info("=== scan point %d ===", i + 1)
            process = run_single(switch)
            # use the run directory the process actually created, so the
            # per-point params.dat written by write_summary has a home
            name = os.path.basename(process.run_path)
            if is_param_scan:
                # pass the (possibly auto-width-updated) card so the summary
                # records the model-computed width for each 'auto' entry
                iterator.store_entry(name, process.get_result(),
                                     param_card_path=card_path)
            else:
                iterator.store_entry(name, process.get_result())
            _release_channel_generators(process)
        os.makedirs("Events", exist_ok=True)
        summary = os.path.join("Events", "scan_%s.txt" % run_name)
        iterator.write_summary(summary)
        logger.info("scan results written to %s", summary)
    finally:
        shutil.move(backup, card_path)


def run_generation(switch=None) -> None:
    """Run the generation, expanding a scan over the run_card or the param_card
    when one is present (scanning both simultaneously is not allowed)."""
    run_card_path = os.path.join("Cards", "run_card.toml")
    param_card_path = os.path.join("Cards", "param_card.dat")
    run_iter = detect_run_scan(run_card_path)
    param_iter = detect_param_scan(param_card_path)
    if run_iter and param_iter:
        raise RuntimeError(
            "Scanning simultaneously over the run_card and the param_card is "
            "not allowed. Please keep the scan:[...] entries in only one card.")
    if run_iter:
        run_scan(run_iter, run_card_path, switch)
    elif param_iter:
        run_scan(param_iter, param_card_path, switch)
    else:
        _release_channel_generators(run_single(switch))


def _lhe_needed_by(switch, card) -> "str | None":
    """Why the run needs an LHE event file, or None when it does not: every
    post-processing tool run_selected_tools drives and the run_card
    [postprocessing] steps read the LHE file. ``card`` is the raw TOML content
    of the run_card."""
    tools = selected_tools(switch) if switch else []
    if tools:
        return tools[0]
    cfg = card.get("postprocessing", {})
    try:
        if float(cfg.get("time_of_flight", -1.0)) >= 0:
            return "time_of_flight"
    except (TypeError, ValueError):
        pass
    # the legacy systematics.py path only runs when madspace did not compute
    # the weights itself (see run_lhe_postprocessing)
    if cfg.get("systematics") \
            and not card.get("systematics", {}).get("enable", True):
        return "systematics"
    return None


def force_lhe_output_if_needed(switch) -> None:
    """The events are written as npy by default; switch the run_card to the
    LHE format when a selected post-processing needs an LHE file. The card is
    rewritten rather than refused, so that asking for a shower in the launch
    question is enough -- the user does not have to know that it constrains
    output_format.

    The card is read as raw TOML and only its output_format line is edited: a
    run_card scan ("scan:[...]" values) is not a valid RunCardMG7 and must
    survive untouched for run_scan."""
    import tomllib
    path = os.path.join("Cards", "run_card.toml")
    if not os.path.exists(path):
        return
    with open(path, "rb") as f:
        card = tomllib.load(f)
    output_format = card.get("run", {}).get(
        "output_format", RunCardMG7()["run"]["output_format"])
    if output_format == "lhe":
        return
    reason = _lhe_needed_by(switch, card)
    if reason is None:
        return
    with open(path) as f:
        text = f.read()
    text, count = re.subn(r'(?m)^(\s*output_format\s*=\s*)("[^"]*"|\'[^\']*\'|\S+)',
                          r'\g<1>"lhe"', text, count=1)
    if not count:
        text, count = re.subn(r'(?m)^\[run\][^\n]*\n', '\\g<0>output_format = "lhe"\n',
                              text, count=1)
    if not count:
        text += '\n[run]\noutput_format = "lhe"\n'
    with open(path, "w") as f:
        f.write(text)
    logging.getLogger("madevent").info(
        "output_format set to 'lhe' (required by %s).", reason)
