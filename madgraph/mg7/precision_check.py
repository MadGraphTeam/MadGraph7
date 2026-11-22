"""The precision check of an mg7 run: a fraction of its events evaluated again
with the matrix elements in double precision.

A run whose [run] precision is not all64 (the default, denom64, keeps only the
momenta and the propagator denominators in double precision) evaluates the
matrix elements in single precision, which is accurate to ~1e-6 on most phase
space points but can lose digits where the amplitudes cancel. With
[run] precision_check > 0 (off by default: it compiles every subprocess a
second time; [run] validation reweights the whole sample through MadtRex
instead), the run builds the CPU library of each MadMatrix subprocess in its
own precision and in all64 (FPTYPE=d) and, once the events are written,
evaluates a sample of them with both, at the same momenta, flavour and alpha_s,
with the same good helicities. With r = M2 / M2_d - 1 per event, it reports:
  * the mean of r: the unweighted events follow dsigma ~ PDF |M2|, so this is
    the relative change of the cross section from the precision;
  * the mean and the maximum of |r|, and the number of events with |r| above
    [run] precision_check_threshold: unstable phase-space regions.

The sample is drawn at random from the whole output (an LHE output is read to
its end, but only the sampled events are parsed). The events are
matched to a library and flavour by their PDG ids (incoming then outgoing,
resonances left out, as in the output), the incoming legs swapped for the
mirrored flavours; events that match none (an AmpliCol subprocess) are not
checked.

Each library is evaluated in a separate Python process (this file run as a
script): the libraries of two precisions export the same C++ symbols, which
must not resolve across them, and the run process has already loaded one of
them through madspace.
"""

import ctypes
import json
import logging
import math
import os
import subprocess
import sys

logger = logging.getLogger("madevent")

RESULT_FILE = "precision_check.json"

UMAMI_SUCCESS = 0
UMAMI_IN_MOMENTA = 0
UMAMI_IN_ALPHA_S = 1
UMAMI_IN_FLAVOR_INDEX = 2
UMAMI_OUT_MATRIX_ELEMENT = 0


# ----------------------------------------------------------------------------
# evaluation (in a separate process, see evaluate())
# ----------------------------------------------------------------------------
def _evaluate_library(library, param_card, parameters, helicity_choice,
                      momenta, alpha_s, flavors):
    """|M|^2 of the CPU MadMatrix library at momenta (shape (n, legs, 4), as
    E, px, py, pz), alpha_s and flavour indices (shape (n,))."""
    import numpy as np

    lib = ctypes.CDLL(os.path.abspath(library), mode=os.RTLD_LOCAL)
    lib.umami_initialize.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    lib.umami_set_parameter.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double, ctypes.c_double]
    lib.umami_free.argtypes = [ctypes.c_void_p]
    lib.umami_matrix_element.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_void_p)]

    handle = ctypes.c_void_p()
    if lib.umami_initialize(ctypes.byref(handle), os.path.abspath(param_card).encode()) \
            != UMAMI_SUCCESS:
        raise RuntimeError(f"{library}: umami_initialize failed with {param_card}")
    try:
        for name, value in parameters.items():
            if lib.umami_set_parameter(handle, name.encode(), float(value), 0.) \
                    != UMAMI_SUCCESS:
                raise RuntimeError(f"{library} refuses the parameter {name}")
        if helicity_choice:
            try:
                set_choice = lib.madmatrix_set_helicity_choice
            except AttributeError:
                set_choice = None
            if set_choice is not None:
                set_choice.argtypes = [
                    ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
                choice = (ctypes.c_int * len(helicity_choice))(*helicity_choice)
                if set_choice(handle, choice, len(helicity_choice)) != UMAMI_SUCCESS:
                    raise RuntimeError(f"{library} refuses the helicity choice")

        count = len(alpha_s)
        # column-major with the event index contiguous: [component][leg][event]
        momenta_in = np.ascontiguousarray(np.transpose(momenta, (2, 1, 0)), dtype=np.float64)
        alpha_s_in = np.ascontiguousarray(alpha_s, dtype=np.float64)
        flavors_in = np.ascontiguousarray(flavors, dtype=np.uint32)
        result = np.zeros(count, dtype=np.float64)
        in_keys = (ctypes.c_int * 3)(UMAMI_IN_MOMENTA, UMAMI_IN_ALPHA_S, UMAMI_IN_FLAVOR_INDEX)
        inputs = (ctypes.c_void_p * 3)(
            momenta_in.ctypes.data, alpha_s_in.ctypes.data, flavors_in.ctypes.data)
        out_keys = (ctypes.c_int * 1)(UMAMI_OUT_MATRIX_ELEMENT)
        outputs = (ctypes.c_void_p * 1)(result.ctypes.data)
        status = lib.umami_matrix_element(
            handle, count, count, 0, 3, in_keys, inputs, 1, out_keys, outputs)
        if status != UMAMI_SUCCESS:
            raise RuntimeError(f"{library}: umami_matrix_element failed ({status})")
        return result
    finally:
        lib.umami_free(handle)


def _evaluate_job(job_path):
    """The worker: evaluate the events of job_path (an npz file and its json
    description) with one library per group, write the results next to it."""
    import numpy as np

    with open(job_path) as f:
        job = json.load(f)
    data = np.load(job["inputs"])
    results = {}
    for name, group in job["groups"].items():
        results[name] = _evaluate_library(
            group["library"], job["param_card"], job["parameters"],
            group["helicity_choice"], data[f"{name}_momenta"],
            data[f"{name}_alpha_s"], data[f"{name}_flavors"])
    np.savez(job["outputs"], **results)


def evaluate(work_dir, tag, groups, inputs, param_card, parameters):
    """|M|^2 of each group of events with its library, evaluated in a separate
    process. groups maps a group name to {"library", "helicity_choice"};
    inputs maps "<name>_momenta"/"_alpha_s"/"_flavors" to arrays."""
    import numpy as np

    inputs_path = os.path.join(work_dir, f"precision_check_{tag}_in.npz")
    outputs_path = os.path.join(work_dir, f"precision_check_{tag}_out.npz")
    job_path = os.path.join(work_dir, f"precision_check_{tag}.json")
    np.savez(inputs_path, **inputs)
    with open(job_path, "w") as f:
        json.dump({"inputs": inputs_path, "outputs": outputs_path, "groups": groups,
                   "param_card": os.path.abspath(param_card),
                   "parameters": parameters}, f)
    try:
        result = subprocess.run([sys.executable, os.path.abspath(__file__), job_path],
                                capture_output=True, text=True)
        if result.returncode != 0:
            message = (result.stderr.strip() or result.stdout.strip()).splitlines()
            raise RuntimeError(message[-1] if message else
                               f"exit status {result.returncode}")
        with np.load(outputs_path) as data:
            return {name: data[name] for name in data.files}
    finally:
        for path in (inputs_path, outputs_path, job_path):
            if os.path.exists(path):
                os.remove(path)


# ----------------------------------------------------------------------------
# events
# ----------------------------------------------------------------------------
def flavor_table(process):
    """{(pdg ids of the legs): (me_path format, flavour index, mirrored)} of
    every MadMatrix subprocess of process. A mirrored flavour is also reached
    with its two incoming legs exchanged, the event's momenta then exchanged
    too."""
    table = {}
    mirrored = {}
    for subproc in process.subprocesses:
        unmerged_meta = getattr(subproc, "unmerged_meta", None)
        if unmerged_meta is None:
            members = [subproc.meta]
        else:
            members = [unmerged_meta[i] for i in subproc.meta["subprocesses"]]
        for meta in members:
            if meta.get("backend") == "amplicol":
                continue
            incoming = len(meta["incoming"])
            for flavor in meta["flavors"]:
                for option in flavor["options"]:
                    key = tuple(option)
                    table.setdefault(key, (meta["me_path"], flavor["index"], False))
                    if flavor.get("mirror") and incoming == 2:
                        swapped = (key[1], key[0]) + key[2:]
                        mirrored.setdefault(
                            swapped, (meta["me_path"], flavor["index"], True))
    for key, value in mirrored.items():
        table.setdefault(key, value)
    return table


def _npy_events(path, count, rng):
    """(pdg ids or None, subprocess and flavour indices or None, momenta,
    alpha_s) of count events of an npy output, drawn at random."""
    import numpy as np

    events = np.load(path, mmap_mode="r", max_header_size=1 << 30)
    total = len(events)
    if total == 0:
        return []
    picked = np.sort(rng.choice(total, size=min(count, total), replace=False))
    sample = events[picked]
    names = sample.dtype.names
    legs = 0
    while f"part{legs + 1}_energy" in names:
        legs += 1
    rows = []
    for event in sample:
        momenta = [[event[f"part{i}_{c}"] for c in ("energy", "px", "py", "pz")]
                   for i in range(1, legs + 1)]
        if "part1_pdg_id" in names:   # lhe_npy
            kept = [i for i in range(legs) if event[f"part{i + 1}_status_code"] in (-1, 1)]
            rows.append(dict(
                pdgs=tuple(int(event[f"part{i + 1}_pdg_id"]) for i in kept),
                momenta=[momenta[i] for i in kept],
                alpha_s=float(event["alpha_qcd"])))
        else:                         # compact_npy
            rows.append(dict(
                indices=(int(event["subprocess_index"]), int(event["flavor_index"])),
                momenta=momenta, alpha_s=float(event["alpha_qcd"])))
    return rows


def _lhe_events(path, count, rng):
    """The pdg ids, momenta and alpha_s of count events of an LHE output,
    drawn at random: one pass over the raw text of the events (reservoir
    sampling), parsing only the ones kept."""
    from madgraph.various import lhe_parser

    texts = []
    lhe = lhe_parser.EventFile(path)
    lhe.parsing = False
    try:
        for seen, text in enumerate(lhe):
            if seen < count:
                texts.append(text)
            else:
                slot = int(rng.integers(0, seen + 1))
                if slot < count:
                    texts[slot] = text
    finally:
        lhe.close()
    rows = []
    for text in texts:
        event = lhe_parser.Event(text)
        kept = [p for p in event if p.status in (-1, 1)]
        rows.append(dict(pdgs=tuple(int(p.pid) for p in kept),
                         momenta=[[p.E, p.px, p.py, p.pz] for p in kept],
                         alpha_s=float(event.aqcd)))
    return rows


def _compact_pdgs(process, subproc_index, flavor_index):
    """The pdg ids of a compact_npy event, from its subprocess and flavour."""
    for subproc in process.subprocesses:
        if subproc.subproc_id != subproc_index:
            continue
        flavor = subproc.meta["flavors"][flavor_index]
        unmerged_meta = getattr(subproc, "unmerged_meta", None)
        if unmerged_meta is not None and "subprocess" in flavor:
            member = unmerged_meta[subproc.meta["subprocesses"][flavor["subprocess"]]]
            flavor = member["flavors"][flavor["flavor"]]
        return tuple(flavor["options"][0])
    return None


# ----------------------------------------------------------------------------
# the check
# ----------------------------------------------------------------------------
def summarize(values, references, threshold):
    """The statistics of r = values / references - 1 over unweighted events
    (see the module docstring)."""
    import numpy as np

    values = np.asarray(values, dtype=float)
    references = np.asarray(references, dtype=float)
    finite = np.isfinite(values) & np.isfinite(references)
    scale = np.abs(references)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(scale > 0, np.abs(values - references) / scale,
                       np.where(values == references, 0., np.inf))
    rel = np.where(finite, rel, np.inf)
    # signed, over the events with a finite, non-zero reference: the events
    # already follow |M|^2, so its mean is the relative change of the cross
    # section (no |M|^2 weight on top)
    usable = finite & (scale > 0)
    signed = (values[usable] - references[usable]) / references[usable]
    # the mean |r| over the same events: one infinitely-off event (counted in
    # non_finite / above_threshold, and in the max) does not hide the others
    return {
        "events": int(len(values)),
        "non_finite": int(np.sum(~finite)),
        "cross_section_bias": float(np.mean(signed)) if len(signed) else 0.,
        "mean_relative_deviation": float(np.mean(np.abs(signed))) if len(signed) else 0.,
        "max_relative_deviation": float(np.max(rel)) if len(rel) else 0.,
        "above_threshold": int(np.sum(rel > threshold)),
        "threshold": float(threshold),
    }


def _json_safe(summary):
    """summary with its non-finite floats as strings ("inf", "nan"): strict
    JSON has no Infinity/NaN."""
    return {key: (repr(value) if isinstance(value, float) and not math.isfinite(value)
                  else value)
            for key, value in summary.items()}


def run(process):
    """The precision check of the events process wrote (see the module
    docstring). Returns its summary, also written to the run directory, or
    None when there was nothing to check. Never fatal to the run."""
    try:
        return _run(process)
    except Exception as error:
        logger.warning("precision check skipped: %s", error)
        return None


def _run(process):
    import numpy as np
    from madgraph.mg7.common import me_parameters

    builds = getattr(process, "precision_check_builds", None)
    events_path = getattr(process, "events_path", None)
    if not builds or not events_path or not os.path.isfile(events_path):
        return None
    (backend, fptype), (ref_backend, ref_fptype) = builds
    run_card = process.run_card["run"]
    total = int(process.run_card["generation"]["events"])
    count = max(1, math.ceil(float(run_card["precision_check"]) * max(total, 1)))
    count = min(count, int(run_card["precision_check_max_events"]))
    threshold = float(run_card["precision_check_threshold"])

    if events_path.endswith(".npy"):
        rows = _npy_events(events_path, count, np.random.default_rng(process.run_seed))
    else:
        rows = _lhe_events(events_path, count, np.random.default_rng(process.run_seed))
    table = flavor_table(process)

    groups = {}
    unmatched = 0
    for row in rows:
        pdgs = row.get("pdgs")
        if pdgs is None:
            pdgs = _compact_pdgs(process, *row["indices"])
        match = table.get(pdgs) if pdgs is not None else None
        if match is None or len(row["momenta"]) != len(pdgs):
            unmatched += 1
            continue
        path_format, flavor_index, swap = match
        momenta = row["momenta"]
        if swap:
            momenta = [momenta[1], momenta[0]] + momenta[2:]
        group = groups.setdefault(path_format, {"momenta": [], "alpha_s": [], "flavors": []})
        group["momenta"].append(momenta)
        group["alpha_s"].append(row["alpha_s"])
        group["flavors"].append(flavor_index)
    if not groups:
        logger.info("precision check: none of the %d events read is from a "
                    "MadMatrix subprocess", len(rows))
        return None

    parameters = me_parameters(process.run_card)
    names = {path_format: f"g{i}" for i, path_format in enumerate(groups)}
    inputs = {}
    for path_format, group in groups.items():
        name = names[path_format]
        inputs[f"{name}_momenta"] = np.array(group["momenta"], dtype=np.float64)
        inputs[f"{name}_alpha_s"] = np.array(group["alpha_s"], dtype=np.float64)
        inputs[f"{name}_flavors"] = np.array(group["flavors"], dtype=np.uint32)

    def libraries(build_backend, build_fptype):
        return {
            names[path_format]: {
                "library": path_format.format(device=build_backend, fptype=build_fptype),
                "helicity_choice": process.helicity_choices.get(path_format),
            }
            for path_format in groups
        }

    work_dir = process.run_path
    values = evaluate(work_dir, fptype, libraries(backend, fptype), inputs,
                      process.param_card_path, parameters)
    references = evaluate(work_dir, ref_fptype, libraries(ref_backend, ref_fptype),
                          inputs, process.param_card_path, parameters)
    order = [names[path_format] for path_format in groups]
    summary = summarize(np.concatenate([values[n] for n in order]),
                        np.concatenate([references[n] for n in order]), threshold)
    summary.update({
        "precision": run_card["precision"],
        "fptype": fptype,
        "reference_fptype": ref_fptype,
        "backend": backend,
        "unmatched_events": unmatched,
        "events_path": os.path.basename(events_path),
    })
    with open(os.path.join(process.run_path, RESULT_FILE), "w") as f:
        json.dump(_json_safe(summary), f, indent=1, allow_nan=False)

    message = ("precision check (%s vs all64, %d events): cross-section bias %.2g, "
               "mean |relative deviation| %.2g, max %.2g; %d above %.0e%s")
    args = (run_card["precision"], summary["events"], summary["cross_section_bias"],
            summary["mean_relative_deviation"], summary["max_relative_deviation"], summary["above_threshold"], threshold,
            ", %d not finite" % summary["non_finite"] if summary["non_finite"] else "")
    if summary["above_threshold"] or summary["non_finite"]:
        logger.warning(message + ". This process may need [run] precision = "
                       "\"all64\" (or \"color32\"); details in %s",
                       *(args + (os.path.join(process.run_path, RESULT_FILE),)))
    else:
        logger.info(message, *args)
    return summary


if __name__ == "__main__":
    _evaluate_job(sys.argv[1])
