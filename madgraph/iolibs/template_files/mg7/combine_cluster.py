#! /usr/bin/env python3
"""Combine the event files of gridpack runs that differ in their seed.

The jobs of the generation pass of the cluster mode each run the same merged
gridpack with another seed. Their events are concatenated as they are, so the
cross section of the sample is the mean of the job cross sections weighted with
the number of events in each (the mean of the event weights), and its error
follows from the independence of the jobs:

    sigma = sum_i n_i sigma_i / N,    error = sqrt(sum_i n_i^2 error_i^2) / N.

    combine_cluster.py -o OUTPUT_DIR JOB_DIR ...

Each JOB_DIR holds the output of a gridpack run in any of its output formats.
"""

import argparse
import gzip
import json
import os
import contextlib
import re
import shutil
import subprocess
import sys

import numpy as np

from madgraph.various import misc

# status entries that are counts of the jobs and add up
_SUMMED = ("count", "count_after_cuts", "count_after_cuts_opt", "count_opt",
           "count_target", "count_unweighted", "count_invalid_step_values")
# status entries that are weighted means over the jobs
_AVERAGED = ("mean", "mean_abs", "rel_std_dev")
# of the systematics summary: the cross sections and bands, averaged over the
# jobs. The bands of the sum are not quite these means, but the jobs sample the
# same distribution, so they only differ at the level of the statistical error
_SYST_AVERAGED = ("cross_section", "central", "uncertainty_up",
                  "uncertainty_down", "min", "max")

EVENT_FILES = ("events.lhe.gz", "events.lhe", "events.npy")
_ROWS_PER_CHUNK = 1 << 20
_COPY_CHUNK = 1 << 22
_LHE_END = b"</LesHouchesEvents>"
_TAIL = 4096


class CombineError(Exception):
    """The runs cannot be combined."""


def weighted_mean(values, weights):
    total = sum(weights)
    if total <= 0:
        return sum(values) / len(values)
    return sum(v * w for v, w in zip(values, weights)) / total


def combine_status(statuses: list) -> dict:
    """The status (an entry of info.json: the process or one channel) of the
    concatenated samples of the jobs, from the status of each."""
    counts = [s["count_unweighted"] for s in statuses]
    total = sum(counts)
    combined = dict(statuses[0])
    for key in _SUMMED:
        if key in combined:
            combined[key] = sum(s[key] for s in statuses)
    for key in _AVERAGED:
        combined[key] = weighted_mean([s[key] for s in statuses], counts)
    for key in ("error", "error_abs"):
        if total > 0:
            combined[key] = sum((n * s[key]) ** 2
                                for n, s in zip(counts, statuses)) ** 0.5 / total
        else:
            combined[key] = (sum(s[key] ** 2 for s in statuses)) ** 0.5 / len(statuses)
    for key in ("done", "optimized"):
        combined[key] = all(s[key] for s in statuses)
    for key in ("iterations", "iters_without_improvement"):
        combined[key] = max(s[key] for s in statuses)
    return combined


def combine_summaries(summaries: list, counts: list):
    """The systematics summary of the combined sample: event counts add up,
    cross sections and bands are averaged with the number of events."""
    first = summaries[0]
    if isinstance(first, dict):
        combined = {}
        for key, value in first.items():
            values = [s[key] for s in summaries]
            if key == "event_count":
                combined[key] = sum(values)
            elif key in _SYST_AVERAGED and isinstance(value, (int, float)):
                combined[key] = weighted_mean(values, counts)
            else:
                combined[key] = combine_summaries(values, counts)
        return combined
    if isinstance(first, list):
        if any(len(s) != len(first) for s in summaries):
            raise CombineError("the systematics of the jobs differ")
        return [combine_summaries([s[i] for s in summaries], counts)
                for i in range(len(first))]
    return first


def combine_histograms(entries: list, counts: list):
    """The histograms of info.json ("histograms", "event_histograms") of the
    combined sample: the bin values (and bands) are means over the jobs
    weighted with their events, the errors are added like those of the cross
    section."""
    total = sum(counts)
    first = entries[0]
    if isinstance(first, dict):
        combined = dict(first)
        for key, value in first.items():
            values = [e[key] for e in entries]
            if key == "bin_errors":
                combined[key] = [
                    (sum((n * e) ** 2 for n, e in zip(counts, errors)) ** 0.5 / total
                     if total > 0 else sum(e ** 2 for e in errors) ** 0.5 / len(errors))
                    for errors in zip(*values)
                ]
            elif isinstance(value, (list, dict)):
                combined[key] = combine_histograms(values, counts)
        return combined
    if isinstance(first, list):
        if any(len(e) != len(first) for e in entries):
            raise CombineError("the histograms of the jobs differ")
        if first and all(isinstance(v, (int, float)) for v in first):
            return [weighted_mean(values, counts) for values in zip(*entries)]
        return [combine_histograms([e[i] for e in entries], counts)
                for i in range(len(first))]
    return first


def combine_info(infos: list) -> dict:
    """info.json of the combined run from those of the jobs."""
    combined = dict(infos[0])
    combined["process"] = combine_status([i["process"] for i in infos])
    if any(len(i["channels"]) != len(infos[0]["channels"]) for i in infos):
        raise CombineError("the jobs have different channels")
    combined["channels"] = [
        combine_status([i["channels"][k] for i in infos])
        for k in range(len(infos[0]["channels"]))
    ]
    counts = [i["process"]["count_unweighted"] for i in infos]
    if "systematics" in infos[0]:
        combined["systematics"] = combine_summaries(
            [i["systematics"] for i in infos], counts)
    for key in ("histograms", "event_histograms"):
        if key in infos[0]:
            # a job without events has no bin values: it adds nothing
            jobs = [(i[key], n) for i, n in zip(infos, counts) if n > 0] \
                or [(i[key], n) for i, n in zip(infos, counts)]
            combined[key] = combine_histograms([h for h, _ in jobs], [n for _, n in jobs])
    combined["run_times"] = {
        step: {key: sum(i["run_times"][step][key] for i in infos)
               for key in infos[0]["run_times"][step]}
        for step in infos[0]["run_times"]
    }
    combined["seed"] = [i["seed"] for i in infos]
    combined["job_count"] = len(infos)
    return combined


def open_text(path: str, mode: str = "rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def patch_init(lines: list, status: dict, negative: bool) -> list:
    """The lines of an <init> block of one process, with the cross section,
    error and maximal weight of ``status`` and the weight convention IDWTUP of
    a sample that has (-4) or has not (3) negative weights."""
    if len(lines) != 2:
        raise CombineError("only an <init> block of a single process is supported")
    beams = lines[0].split()
    beams[8] = "-4" if negative else "3"
    return [
        " ".join(beams) + "\n",
        "%.10e %.10e %.10e %s\n" % (
            status["mean"], status["error"], status["mean_abs"], lines[1].split()[3]),
    ]


def has_negative_weights(statuses: list) -> bool:
    return any(s["mean"] < 0 or s["mean_abs"] > abs(s["mean"]) * (1 + 1e-12)
               for s in statuses)


def init_of(path: str) -> list:
    """The two lines inside <init> of an LHE file or header."""
    with open_text(path) as f:
        lines = []
        inside = False
        for line in f:
            if line.startswith("</init>"):
                return lines
            if inside:
                lines.append(line)
            elif line.startswith("<init>"):
                inside = True
    raise CombineError(f"no <init> block in {path}")


@contextlib.contextmanager
def compressed_output(path: str):
    """A binary file to write ``path`` with, compressed by the gzip tool
    misc.configure_gzip chose (pigz: on several cores) when it ends in .gz."""
    tool = shutil.which(misc._gzip_tool) if misc._gzip_tool else None
    if not path.endswith(".gz") or tool is None:
        with (gzip.open(path, "wb", compresslevel=misc.GZIP_COMPRESSLEVEL)
              if path.endswith(".gz") else open(path, "wb")) as f:
            yield f
        return
    command = [tool, "-c", "-%d" % misc.GZIP_COMPRESSLEVEL]
    if misc._gzip_tool_supports_multithreading and misc._gzip_tool_max_cores:
        command += ["-p", str(misc._gzip_tool_max_cores)]
    with open(path, "wb") as out:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=out)
        try:
            yield process.stdin
        except BaseException:
            process.kill()
            process.wait()
            os.remove(path)
            raise
        process.stdin.close()
        if process.wait() != 0:
            os.remove(path)
            raise CombineError(f"compressing {path} with {tool} failed")


def copy_events(source, out) -> None:
    """Copy the rest of an LHE file, the events, without its closing tag."""
    carry = b""
    while True:
        chunk = source.read(_COPY_CHUNK)
        if not chunk:
            break
        data = carry + chunk
        # keep the tail back: the closing tag, and what follows it, may be cut
        # by the block boundary
        out.write(data[:-_TAIL])
        carry = data[-_TAIL:]
    end = carry.rfind(_LHE_END)
    out.write(carry if end < 0 else carry[:end])


def write_lhe(paths: list, output: str, status: dict, negative: bool) -> None:
    """Concatenate the events of the LHE files ``paths``, with the header of
    the first and the cross section of ``status``. The events are copied in
    blocks, not parsed."""
    with compressed_output(output) as out:
        for index, path in enumerate(paths):
            with open_text(path, "rb") as f:
                init = None
                for line in f:
                    if init is not None and not line.startswith(b"</init>"):
                        init.append(line.decode())
                    elif line.startswith(b"<init>"):
                        init = []
                        if index == 0:
                            out.write(line)
                    elif line.startswith(b"</init>"):
                        if index == 0:
                            out.write("".join(patch_init(init, status, negative)).encode())
                            out.write(line)
                        break
                    elif index == 0:
                        out.write(line)
                else:
                    raise CombineError(f"no <init> block in {path}")
                copy_events(f, out)
        out.write(_LHE_END + b"\n")


def write_header(path: str, output: str, status: dict, negative: bool) -> None:
    """header.lhe of the combined run: that of the first job, with the cross
    section of ``status``."""
    with open(path) as f:
        text = f.read()
    match = re.search(r"(<init>\n)(.*?)(</init>)", text, flags=re.S)
    if match is None:
        raise CombineError(f"no <init> block in {path}")
    init = patch_init(match.group(2).splitlines(keepends=True), status, negative)
    with open(output, "w") as f:
        f.write(text[:match.start(2)] + "".join(init) + text[match.end(2):])


def concatenate_npy(paths: list, output: str) -> None:
    """Concatenate the structured arrays in the npy files ``paths``."""
    arrays = [np.load(path, mmap_mode="r") for path in paths]
    if any(a.dtype != arrays[0].dtype for a in arrays):
        raise CombineError("the event files have different columns")
    combined = np.lib.format.open_memmap(
        output, mode="w+", dtype=arrays[0].dtype, shape=(sum(len(a) for a in arrays),))
    start = 0
    for array in arrays:
        for chunk in range(0, len(array), _ROWS_PER_CHUNK):
            part = array[chunk:chunk + _ROWS_PER_CHUNK]
            combined[start:start + len(part)] = part
            start += len(part)
    combined.flush()


def event_file_of(job_dir: str) -> str:
    for name in EVENT_FILES:
        if os.path.exists(os.path.join(job_dir, name)):
            return name
    raise CombineError(f"no event file in {job_dir}")


def combine_runs(job_dirs: list, output: str) -> dict:
    """Combine the gridpack runs in ``job_dirs`` into the directory ``output``,
    and return the info.json of the result."""
    if not job_dirs:
        raise CombineError("no runs to combine")
    names = {event_file_of(d) for d in job_dirs}
    if len(names) > 1:
        raise CombineError(f"the runs have different event files: {sorted(names)}")
    name = names.pop()
    infos = []
    for job_dir in job_dirs:
        with open(os.path.join(job_dir, "info.json")) as f:
            infos.append(json.load(f))
        if infos[-1].get("status") != "done":
            raise CombineError(f"{job_dir} is not finished")
    info = combine_info(infos)
    status = info["process"]
    negative = has_negative_weights([i["process"] for i in infos])

    os.makedirs(output, exist_ok=True)
    paths = [os.path.join(d, name) for d in job_dirs]
    if name.startswith("events.lhe"):
        write_lhe(paths, os.path.join(output, name), status, negative)
    else:
        concatenate_npy(paths, os.path.join(output, name))
        header = os.path.join(job_dirs[0], "header.lhe")
        if os.path.exists(header):
            write_header(header, os.path.join(output, "header.lhe"), status, negative)
        completer = os.path.join(job_dirs[0], "lhe_completer.json")
        if os.path.exists(completer):
            shutil.copy(completer, output)

    sidecars = [os.path.join(d, "events.weights.json") for d in job_dirs]
    if all(os.path.exists(p) for p in sidecars):
        datas = []
        for path in sidecars:
            with open(path) as f:
                datas.append(json.load(f))
        counts = [i["process"]["count_unweighted"] for i in infos]
        combined = combine_summaries(datas, counts)
        with open(os.path.join(output, "events.weights.json"), "w") as f:
            json.dump(combined, f, indent=1)

    with open(os.path.join(output, "info.json"), "w") as f:
        json.dump(info, f)
    return info


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", help="output directories of gridpack runs")
    parser.add_argument("-o", "--output", required=True,
                        help="directory of the combined run")
    args = parser.parse_args(argv)
    try:
        info = combine_runs(args.runs, args.output)
    except CombineError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    process = info["process"]
    print("combined %d runs: %d events, cross section %.6g +- %.2g pb" % (
        len(args.runs), process["count_unweighted"], process["mean"], process["error"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
