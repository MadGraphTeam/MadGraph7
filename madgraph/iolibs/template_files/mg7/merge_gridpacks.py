#! /usr/bin/env python3
"""Merge gridpacks of different subprocess selections into one.

A run with [phasespace] subprocess_selection integrates only some of the
subprocesses, and saves a gridpack with only their channels. The channels of the
different selections have distinct global names and refer to distinct matrix
elements, so their gridpacks combine by taking the union of their data/. Only
the standard library is needed, so this also works on a machine without
madspace.

    merge_gridpacks.py -o merged GRIDPACK GRIDPACK ...

A gridpack is a directory or a .tar.gz archive of one. When they are complete
gridpacks, the merged one is complete too, with the libraries and executables of
the first.
"""

import argparse
import filecmp
import hashlib
import json
import os
import shutil
import sys
import re
import tarfile
import tempfile
import tomllib

# data.json entries that have to be identical everywhere
_IDENTICAL_DATA = (
    "matrix_elements", "custom_functions", "me_parameters", "source_hash"
)


# What the run cards of the gridpacks to merge may differ in: the selection,
# how and where each was run, and the sections about running the gridpack.
_CARD_SECTIONS_IGNORED = ("gridpack", "cluster")
_CARD_KEYS_IGNORED = {
    "phasespace": ("subprocess_selection",),
    "run": ("run_name", "seed", "device", "cpu_mode", "madspace_cpu_mode",
            "cpu_thread_pool_size", "gpu_thread_pool_size", "verbosity"),
}


class MergeError(Exception):
    """The gridpacks do not belong together."""


def comparable_card(text: str) -> dict:
    """The run card in a header of lhe_meta.json without what may differ."""
    card = tomllib.loads(text)
    for section in _CARD_SECTIONS_IGNORED:
        card.pop(section, None)
    for section, keys in _CARD_KEYS_IGNORED.items():
        for key in keys:
            card.get(section, {}).pop(key, None)
    return card


def merge_lhe_meta(paths: list, output: str) -> None:
    """The LHE header description the gridpacks share. Their run cards differ
    in the selection and the like, which is dropped from the merged one."""
    metas = []
    for path in paths:
        with open(path) as f:
            metas.append(json.load(f))
    cards = []
    for meta in metas:
        cards.append([h for h in meta["headers"] if h["name"] == "MG7RunCard"])
        meta["headers"] = [h for h in meta["headers"] if h["name"] != "MG7RunCard"]
    for path, meta in zip(paths[1:], metas[1:]):
        if meta != metas[0]:
            raise MergeError(f"the LHE header data in {path} differs from {paths[0]}")
    for path, card in zip(paths[1:], cards[1:]):
        if [comparable_card(h["content"]) for h in card] != \
                [comparable_card(h["content"]) for h in cards[0]]:
            raise MergeError(f"the run card in {path} differs from {paths[0]}")
    merged = metas[0]
    for header in cards[0]:
        header = dict(header, content=re.sub(
            r"^subprocess_selection\s*=.*\n", "", header["content"], flags=re.M))
        merged["headers"].append(header)
    with open(output, "w") as f:
        json.dump(merged, f)


def split_channels(path: str) -> dict:
    """The channels of a channels.json as {name: (json text, parsed json)}.
    The text is kept as it is, so that the (large) channel descriptions are
    only parsed to inspect them and never written out again."""
    with open(path) as f:
        text = f.read()
    decoder = json.JSONDecoder()
    channels = {}
    position = text.index("{") + 1
    while True:
        while text[position] in " \t\r\n,":
            position += 1
        if text[position] == "}":
            return channels
        name, position = decoder.raw_decode(text, position)
        position = text.index(":", position) + 1
        while text[position] in " \t\r\n":
            position += 1
        value, end = decoder.raw_decode(text, position)
        channels[name] = (text[position:end], value)
        position = end


def merge_globals(inputs: list, output: str) -> None:
    """Union of the globals.tar archives. A global of the same name in several
    of them (the PDF and alpha_s grids) has to be identical."""
    digests = {}
    with tarfile.open(output, "w") as out:
        for path in inputs:
            with tarfile.open(path) as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    digest = hashlib.sha256(archive.extractfile(member).read()).digest()
                    if member.name in digests:
                        if digests[member.name] != digest:
                            raise MergeError(
                                f"global {member.name} differs between the gridpacks"
                            )
                        continue
                    digests[member.name] = digest
                    out.addfile(member, archive.extractfile(member))


def check_identical(paths: list, what: str) -> None:
    for path in paths[1:]:
        if not filecmp.cmp(paths[0], path, shallow=False):
            raise MergeError(f"{what} differs between {paths[0]} and {path}")


def merge_gridpack_data(inputs: list, output: str) -> None:
    """Merge the data/ directories ``inputs`` into the new directory ``output``.

    They have to describe the same process (same matrix elements, cards,
    madspace and LHE completer), integrate different subprocesses and have been
    saved in the same run mode: with or without the max weights and integrals
    that gridpack.py then takes over.
    """
    if not inputs:
        raise MergeError("no gridpack to merge")
    for name in os.listdir(inputs[0]):
        if name not in ("channels.json", "data.json", "globals.tar", "lhe_meta.json"):
            check_identical(
                [os.path.join(path, name) for path in inputs], f"data/{name}"
            )
    for path in inputs[1:]:
        if sorted(os.listdir(path)) != sorted(os.listdir(inputs[0])):
            raise MergeError(f"{path} holds other data files than {inputs[0]}")

    datas = []
    for path in inputs:
        with open(os.path.join(path, "data.json")) as f:
            datas.append(json.load(f))
    for key in _IDENTICAL_DATA:
        for path, data in zip(inputs[1:], datas[1:]):
            if data.get(key) != datas[0].get(key):
                raise MergeError(f"data.json {key} differs in {path}")

    channels = {}
    estimates = set()
    for path in inputs:
        for name, (text, value) in split_channels(
            os.path.join(path, "channels.json")
        ).items():
            if name in channels:
                raise MergeError(
                    f"channel {name} is in several gridpacks: their subprocess "
                    "selections overlap"
                )
            channels[name] = text
            estimates.add("max_weight" in value)
    if len(estimates) > 1:
        raise MergeError(
            "some gridpacks hold the max weights and integrals of their channels "
            "(run_mode fix_max_weight) and some do not"
        )

    # a gridpack from before the indices were recorded loads every matrix element
    indices = sorted({
        i for data in datas
        for i in data.get("matrix_element_indices", range(len(data["matrix_elements"])))
    })
    os.makedirs(output)
    for name in os.listdir(inputs[0]):
        if name not in ("channels.json", "data.json", "globals.tar", "lhe_meta.json"):
            shutil.copy(os.path.join(inputs[0], name), output)
    merge_lhe_meta([os.path.join(path, "lhe_meta.json") for path in inputs],
                   os.path.join(output, "lhe_meta.json"))
    with open(os.path.join(output, "channels.json"), "w") as f:
        f.write(
            "{" + ",".join(f"{json.dumps(n)}:{t}" for n, t in channels.items()) + "}"
        )
    data = dict(datas[0], matrix_element_indices=indices)
    with open(os.path.join(output, "data.json"), "w") as f:
        json.dump(data, f)
    merge_globals([os.path.join(path, "globals.tar") for path in inputs],
                  os.path.join(output, "globals.tar"))


def open_gridpack(path: str, scratch: str) -> str:
    """The directory of the gridpack at ``path``, extracting an archive into
    ``scratch``."""
    if os.path.isdir(path):
        return os.path.abspath(path)
    target = tempfile.mkdtemp(dir=scratch)
    with tarfile.open(path) as archive:
        archive.extractall(target)
    entries = os.listdir(target)
    return os.path.join(target, entries[0]) if len(entries) == 1 else target


def merge_gridpacks(inputs: list, output: str, compress: bool = False) -> None:
    """Merge the gridpacks ``inputs`` into the new gridpack ``output``."""
    if os.path.exists(output):
        raise MergeError(f"{output} exists already")
    with tempfile.TemporaryDirectory() as scratch:
        directories = [open_gridpack(path, scratch) for path in inputs]
        complete = [os.path.isdir(os.path.join(d, "bin")) for d in directories]
        if any(complete) and not all(complete):
            raise MergeError("cannot merge complete gridpacks with bare data/")
        if all(complete):
            cards = [os.path.join(d, "Cards", "param_card.dat") for d in directories]
            check_identical(cards, "Cards/param_card.dat")
            # the shell of the first one: libraries, packages and executables
            shutil.copytree(directories[0], output, ignore=shutil.ignore_patterns("data"))
        merge_gridpack_data([os.path.join(d, "data") if c else d
                             for d, c in zip(directories, complete)],
                            os.path.join(output, "data"))
    if compress and all(complete):
        shutil.make_archive(output, "gztar", os.path.dirname(os.path.abspath(output)),
                            os.path.basename(output))
        shutil.rmtree(output)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("gridpacks", nargs="+",
                        help="gridpack directories or .tar.gz archives")
    parser.add_argument("-o", "--output", required=True,
                        help="the merged gridpack, which must not exist yet")
    parser.add_argument("--compress", action="store_true",
                        help="write OUTPUT.tar.gz instead of a directory")
    args = parser.parse_args(argv)
    try:
        merge_gridpacks(args.gridpacks, args.output, args.compress)
    except MergeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
