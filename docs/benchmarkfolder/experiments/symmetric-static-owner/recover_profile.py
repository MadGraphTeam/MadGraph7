#!/usr/bin/env python3
"""Document raw-stream recovery and attribute the completed diagnostic samples."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "madgraph/__init__.py").is_file())
TAG = "full-evaluation-profile-v1"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = HERE / TAG
    work = ROOT / ".benchmark-work/experiments/symmetric-static-owner" / TAG
    raw = work / "perf.data"
    recovered = work / "perf-recovered.data"
    data = bytearray(raw.read_bytes())
    assert data[:8] == b"PERFILE2"
    offset, old_size = struct.unpack_from("<QQ", data, 40)
    assert old_size == 0
    position = offset
    types = Counter()
    while position < len(data):
        record_type, misc, size = struct.unpack_from("<IHH", data, position)
        assert size >= 8 and position + size <= len(data)
        types[record_type] += 1
        position += size
    assert position == len(data) and types[4] == 1
    assert types[9] == 6476
    struct.pack_into("<Q", data, 48, len(data) - offset)
    struct.pack_into("<4Q", data, 72, 0, 0, 0, 0)
    if recovered.exists():
        assert recovered.read_bytes() == data
    else:
        recovered.write_bytes(data)
    assert raw.read_bytes()[offset:] == recovered.read_bytes()[offset:]
    directory = next(work.glob("SubProcesses/P*_*"))
    for filename in ("values.dat", "timed_status.dat", "timing.dat", "dimensions.dat"):
        destination = report / filename
        if destination.exists():
            assert destination.read_bytes() == (directory / filename).read_bytes()
        else:
            shutil.copy2(directory / filename, destination)
    expected = HERE / "qcd3-operation-trace-v1/original-0-values.dat"
    assert (report / "values.dat").read_bytes() == expected.read_bytes()
    statuses = dict(map(int, line.split()) for line in
                    (report / "timed_status.dat").read_text().splitlines())
    assert statuses == {216: 490, 328: 10}
    script = report / "perf-script-recovered-no-inline.txt"
    cycles = Counter()
    counts = Counter()
    leaves = defaultdict(Counter)
    main_addresses = Counter()
    total = samples = 0
    for block in script.read_text().strip().split("\n\n"):
        lines = block.splitlines()
        match = re.search(r"(\d+) cycles:u:", lines[0])
        assert match
        weight = int(match[1])
        total += weight
        samples += 1
        names = [re.sub(r"^\s*[0-9a-f]+ ", "", line).split("+")[0]
                 for line in lines[1:]]
        owner = next((name for name in names if re.match(
            r"(?:mp_)?ml5_\d+_(?:update|terminal|merge|create_loop_coefs)", name)), None)
        if owner is None:
            owner = "reduction" if any("ninja::" in name for name in names) else "other"
        cycles[owner] += weight
        counts[owner] += 1
        leaves[owner][names[0]] += weight
        for line in lines[1:]:
            if "MAIN__+" in line:
                main_addresses[re.search(r"MAIN__\+([^ ]+)", line)[1]] += weight
    assert samples == types[9]
    selected = ("mp_ml5_0_update_wl_2_1_", "mp_ml5_0_terminal_wl_2_1_",
                "mp_ml5_0_terminal_wl_3_1_")
    result = dict(complete=True, acceptance_eligible=False, samples=samples,
        total_sampled_cycle_period=total, timed_status_counts=statuses,
        all_printed_values_accuracy_statuses_exact=True,
        recovered_header_only=True, original_raw_preserved=True,
        record_types=dict(types), raw_sha256=sha(raw),
        recovered_sha256=sha(recovered), raw_data_offset=offset,
        original_header_data_size=0, recovered_header_data_size=len(data) - offset,
        targeted_QP_rank_category_percent=100 * sum(cycles[k] for k in selected) / total,
        owner_categories=[dict(name=name, samples=counts[name],
            percent_sampled_cycles=100 * value / total,
            top_leaves=[dict(name=leaf, percent_all_cycles=100 * weight / total)
                        for leaf, weight in leaves[name].most_common(8)])
            for name, value in cycles.most_common()],
        main_callsite_periods=dict(main_addresses),
        evidence_sha256={p.name: sha(p) for p in report.iterdir() if p.is_file()
                         and p.name != "analysis.json"},
        analyzer_sha256=sha(Path(__file__)),
        limitations="Original recorder finalization exceeded its30-second wait "
                    "after numerical completion; failure is retained in profile.json "
                    "and the immutable original raw stream. Recovery changes only "
                    "header data length and clears absent optional feature sections. "
                    "All6980 event records, including6476 samples and target EXIT, "
                    "are complete with no trailing bytes. Decoding disables costly "
                    "inline source lookup; symbols and DWARF callchains remain. "
                    "Rank categories include every vertex using those original "
                    "routines, so their18.3% is an upper bound for the selected "
                    "sharing vertices. Sampling includes a negligible pre-timer "
                    "stdin-release tail and post-loop output; every raw sample is "
                    "retained. This is a diagnostic concurrent-host profile, not "
                    "acceptance timing or an exact stage-time decomposition.")
    path = report / "analysis.json"
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(samples, result["targeted_QP_rank_category_percent"])


if __name__ == "__main__":
    main()
