"""Static, independent byte diagnosis of closed run3; no native calls or gate changes."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "source-proposal3"
ROOT = HERE.parents[4]
WORK = ROOT / ".tirust-build/qcd-n3-genuine-qp-diagnostic"
RUN = WORK / "run3"
TAIL_BYTES = (254 + 11) * 8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_trace(path):
    """Independent framing decoder; does not import the failed adjudicator."""
    lines = path.read_text().splitlines()
    header = lines[0].split()
    require(len(header) == 5 and header[0] == "QG1" and lines[-1] == "COMPLETE", "Trace framing")
    observed, count, total, health = map(int, header[1:])
    require(observed == 1 and health == 0, "Observed healthy trace required")
    rows = []
    for number, line in enumerate(lines[1:-1], 1):
        fields = line.split()
        require(len(fields) == 11 and fields[0] == "E", "Record framing")
        seq, call, kind, part, group, so, rank, mode, size = map(int, fields[1:10])
        data = bytes.fromhex(fields[10])
        require(seq == number and len(data) == size, "Record sequence/length")
        rows.append({"seq": seq, "call": call, "kind": kind, "part": part, "bytes": size, "data": data})
    require(len(rows) == count and sum(r["bytes"] for r in rows) == total, "Trace totals")
    return rows


def exact(data, width):
    size = width // 8
    fraction_bits, bias, exponent_bits = (52, 1023, 11) if width == 64 else (112, 16383, 15)
    require(len(data) % size == 0, "IEEE array alignment")
    values = []
    for offset in range(0, len(data), size):
        word = int.from_bytes(data[offset:offset + size], "little")
        exponent = (word >> fraction_bits) & ((1 << exponent_bits) - 1)
        require(exponent != (1 << exponent_bits) - 1, "Nonfinite IEEE input")
        significand = word & ((1 << fraction_bits) - 1)
        significand += (1 << fraction_bits) if exponent else 0
        power = (exponent or 1) - bias - fraction_bits
        sign = -1 if word >> (width - 1) else 1
        values.append(sign * Fraction(significand) * Fraction(2) ** power)
    return values


def promoted64(data):
    out = bytearray()
    for offset in range(0, len(data), 8):
        word = int.from_bytes(data[offset:offset + 8], "little")
        sign = word >> 63
        exponent = (word >> 52) & 2047
        fraction = word & ((1 << 52) - 1)
        require(exponent != 2047 and (exponent != 0 or fraction == 0), "Fixture normal/zero binary64 inputs only")
        value = sign << 127
        if exponent:
            value |= (exponent - 1023 + 16383) << 112
            value |= fraction << 60
        out.extend(value.to_bytes(16, "little"))
    return bytes(out)


def rounded64(data):
    out = bytearray()
    for offset, value in zip(range(0, len(data), 16), exact(data, 128)):
        if value == 0:
            out.extend((int.from_bytes(data[offset:offset + 16], "little") >> 127 << 63).to_bytes(8, "little"))
        else:
            out.extend(struct.pack("<d", float(value)))
    return bytes(out)


def dp_words(data):
    return [{"leg_one_based": i // 4 + 1, "component": i % 4,
             "little_endian_hex": data[8 * i:8 * i + 8].hex(),
             "float_hex": value.hex(), "decimal_repr": repr(value),
             "exact_fraction": str(Fraction(value))}
            for i, value in enumerate(struct.unpack("<20d", data))]


def residual(values):
    return [str(sum(values[4 * leg + component] for leg in (0, 1)) -
                sum(values[4 * leg + component] for leg in (2, 3, 4)))
            for component in range(4)]


def main():
    require(not (HERE / "diagnosis.json").exists(), "Refuse to overwrite diagnosis")
    binding = json.loads((SOURCE / "bindings.json").read_text())
    donor = Path(binding["process"])
    trace = RUN / "b1-m4-o1/trace.txt"
    paths = [SOURCE / "source-index.json", SOURCE / "bindings.json", SOURCE / "observer.f90",
             SOURCE / "driver.f", SOURCE / "analyze.py", trace, RUN / "result.json",
             donor / "improve_ps.f", donor / "born_matrix.f", donor / "loop_matrix.f"]
    paths += [RUN / lane / name for lane in ("b1-m4-o0", "b1-m4-o1")
              for name in ("worker.json", "mapped-runtime.json", "loader.json", "loader.log")]
    before = {str(p): sha(p) for p in paths}
    require(before[str(SOURCE / "source-index.json")] == "fb8a6be7e9e5f24a6f5adb03dcc357563fc1db5fc6434e61f6a1eb03c81e5a45", "Source3 seal changed")
    for name in ("improve_ps.f", "born_matrix.f", "loop_matrix.f"):
        require(before[str(donor / name)] == binding["dependencies"][str(donor / name)], "Donor source pin changed")
    rows = read_trace(trace)
    point_results = []
    for call, point in ((101, 1), (102, 93), (103, 39)):
        chosen = [r for r in rows if r["call"] == call and r["kind"] in (1, 10, 20, 21, 30)]
        require(len(chosen) == 10, "One host/Born/DP/QP/construction before-after pair per saved call")
        pairs = {(r["kind"], r["part"]): r for r in chosen}

        def payload(kind, part):
            return pairs[kind, part]["data"][:-TAIL_BYTES]

        host = payload(1, 0)[:160]
        born = payload(10, 0)[:160]
        dp_before, dp_after = payload(20, 0)[:160], payload(20, 1)[:160]
        qp_before, qp_after = payload(21, 0)[:320], payload(21, 1)[:320]
        born_tail = struct.unpack("<2q", payload(10, 0)[160:176])
        policy = struct.unpack("<11q", payload(1, 0)[160:248])
        require(policy[:4] == (1, 4, point, call), "Saved point identity")
        relations = {
            "Born_before_equals_host_original": born == host,
            "Born_after_equals_Born_before": payload(10, 1)[:176] == payload(10, 0),
            "DP_before_equals_host_original": dp_before == host,
            "Born_before_equals_DP_after_frozen_gate": born == dp_after,
            "QP_before_equals_exact_promotion_of_host_original": qp_before == promoted64(host),
            "QP_before_equals_exact_promotion_of_DP_after": qp_before == promoted64(dp_after),
            "DP_after_equals_nearest_binary64_rounding_of_QP_after": dp_after == rounded64(qp_after),
            "MP_construction_before_equals_QP_after": payload(30, 0)[:320] == qp_after,
            "host_returned_point_equals_host_original": payload(1, 1)[:160] == host,
        }
        require(relations["Born_before_equals_host_original"] and relations["Born_after_equals_Born_before"] and
                relations["DP_before_equals_host_original"] and not relations["Born_before_equals_DP_after_frozen_gate"] and
                relations["QP_before_equals_exact_promotion_of_host_original"] and
                not relations["QP_before_equals_exact_promotion_of_DP_after"] and
                relations["DP_after_equals_nearest_binary64_rounding_of_QP_after"] and
                relations["MP_construction_before_equals_QP_after"] and relations["host_returned_point_equals_host_original"],
                "Unexpected byte relations")
        differences = []
        original, repaired = dp_words(born), dp_words(dp_after)
        for i, (old, new) in enumerate(zip(original, repaired)):
            if old["little_endian_hex"] == new["little_endian_hex"]:
                continue
            difference = exact(dp_after[8 * i:8 * i + 8], 64)[0] - exact(born[8 * i:8 * i + 8], 64)[0]
            differences.append({"word_zero_based": i, "byte_offset_zero_based": i * 8,
                                "leg_one_based": i // 4 + 1, "component": i % 4,
                                "Born_or_DP_before": old, "DP_after": new,
                                "delta_exact": str(difference), "delta_decimal_repr": repr(float(difference))})
        point_results.append({"call_id": call, "saved_point_id": point, "event_sequence": [
            {k: r[k] for k in ("seq", "kind", "part", "bytes")} for r in chosen],
            "payload_offsets": {"Born_momenta": [0, 160], "Born_hel_flavor": [160, 176],
                                "DP_momenta": [0, 160], "DP_keep_flags": [160, 200],
                                "QP_momenta": [0, 320], "QP_keep_flags": [320, 360]},
            "Born_hel_flavor": born_tail,
            "DP_keep_before": struct.unpack("<5q", payload(20, 0)[160:200]),
            "DP_keep_after": struct.unpack("<5q", payload(20, 1)[160:200]),
            "QP_keep_before": struct.unpack("<5q", payload(21, 0)[320:360]),
            "QP_keep_after": struct.unpack("<5q", payload(21, 1)[320:360]),
            "relations": relations, "different_DP_components": len(differences), "differences": differences,
            "original_momenta": original, "repaired_DP_momenta": repaired,
            "QP_before_hex": qp_before.hex(), "QP_after_hex": qp_after.hex(),
            "QP_after_exact_fractions": [str(v) for v in exact(qp_after, 128)],
            "momentum_conservation_residual_exact": {"original": residual(exact(host, 64)),
                                                     "DP_repaired": residual(exact(dp_after, 64)),
                                                     "QP_repaired": residual(exact(qp_after, 128))}})
    canonical = {str(Path(p).resolve()): wanted for p, wanted in binding["runtime"].items()}
    maps = []
    for lane in ("b1-m4-o0", "b1-m4-o1"):
        mapped = json.loads((RUN / lane / "mapped-runtime.json").read_text())
        worker = json.loads((RUN / lane / "worker.json").read_text())
        require(worker["returncode"] == 0 and worker["process_reaped"] is True, "Worker did not finish normally")
        require(mapped["expected_runtime_sha256"] == canonical and mapped["mapped_files_sha256_after_exit"] == canonical,
                "Original donor runtime map/hash mismatch")
        require(set(mapped["mapped_elf_paths"]) == set(canonical) | {str(WORK / "build2/diagnostic")}, "Mapped ELF set mismatch")
        maps.append({"lane": lane, "runtime_library_count": len(canonical), "original_fifteen_paths_and_hashes_preserved": True,
                     "maps_sha256": mapped["maps_sha256"], "worker_returncode": worker["returncode"],
                     "observation_scope": mapped["observation_scope"]})
    sources = {}
    for path, spans in [(donor / "loop_matrix.f", [(795, 810), (918, 945), (973, 982), (2127, 2159)]),
                        (donor / "improve_ps.f", [(1, 38)]), (donor / "born_matrix.f", [(60, 89)]),
                        (SOURCE / "observer.f90", [(125, 172)]), (SOURCE / "driver.f", [(106, 139)]),
                        (SOURCE / "analyze.py", [(190, 204)])]:
        lines = path.read_text().splitlines()
        sources[str(path)] = [{"first_line": first, "last_line": last, "lines": lines[first - 1:last]} for first, last in spans]
    after = {str(p): sha(p) for p in paths}
    require(before == after, "Closed input bytes changed during diagnosis")
    outcome = {"schema": "qcd-n3-run3-static-input-diagnosis-1", "native_or_compiler_or_loader_calls": 0,
               "source3_build2_run3_modified": False, "all_input_bytes_unchanged": True,
               "input_sha256": before, "observed_trace_rows": len(rows), "saved_point_count": 3,
               "diagnosis": "The frozen parser assumes Born uses the DP-repaired point. Actual MadLoop calls Born with original P_USER first, then repairs a separate DP copy and independently repairs a QP promotion of original P_USER. Observed bytes and event order match those source paths. The strict source3 gate remains failed.",
               "ABI_and_layout_assessment": "No evidence of an ABI/layout offset failure: all twenty Born/DP-before components match host bytes; the exact promotion/rounding/construction relations hold for all three points. Wrapper arrays and scalar arguments match generated signatures; logical KEEP flags are observed as unchanged zero words.",
               "point_results": point_results, "actual_worker_map_checks": maps, "source_excerpts": sources,
               "qualification": "Static diagnosis only. No parser gate was changed and no portable numerical adjudication pass is claimed."}
    (HERE / "diagnosis.json").write_text(json.dumps(outcome, indent=2, sort_keys=True) + "\n")
    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(HERE.iterdir()) if p.is_file() and p.name != "index.json"}
    packet = {"schema": "qcd-n3-run3-static-input-analysis-index-1", "scope": "Independent byte/source-order diagnosis of the closed failed run3; no gate changes or native calls.", "files": files}
    (HERE / "index.json").write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"index_sha256": sha(HERE / "index.json"), "payload_count": len(files),
                      "different_DP_components": [p["different_DP_components"] for p in point_results],
                      "worker_maps": maps}, indent=2))


if __name__ == "__main__":
    main()
