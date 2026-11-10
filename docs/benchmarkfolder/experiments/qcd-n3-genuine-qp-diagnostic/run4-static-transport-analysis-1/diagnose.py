"""Read-only diagnosis of the closed run4 decimal-output rejection.

No numerical executable, compiler, loader or formatter is invoked. Source4,
build2 and run4 remain unchanged. This script does not propose a parser change.
"""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import tarfile
import tomllib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SOURCE = HERE.parent / "source-proposal4"
WORK = ROOT / ".tirust-build/qcd-n3-genuine-qp-diagnostic"
RUN = WORK / "run4"
CARGO = ROOT / ".tirust-build/cargo-home"
NRS_FLOAT = CARGO / "git/checkouts/symbolica-7e92af2066df6214/9cdc294/lib/numerica/src/domains/float/multiprecision.rs"
TIR_FLOAT = CARGO / "git/checkouts/symbolica-cc465e8cf60c7b8e/dfe6bd9/lib/numerica/src/domains/float/multiprecision.rs"
NRS_REEXPORT = CARGO / "git/checkouts/ninjars-3842d69907c1a3a3/41a4c00/crates/ninjars/src/lib.rs"
RUG_SOURCE = Path("/home/external/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/rug-1.30.0")
RUG_CRATE = Path("/home/external/.cargo/registry/cache/index.crates.io-1949cf8c6b5b557f/rug-1.30.0.crate")
PATTERN = re.compile(rb"[+-]?(?:\d+\.?\d*|\.\d+)[EeDd][+-]?\d+")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def span(path, first, last):
    return {"path": str(path), "first_line": first, "last_line": last,
            "lines": path.read_text().splitlines()[first - 1:last]}


def main():
    require(not (HERE / "diagnosis.json").exists(), "Refuse to overwrite diagnosis")
    binding = json.loads((SOURCE / "bindings.json").read_bytes())
    donor = Path(binding["process"])
    tir_install = ROOT / ".tirust-build/qp-transport-install/mg/HEPTools/tirust"
    paths = [SOURCE / "source-index.json", SOURCE / "analyze.py", SOURCE / "observer.c",
             RUN / "result.json", RUN / "b8-m4-o1/trace.txt", RUN / "b8-m4-o1/worker.json",
             RUN / "b8-m4-o1/mapped-runtime.json", ROOT / "vendor/ninjars/src/lib.rs",
             ROOT / "vendor/ninjars/mninjars.f90", ROOT / "vendor/ninjars/ninjars.h",
             ROOT / "vendor/ninjars/Cargo.toml", ROOT / "vendor/ninjars/Cargo.lock",
             ROOT / "vendor/tirust/src/lib.rs", ROOT / "vendor/tirust/mtirust.f90",
             ROOT / "vendor/tirust/Cargo.toml", ROOT / "vendor/tirust/Cargo.lock",
             ROOT / "HEPTools/ninjars/include/mninjars.f90", ROOT / "HEPTools/ninjars/include/ninjars.h",
             ROOT / "HEPTools/ninjars/lib/ninjars-config.json", ROOT / "HEPTools/ninjars/lib/libninjars.a",
             tir_install / "include/mtirust.f90", tir_install / "lib/tirust-config.json",
             tir_install / "lib/libmadloop_tirust.so", donor / "CT_interface.f",
             NRS_FLOAT, TIR_FLOAT, NRS_REEXPORT, RUG_SOURCE / "src/float/big.rs",
             RUG_SOURCE / "src/float/traits.rs", RUG_CRATE]
    paths += [p.parent.parent / "backend/float.rs" for p in (NRS_FLOAT, TIR_FLOAT)]
    before = {str(p): sha(p) for p in paths}
    require(sha(SOURCE / "source-index.json") == "34b185b07b4518001deff43c534b48a1dbb958256721dfe746374fe582e52d2d", "Source4 seal mismatch")
    for path in [ROOT / "HEPTools/ninjars/include/mninjars.f90", ROOT / "HEPTools/ninjars/lib/libninjars.a", donor / "CT_interface.f"]:
        require(sha(path) == binding["dependencies"][str(path)], "Production NinjaRS input pin mismatch")
    baseline_blobs = {"vendor/ninjars/mninjars.f90": "477bc83bea9ea65d9a326fc70797020259379008",
                      "vendor/ninjars/ninjars.h": "d2d841f3198a0f98b4e0ffc366df726ae3294002",
                      "vendor/ninjars/src/lib.rs": "f6b79036f1c3fba0f2570e6e27d8509959b6bbd1"}
    for name, blob in baseline_blobs.items():
        raw = (ROOT / name).read_bytes()
        require(hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() == blob, "NinjaRS source differs from frozen baseline blob")
    require((ROOT / "vendor/ninjars/mninjars.f90").read_bytes() == (ROOT / "HEPTools/ninjars/include/mninjars.f90").read_bytes(), "Installed NinjaRS Fortran source differs")
    require((ROOT / "vendor/ninjars/ninjars.h").read_bytes() == (ROOT / "HEPTools/ninjars/include/ninjars.h").read_bytes(), "Installed NinjaRS header differs")
    require((ROOT / "vendor/tirust/mtirust.f90").read_bytes() == (tir_install / "include/mtirust.f90").read_bytes(), "Installed TIRust Fortran source differs")
    locks = {name: tomllib.loads((ROOT / "vendor" / name / "Cargo.lock").read_text()) for name in ("ninjars", "tirust")}
    rug_pins = {name: next(p for p in lock["package"] if p["name"] == "rug") for name, lock in locks.items()}
    require(rug_pins["ninjars"]["version"] == rug_pins["tirust"]["version"] == "1.30.0", "Rug versions differ")
    require(rug_pins["ninjars"]["checksum"] == rug_pins["tirust"]["checksum"] == sha(RUG_CRATE), "Locked Rug crate checksum differs")
    with tarfile.open(RUG_CRATE, "r:gz") as archive:
        for name in ("src/float/big.rs", "src/float/traits.rs"):
            require(archive.extractfile("rug-1.30.0/" + name).read() == (RUG_SOURCE / name).read_bytes(), "Inspected Rug source differs from checksum-pinned crate")

    def lower_exp(path):
        text = path.read_text(); first = text.index("impl LowerExp for Float")
        last = text.index("\nimpl ", first + 5)
        return text[first:last]

    require(lower_exp(NRS_FLOAT) == lower_exp(TIR_FLOAT), "NinjaRS/TIRust Float formatter implementation differs")
    require('LowerExp::fmt(&self.0, f)' in lower_exp(NRS_FLOAT), "Explicit precision does not delegate to backend")
    for p in (NRS_FLOAT, TIR_FLOAT):
        require('Float as MultiPrecisionFloat' in (p.parent.parent / "backend/float.rs").read_text(), "Float backend is not Rug")
    require('pub use symbolica::domains::float::{Complex, DoubleFloat, Float, FloatLike, Real, SingleFloat};' in NRS_REEXPORT.read_text(), "NinjaRS Float type reexport differs")
    nrs_config = json.loads((ROOT / "HEPTools/ninjars/lib/ninjars-config.json").read_bytes())
    for name, key in (("Cargo.toml", "manifest_sha256"), ("Cargo.lock", "lock_sha256")):
        require(sha(ROOT / "vendor/ninjars" / name) == nrs_config["build_provenance"][key], "Installed NinjaRS manifest/lock provenance mismatch")

    trace_path = RUN / "b8-m4-o1/trace.txt"
    raw_trace = trace_path.read_bytes(); lines = raw_trace.splitlines(keepends=True)
    header = lines[0].split(); require(header[:2] == [b"QG1", b"1"] and header[4] == b"0", "Observer header/health")
    require(lines[-1].strip() == b"COMPLETE", "Trace not complete")
    records, invalid, token_counts = [], [], Counter()
    output_digits, sizes, statuses = Counter(), Counter(), Counter()
    call_records, current = [], None
    text_start, journal_start = len(lines[0]), 0
    for sequence, line in enumerate(lines[1:-1], 1):
        fields = line.split()
        seq, call, kind, part, group, so, rank, mode, size = map(int, fields[1:10])
        data = bytes.fromhex(fields[10].decode())
        require(seq == sequence and size == len(data), "Trace row sequence/length")
        record = {"seq": seq, "call": call, "kind": kind, "part": part, "group": group, "so": so, "rank": rank, "mode": mode,
                  "bytes": size, "decoded_journal_byte_start": journal_start, "trace_line_one_based": seq + 1,
                  "trace_text_byte_start": text_start, "data": data}
        records.append(record)
        if kind == 108:
            if part == 0:
                current = {}; call_records.append(current)
            require(current is not None and part not in current, "Native part duplication")
            current[part] = record
            if part == 9:
                statuses[struct.unpack("<i", data)[0]] += 1
            if part in (2, 3, 4, 5, 6, 10):
                require(size % 64 == 0, "ASCII record alignment")
                for offset in range(0, size, 64):
                    raw = data[offset:offset + 64]
                    stripped = raw.strip(b" \x00")
                    good = bool(PATTERN.fullmatch(stripped))
                    token_counts["output" if part == 10 else "input", "scientific" if good else stripped.decode("ascii")] += 1
                    if part == 10:
                        sizes[size] += int(offset == 0)
                        output_digits[len(re.sub(rb"[^0-9]", b"", stripped.split(b"e")[0]))] += 1
                    if not good:
                        invalid.append({"seq": seq, "call": call, "part": part, "group": group, "rank": rank,
                                        "slot_zero_based": offset // 64, "record_byte_offset_zero_based": offset,
                                        "decoded_journal_byte_offset_zero_based": journal_start + offset,
                                        "trace_line_one_based": seq + 1,
                                        "trace_text_hex_offset_zero_based": text_start + line.index(fields[10]) + 2 * offset,
                                        "token": stripped.decode("ascii")})
        journal_start += size; text_start += len(line)
    require(len(records) == int(header[2]) and journal_start == int(header[3]), "Trace totals differ")
    require(len(call_records) == 348 and statuses == {0: 348}, "Unexpected native call/status counts")
    require(len(invalid) == 630 and all(r["part"] == 10 and r["token"] == "0" for r in invalid), "Unexpected rejected token class")
    require(sizes == {512: 348}, "Unexpected output byte counts")
    for native in call_records:
        n = struct.unpack("<i", native[0]["data"])[0]
        rank = struct.unpack("<i", native[1]["data"])[0]
        count = (rank + 1) * (rank + 2) * (rank + 3) * (rank + 4) // 24
        expected = {0: 4, 1: 4, 2: 2 * 64 * count, 3: 4 * 64 * n, 4: 2 * 64 * n, 5: 64, 6: n * n * 64, 7: 4, 9: 4, 10: 8 * 64}
        require({part: r["bytes"] for part, r in native.items()} == expected, "NinjaRS ABI array lengths differ")
    first = invalid[0]
    first_call = next(c for c in call_records if c[10]["seq"] == first["seq"])
    first_data = first_call[10]["data"]
    original_zero = first_data[first["record_byte_offset_zero_based"]:first["record_byte_offset_zero_based"] + 64]
    require(original_zero == b"0" + b" " * 63, "First bad slot is not canonical space-padded zero")
    first["raw_slot_hex"] = original_zero.hex()
    first["raw_output_hex"] = first_data.hex()
    first["raw_output_tokens"] = [first_data[i:i + 64].decode("ascii").strip() for i in range(0, 512, 64)]
    first["preceding_status_record_seq"] = first_call[9]["seq"]
    first["native_status"] = struct.unpack("<i", first_call[9]["data"])[0]
    first["native_n"] = struct.unpack("<i", first_call[0]["data"])[0]
    first["native_rank"] = struct.unpack("<i", first_call[1]["data"])[0]
    result = json.loads((RUN / "result.json").read_bytes())
    require(result["passed"] is False and result["workers_completed"] == 9 and result["failure"] == {"type": "ValueError", "message": "malformed decimal transport"}, "Closed run4 failure differs")
    attempted = []
    for folder in sorted(RUN.glob("b*-m*-o*")):
        worker = json.loads((folder / "worker.json").read_bytes())
        attempted.append({"worker": folder.name, "returncode": worker["returncode"], "parsed_result_retained": (folder / "parsed.json").exists()})
    require(len(attempted) == 10 and all(w["returncode"] == 0 for w in attempted), "Attempted worker inventory differs")
    after = {str(p): sha(p) for p in paths}
    require(before == after, "Input bytes changed during static diagnosis")
    diagnosis = {"schema": "qcd-n3-run4-static-output-transport-diagnosis-1", "scope": "Static diagnosis only; no parser changes, new source or native execution.",
                 "native_or_compiler_or_loader_calls": 0, "all_input_bytes_unchanged": True, "input_sha256": before,
                 "baseline_commit": "9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f", "baseline_NinjaRS_git_blob_ids": baseline_blobs,
                 "closed_failure": result["failure"], "attempted_worker_inventory": attempted, "workers_native_attempted": 10,
                 "workers_frozen_parser_completed": 9, "workers_unexecuted": 6,
                 "native_NinjaRS_call_count": len(call_records), "native_status_counts": dict(statuses),
                 "all_NinjaRS_ABI_counts_lengths_and_64byte_strides_match": True,
                 "observer_health": int(header[4]), "first_failing_slot": first,
                 "invalid_scientific_token_count": len(invalid), "all_invalid_tokens_are_exact_positive_zero_outputs": True,
                 "invalid_locations": invalid, "token_counts": [{"direction": k[0], "class": k[1], "count": v} for k, v in sorted(token_counts.items())],
                 "actual_nonzero_output_significant_digit_counts": dict(output_digits),
                 "status_checks": ["Observer C calls original, records status, and copies output only when status==0.",
                                   "Frozen parser reads status and establishes success and expected output shape before decoding part10.",
                                   "All348actualNinjaRSMPcalls returned status0 and supplied512outputbytes."],
                 "formatter_type_chain": {"NinjaRS_revision": "41a4c00e7635e3ab188fc26bacf0aae6941e8082", "NinjaRS_symbolica_revision": "9cdc2949cdb153478c1eca0d40690dc0abfc616e",
                                          "TIRust_symbolica_revision": "dfe6bd97b974b1028dc63071ea165136cd2b0ef5", "explicit_LowerExp_implementation_identical": True,
                                          "Rug_version": "1.30.0", "Rug_crate_checksum": sha(RUG_CRATE), "Rug_archive_sources_match_inspected_files": True},
                 "diagnosis": "The diagnostic's exponent-required regex rejects canonical zero output emitted by the pinned production Float/Rug formatter. Rug emits0or-0 before considering exponent formatting. The observedfailure is a decimal-grammar mismatch, with no observed ABI/stride/count/status mismatch.",
                 "precision_semantics": ["Fortran inputs use ES64.39E4 (40 significant decimal digits) from actual binary128 values.",
                                         "Both Rust bridges parse each input at113bits, reject nonfinite values and widen that represented value to256bits while preserving signedzero.",
                                         "Both output bridges use format!(\"{value:.39e}\") and64byte spacepadding. The pinned numerica Float delegates explicitprecision toRug; its zero branch emits0or-0 without an exponent.",
                                         "Both Fortran output consumers use list-directed READ into binary128 and therefore accept plainzero.",
                                         "Actual nonzero NinjaRS outputs contain39significant digits under Rug's precision semantics, although the ABI prose says40. No binary64 conversion was used in the codec.",
                                         "The offline parser rounds exact decimal rationals at113bits for qualified normal-range values. Fraction itself has no signedzero; raw records must remain the sign witness. This does not establish edge-range qualification or universal correct rounding of arbitrary256bit results."],
                 "TIRust_scope": "Pinned TIRust source proves the same successful-output formatter/Fortran-reader contract. No backend9nativeworker was executed in run4; no backend9numerical result or future parser pass is claimed.",
                 "source_excerpts": [span(SOURCE / "analyze.py", 81, 91), span(SOURCE / "analyze.py", 151, 175), span(SOURCE / "observer.c", 110, 129),
                                     span(ROOT / "vendor/ninjars/src/lib.rs", 573, 586), span(ROOT / "vendor/ninjars/src/lib.rs", 700, 721),
                                     span(ROOT / "vendor/ninjars/mninjars.f90", 139, 160), span(ROOT / "vendor/ninjars/mninjars.f90", 197, 215),
                                     span(ROOT / "vendor/tirust/src/lib.rs", 498, 514), span(ROOT / "vendor/tirust/src/lib.rs", 550, 577),
                                     span(ROOT / "vendor/tirust/mtirust.f90", 266, 291),
                                     span(NRS_FLOAT, 132, 145), span(TIR_FLOAT, 132, 145), span(RUG_SOURCE / "src/float/traits.rs", 89, 98),
                                     span(RUG_SOURCE / "src/float/big.rs", 12625, 12634)],
                 "scientific_acceptance": False, "frozen_run4_remains_failed": True}
    (HERE / "diagnosis.json").write_text(json.dumps(diagnosis, indent=2, sort_keys=True) + "\n")
    packet = {"schema": "qcd-n3-run4-static-transport-index-1", "scope": "Exact closed-byte and pinned production formatter diagnosis; no gate changes or native calls.",
              "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(HERE.iterdir()) if p.is_file() and p.name != "index.json"}}
    (HERE / "index.json").write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"index_sha256": sha(HERE / "index.json"), "payload_count": len(packet["files"]),
                      "native_calls_status0": 348, "invalid_tokens_all_exact_zero": 630,
                      "first_bad_seq": first["seq"], "first_bad_slot": first["slot_zero_based"],
                      "first_bad_decoded_journal_byte_offset": first["decoded_journal_byte_offset_zero_based"],
                      "first_bad_trace_text_hex_offset": first["trace_text_hex_offset_zero_based"]}, indent=2))


if __name__ == "__main__":
    main()
