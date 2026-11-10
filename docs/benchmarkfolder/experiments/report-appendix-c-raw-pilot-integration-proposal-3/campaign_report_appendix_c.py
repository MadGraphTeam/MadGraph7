"""Optional closed raw-DP pilot figures; no discovery or acceptance mutation.

Only the exact root-admitted closed data/figure/plotter closure is accepted.
Native request completion is distinct from diagnostic availability or accuracy.
"""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
import tempfile


ADMITTED = {
    (
        'stability_appendix_c/raw-dp-pilot-5-figures-1/index.json',
        'b131e6294fb3c6900e6776339948928ae71518092cc99de59a8bb0d39fdabcfd',
    ): {
        "data": {
            "index": "stability_appendix_c/raw-dp-pilot-5-1/index.json",
            "sha256": "ebd8f1faf5228f205b1038a490329037066988dc03e7e774210d648b38dbbb6e"
        },
        "data_verifier_sha256": "efac5f8debda3513c935e1f2cd24de82ad1e8a6ff8561452fbff4ba3220186e0",
        "figure_verifier_sha256": "9f3141c7cc1e0b92daf891a19c624ba23b2bd654643af51bf7b302f18f26e166",
        "figures": {
            "index": "stability_appendix_c/raw-dp-pilot-5-figures-1/index.json",
            "sha256": "b131e6294fb3c6900e6776339948928ae71518092cc99de59a8bb0d39fdabcfd"
        },
        "plotter": {
            "index": "stability_appendix_c/plotter-source-6/index.json",
            "sha256": "a0521f17a658e04f8dc320509e733de8fd0e93ac4d02481d908373291ac706ae"
        }
    }
}
CASES = ("qcd_n2_gg_ttx", "qcd_n3_uux_zgg", "ew_n2_epem_mupmum", "qcd_n4_uux_zzgg")
BACKENDS = (1, 6, 7, 8, 9)
GROUP_FAILURES = {"coverage_missing", "active_group_failure", "nonfinite_coefficients",
                  "nonfinite_results", "floating_point_failure_latch"}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def safe(name):
    require(isinstance(name, str) and name and "\\" not in name and "\0" not in name,
            "Invalid Appendix C evidence path")
    path = PurePosixPath(name)
    require(path.parts and not path.is_absolute() and ".." not in path.parts and str(path) == name,
            "Unsafe Appendix C evidence path")
    return name


def require_admitted(pin):
    require(isinstance(pin, dict) and set(pin) == {"index", "sha256"}
            and digest(pin.get("sha256")), "Appendix C needs an exact closed figure index pin")
    safe(pin["index"])
    contract = ADMITTED.get((pin["index"], pin["sha256"]))
    require(isinstance(contract, dict) and set(contract) == {
        "data", "figures", "plotter", "data_verifier_sha256", "figure_verifier_sha256"}
        and contract["figures"] == pin, "Appendix C capsule is not explicitly admitted")
    for role in ("data", "figures", "plotter"):
        value = contract[role]
        require(isinstance(value, dict) and set(value) == {"index", "sha256"}
                and digest(value.get("sha256")), "Appendix C dependency needs an exact pin")
        safe(value["index"])
    require(all(digest(contract[key]) for key in ("data_verifier_sha256", "figure_verifier_sha256")),
            "Appendix C verifier identity is absent")
    return contract


def decode(raw):
    def reject(value):
        raise ValueError("Nonfinite Appendix C summary JSON: " + value)
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "Duplicate Appendix C JSON key")
            value[key] = item
        return value
    def finite(value):
        number = float(value)
        require(math.isfinite(number), "Nonfinite Appendix C summary number")
        return number
    return json.loads(raw, parse_constant=reject, parse_float=finite, object_pairs_hook=pairs)


def bind(evidence, pin, root):
    """Copy one authenticated physical payload at a time into an isolated root."""
    raw = evidence.add(pin["index"], pin["sha256"])
    index = decode(raw)
    require(isinstance(index.get("files"), dict) and index["files"], "Empty Appendix C capsule")
    prefix = str(PurePosixPath(pin["index"]).parent)
    index_name = PurePosixPath(pin["index"]).name
    root.mkdir()
    (root / index_name).write_bytes(raw)
    for name, item in index["files"].items():
        safe(name)
        require(name != index_name and isinstance(item, dict) and digest(item.get("sha256"))
                and type(item.get("bytes")) is int and item["bytes"] >= 0,
                "Invalid Appendix C physical inventory")
        data = evidence.add(prefix + "/" + name, item["sha256"])
        require(len(data) == item["bytes"], "Appendix C payload byte count differs")
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    return {"index": index, "prefix": prefix}


def verifier(packet, root, wanted, name, filename="verify.py"):
    require(filename in packet["index"]["files"], "Appendix C verifier source is unindexed")
    source = (root / filename).read_bytes()
    require(hashlib.sha256(source).hexdigest() == wanted, "Appendix C verifier source differs")
    namespace = {"__name__": name, "__file__": str(root / filename)}
    exec(compile(source, namespace["__file__"], "exec", optimize=0), namespace)
    require(callable(namespace.get("verify")), "Appendix C verifier API is absent")
    return namespace["verify"]


def counts(value, name, required=None):
    require(isinstance(value, dict) and all(isinstance(key, str) and type(number) is int and number >= 0
            for key, number in value.items()), "Invalid Appendix C " + name)
    if required is not None:
        require(set(value) == required, "Appendix C failure-count categories differ")
    return value


def finite_json(value):
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Appendix C retained summary or verification is not finite JSON") from error


def normalize(summary, figures):
    finite_json(summary)
    require(summary.get("schema") == "madloop-raw-stability-summary-v1", "Appendix C summary schema differs")
    rows = summary.get("rows")
    wanted = [(case, backend) for case in CASES for backend in BACKENDS]
    require(isinstance(rows, list) and len(rows) == 20
            and all(isinstance(row, dict) for row in rows)
            and [(row.get("case"), row.get("backend")) for row in rows] == wanted,
            "Appendix C must retain all twenty case/backend results in fixed order")
    normalized = []
    for row in rows:
        require(type(row["backend"]) is int and isinstance(row.get("result_state"), str) and row["result_state"]
                and type(row.get("native_started")) is bool
                and all(name in row for name in ("error", "diagnostic_error", "readonly_guard_error")),
                "Appendix C result state or failure evidence is absent")
        fields = ("intended_points", "available_points", "unavailable_points", "zero_points", "infinite_points")
        require(all(type(row.get(name)) is int and row[name] >= 0 for name in fields)
                and row["intended_points"] == 100
                and row["available_points"] + row["unavailable_points"] == 100
                and row["zero_points"] + row["infinite_points"] <= row["available_points"],
                "Appendix C point availability or zero/infinite counts differ")
        if not row["native_started"]:
            require(row["available_points"] == 0, "An unstarted Appendix C request has measured points")
        for name in ("return_code_counts", "bootstrap_return_code_counts", "primary_outcome_counts"):
            counts(row.get(name), name)
        require(sum(row["primary_outcome_counts"].values()) == 300,
                "Appendix C must preserve all 300 intended primary-call outcomes")
        if not row["native_started"]:
            require(row["primary_outcome_counts"] == {"unexecuted": 300},
                    "An unstarted Appendix C request must retain 300 unexecuted outcomes")
        counts(row.get("group_failure_call_counts"), "group failure counts", GROUP_FAILURES)
        tail = row.get("tail_at_1e3")
        require(isinstance(tail, dict) and type(tail.get("n")) is int and tail["n"] == 100
                and type(tail.get("known_exceedances")) is int
                and row["infinite_points"] <= tail["known_exceedances"] <= row["available_points"] - row["zero_points"]
                and type(tail.get("unavailable")) is int and tail["unavailable"] == row["unavailable_points"],
                "Appendix C tail denominator or unavailable accounting differs")
        lower = tail["known_exceedances"] / 100
        upper = (tail["known_exceedances"] + tail["unavailable"]) / 100
        require(all(type(tail.get(name)) in (int, float) and math.isfinite(tail[name]) for name in ("lower", "upper"))
                and tail.get("lower") == lower and tail.get("upper") == upper
                and (tail.get("fraction") is None if tail["unavailable"] else
                     type(tail.get("fraction")) in (int, float) and math.isfinite(tail["fraction"]))
                and tail.get("fraction") == (None if tail["unavailable"] else lower),
                "Appendix C unknown outcomes were relabeled as measured failures")
        normalized.append({**row,
            "return_code_counts_text": json.dumps(row["return_code_counts"], sort_keys=True),
            "bootstrap_return_code_counts_text": json.dumps(row["bootstrap_return_code_counts"], sort_keys=True),
            "primary_outcome_counts_text": json.dumps(row["primary_outcome_counts"], sort_keys=True),
            "group_failure_counts_text": json.dumps(row["group_failure_call_counts"], sort_keys=True)})
    return [{"id": case, "process": case, **figures[case], "lanes": [row for row in normalized if row["case"] == case]}
            for case in CASES]


def collect(evidence, pin):
    contract = require_admitted(pin)  # No reads for unadmitted or active inputs.
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        with tempfile.TemporaryDirectory(prefix="tirust-report-appendix-c-") as temporary:
            roots = {role: Path(temporary) / role for role in ("data", "figures", "plotter")}
            packets = {role: bind(evidence, contract[role], roots[role]) for role in roots}
            for role, schema in (("data", "madloop-raw-stability-capsule-v1"),
                                 ("figures", "madloop-raw-stability-figures-v1"),
                                 ("plotter", "madloop-stability-plotter-source-v1")):
                require(packets[role]["index"].get("schema") == schema, "Appendix C indexed role schema differs")
            paths = {}
            for case in CASES:
                names = {"figure_path": "figures/" + case + ".png",
                         "export_pdf_path": "figures/" + case + ".pdf",
                         "figure_data_path": "figures/" + case + ".json"}
                require(all(name in packets["figures"]["index"]["files"] for name in names.values()),
                        "Appendix C figure packet does not contain all four PNG/PDF/JSON sets")
                paths[case] = {key: packets["figures"]["prefix"] + "/" + name for key, name in names.items()}
            data = verifier(packets["data"], roots["data"], contract["data_verifier_sha256"],
                            "retained_appendix_c_data")(roots["data"])
            figures = verifier(packets["figures"], roots["figures"], contract["figure_verifier_sha256"],
                               "retained_appendix_c_figures", "figures.py")(
                                   roots["figures"], data_root=roots["data"], plotter_root=roots["plotter"])
    finally:
        sys.dont_write_bytecode = previous
    for role, proof in (("data", data), ("figures", figures)):
        finite_json(proof)
        require(isinstance(proof, dict) and proof.get("verified") is True
                and proof.get("index_sha256") == contract[role]["sha256"],
                "Appendix C portable verification differs")
    require(isinstance(data.get("summary"), dict)
            and figures.get("cases") == list(CASES) and type(figures.get("artifacts")) is int
            and figures["artifacts"] == 12 and isinstance(figures.get("dependencies"), dict)
            and figures["dependencies"].get("data_index_sha256") == contract["data"]["sha256"]
            and figures["dependencies"].get("plotter_index_sha256") == contract["plotter"]["sha256"],
            "Appendix C figure case coverage or explicit dependency proof differs")
    cases = normalize(data["summary"], paths)
    return {"status": "closed diagnostic pilot", "selection": pin, "dependencies": contract,
            "scope": "100 cut-qualified points per case; host forced-DP/card-selected backend; three-probe internal consistency",
            "cases": cases, "retained_data_summary": data["summary"], "retained_figure_verification": figures,
            "performance_eligible": False, "independent_truth_qualified": False,
            "observed_actual_backend": False, "observed_per_result_internal_precision": False,
            "main_campaign_modified": False}
