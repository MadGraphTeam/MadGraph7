"""Pure validation of the proposed private replay; never reads failed tensors."""
import json
from pathlib import Path

MAX_DIAGNOSTIC_BYTES = 65536


def require(value, message):
    if not value:
        raise ValueError(message)


def input_record(raw):
    """Parse captured words as integers; perform no floating-point arithmetic."""
    require(type(raw) is bytes and len(raw) <= 32768, "Input byte bound")
    text = raw.decode("ascii")
    require(text.endswith("\n") and "\r" not in text and "\t" not in text, "Input line format")
    lines = text.splitlines()
    require(lines[0] == "MLTCAP_INPUT_V1", "Input format version")
    fields = lines[1].split(" ")
    require(len(fields) == 15 and all(x.isdecimal() and x.isascii() for x in fields), "Input header")
    h = list(map(int, fields))
    case, phase, point, method, group, n, rank, plen, slen, mlen, olen, poles, finite, check, cache = h
    require((case, phase, point, n, rank, plen, slen, mlen, olen, poles, finite, cache) ==
            (1, 1, 25, 5, 4, 20, 25, 5, 280, 0, 1, 0), "Only frozen Zgg requests admitted")
    require(method in (1, 2, 3) and group in (59, 61) and check == int(method == 2), "Frozen direction/group")
    words = lines[2:]
    require(len(words) == 1 + plen + slen + 2 * mlen, "Exact operand count")
    require(all(len(x) == 16 and all(c in "0123456789abcdefABCDEF" for c in x) for x in words), "Binary64 word format")
    return {"input_key": {"case_id": case, "phase": phase, "point": point, "method": method, "group": group},
            "header": h, "mu2_word": words[0].lower(), "shift_words": words[1:1+plen],
            "invariant_words": words[1+plen:1+plen+slen], "mass_words": words[1+plen+slen:]}


def diagnostic_text(row):
    require(row.get("private_diagnostic_encoding") == "latin1-bytes", "Diagnostic byte encoding")
    require(row.get("private_diagnostic_copy_truncated") is False, "Diagnostic copy truncated")
    data = row["private_diagnostic"].encode("latin1")
    require(1 <= len(data) + 1 <= MAX_DIAGNOSTIC_BYTES and row.get("private_diagnostic_required") == len(data)+1,
            "Diagnostic byte count")
    text = data.decode("utf-8")
    lines = text.splitlines()
    require(lines and lines[0] == "private_dp_diagnostics_v1" and lines[-1] == "truncated=false", "Incomplete snapshot")
    require("reduce_batch_executed=true" in lines and "reduce_batch=Ok" in lines and
            "workspace_metadata_fresh=true" in lines and "precision_bits=53" in lines, "No fresh full-DP snapshot")
    fields = {}
    for prefix in ("embedder_options=", "lane_statuses=", "fallback_used=", "analytic_used=",
                   "last_lane_routes=", "engine_execution=", "last_diagnostics="):
        matched = [line[len(prefix):] for line in lines if line.startswith(prefix)]
        require(len(matched) == 1, "Missing/duplicate public metadata: " + prefix)
        fields[prefix[:-1]] = matched[0]
    # This is pinned Rust Debug text, not a new production numerical ABI.
    for value in ("method_choice: true", "partial_rescue: false", "precision_tiers: false",
                  "gate: None", "method_choice_gate: None", "master_policy: Default"):
        require(value in fields["embedder_options"], "Embedded numerical options changed")
    return fields


def replay_summary(text, inputs):
    require(len(inputs) == 6, "Exactly six inputs required")
    expected_keys = {(1, 1, 25, method, group) for method in (1, 2, 3) for group in (59, 61)}
    actual_keys = [tuple(x["input_key"][k] for k in ("case_id", "phase", "point", "method", "group")) for x in inputs]
    require(len(set(actual_keys)) == 6 and set(actual_keys) == expected_keys, "Exact frozen six-input keys")
    rows = [json.loads(line) for line in text.splitlines()]
    require(len(rows) == 30, "Six create/register/evaluate/evaluate/destroy sequences required")
    evaluations = []
    for i, expected in enumerate(inputs):
        group = rows[i*5:(i+1)*5]
        require([row.get("record_type") for row in group] == ["create", "register", "evaluate", "evaluate", "destroy"],
                "Protocol stage order")
        require([row.get("context_present") for row in group] == [True, True, True, True, False],
                "Context lifecycle changed")
        require(group[-1].get("destroy_succeeded") is True, "Context destruction not confirmed")
        require([row.get("evaluation_count") for row in group] == [2*i, 2*i, 2*i+1, 2*i+2, 2*i+2],
                "Stage evaluation counts changed")
        require([row.get("repetition") for row in group] == [1, 1, 1, 2, 2], "Stage repetition labels changed")
        for row in group:
            require(row.get("input_key") == expected["input_key"], "Input key changed")
            require(Path(row.get("input_path", "")).name == expected["basename"], "Input path changed")
            require(row.get("error_encoding") == "latin1-bytes" and row.get("error_truncated") is False,
                    "Boundary error incomplete")
            if row["record_type"] != "evaluate":
                require(row.get("status") == 0 and row.get("error") == "" and row.get("error_required") == 1,
                        "Setup/destruction failed")
                require("output_bits" not in row and "private_diagnostic" not in row, "Unexpected setup payload")
                continue
            # Match the already observed EB3 refusal, rather than accepting a
            # changed numerical result or interpreting a partial output array.
            require(row.get("status") == expected["expected_status"] == 1, "Boundary status changed")
            require("output_bits" not in row, "Failed output must never be read or serialized")
            require(row.get("error") == expected["expected_error"], "Boundary LAST_ERROR changed")
            require(row.get("error_required") == len(row["error"].encode("latin1"))+1, "Boundary error byte count")
            metadata = diagnostic_text(row)
            evaluations.append({"input_key": expected["input_key"], "repetition": row["repetition"],
                                "status": row["status"], "boundary_error": row["error"], "public_metadata": metadata})
        require([row["repetition"] for row in group[2:4]] == [1, 2], "Fresh/repeat labels changed")
    require([row["evaluation_count"] for row in rows if row["record_type"] == "evaluate"] == list(range(1, 13)),
            "Twelve-call ceiling/order")
    return {"protocol_complete": True, "inputs": 6, "evaluations": 12, "failed_tensor_outputs_read": 0,
            "all_boundary_statuses_match_eb3": True, "public_api_metadata_only": True,
            "inner_variant_or_operand_observer": False, "numerical_acceptance_claimed": False,
            "performance_acceptance_claimed": False, "evaluations_metadata": evaluations}
