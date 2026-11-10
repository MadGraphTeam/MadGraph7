"""Pure, exact-source bridge transformation. No CLI, files or native execution.

Copy observer.rs beside the transformed lib.rs, and link observer_flags.c only
in an isolated diagnostic build. This function neither changes a source tree
nor changes the Cargo/numerical dependency graph.
"""
import hashlib

BRIDGE_SHA256 = "455529b929b725e8fe2d7337ab4a41fa3de9dee1b3a953afe6dc032cf99e5a4c"


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Observer anchor is absent or ambiguous: " + old[:80])
    return source.replace(old, new, 1)


def transform(raw):
    if hashlib.sha256(raw).hexdigest() != BRIDGE_SHA256:
        raise ValueError("Observer only supports the pinned current bridge")
    source = raw.decode("utf-8")
    source = once(source, "use iregi_rust::{\n    Convention,", "mod observer;\nuse iregi_rust::{\n    Convention,")
    start = source.index("    fn dp(&mut self,")
    end = source.index("    fn dp_checked(", start)
    dp = source[start:end]
    old = "        let result = workspace.reduce_batch(&batch).map_err(|e| e.to_string())?;"
    new = ("        let observer_result = workspace.reduce_batch(&batch);\n"
           "        observer::dp_workspace(workspace, observer_result.is_ok());\n"
           "        let result = observer_result.map_err(|e| e.to_string())?;")
    source = source[:start] + once(dp, old, new) + source[end:]
    source = once(source, "                self.qp_cache.hits += 1;\n                return Ok((SUCCESS, values.to_vec()));",
                         "                self.qp_cache.hits += 1;\n                observer::qp_cache_hit();\n                return Ok((SUCCESS, values.to_vec()));")
    for function, precision, following in (
        ("ml_tirust_evaluate_dp", 53, "/// Evaluate genuine binary128"),
        ("ml_tirust_evaluate_mp", 113, "/// # Safety\n/// NULL/zero queries required size"),
    ):
        start = source.index("pub unsafe extern \"C\" fn " + function + "(")
        end = source.index(following, start)
        body = source[start:end]
        body = once(body, ") -> i32 {\n    guard(|| {",
                    ") -> i32 {\n    observer::native_begin(handle as usize, " + str(precision) +
                    ", n, rank, poles, finite, check, cache);\n    let observer_status = guard(|| {")
        body = once(body, "        Ok(status)\n    })\n}",
                    "        Ok(status)\n    });\n    observer::native_end(observer_status);\n    observer_status\n}")
        source = source[:start] + body + source[end:]
    rendered = source.encode("utf-8")
    if restore(rendered) != raw:
        raise ValueError("Observer erasure does not reproduce original source")
    return rendered


def restore(raw):
    """Erase only the declared observation insertions, requiring each once."""
    source = raw.decode("utf-8")
    source = once(source, "mod observer;\nuse iregi_rust::{\n    Convention,", "use iregi_rust::{\n    Convention,")
    source = once(source,
        "        let observer_result = workspace.reduce_batch(&batch);\n"
        "        observer::dp_workspace(workspace, observer_result.is_ok());\n"
        "        let result = observer_result.map_err(|e| e.to_string())?;",
        "        let result = workspace.reduce_batch(&batch).map_err(|e| e.to_string())?;")
    source = once(source, "                observer::qp_cache_hit();\n", "")
    for function, precision, following in (
        ("ml_tirust_evaluate_dp", 53, "/// Evaluate genuine binary128"),
        ("ml_tirust_evaluate_mp", 113, "/// # Safety\n/// NULL/zero queries required size"),
    ):
        start = source.index("pub unsafe extern \"C\" fn " + function + "(")
        end = source.index(following, start)
        body = source[start:end]
        body = once(body,
                    ") -> i32 {\n    observer::native_begin(handle as usize, " + str(precision) +
                    ", n, rank, poles, finite, check, cache);\n    let observer_status = guard(|| {",
                    ") -> i32 {\n    guard(|| {")
        body = once(body, "        Ok(status)\n    });\n    observer::native_end(observer_status);\n    observer_status\n}",
                    "        Ok(status)\n    })\n}")
        source = source[:start] + body + source[end:]
    restored = source.encode("utf-8")
    if hashlib.sha256(restored).hexdigest() != BRIDGE_SHA256:
        raise ValueError("Erased source differs from pinned original bridge")
    return restored
