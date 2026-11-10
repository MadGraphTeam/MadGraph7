"""Derive an isolated diagnostic-only bridge from exact installed EB3 bytes.

No compiler or native invocation. Production, installed and upstream sources
are never written. prepare(basebytes) returns the private Rust source bytes.
"""
import difflib
import hashlib

BASE_SHA256 = "455529b929b725e8fe2d7337ab4a41fa3de9dee1b3a953afe6dc032cf99e5a4c"
GETTER = "ml_tirust_private_dp_diagnostics_v1"
MAX_BYTES = 65536
ANCHOR = "\nstruct State {\n"
REDUCE = "        let result = workspace.reduce_batch(&batch).map_err(|e| e.to_string())?;\n"
SNAPSHOT = """        let private_reduced = workspace.reduce_batch(&batch);
        private_dp_diagnostic_snapshot(workspace, &private_reduced);
        let result = private_reduced.map_err(|e| e.to_string())?;
"""
ENTRY = """    PRIVATE_DP_DIAGNOSTICS_V1.with(|record| record.borrow_mut().clear());
"""
DP_SIGNATURE = "pub unsafe extern \"C\" fn ml_tirust_evaluate_dp(\n"
DP_ENTRY_END = ") -> i32 {\n    guard(|| {\n"
DP_CORE = "    fn dp(&mut self, key: Key, batch: KinematicsBatch<f64>, mode: Mode) -> Result<(i32, Vec<C64>)> {\n"
DP_CORE_END = "    fn dp_checked(\n"

PRIVATE_BLOCK = r'''
// BEGIN PRIVATE DP DIAGNOSTICS V1 (isolated experiment; no numerical policy).
thread_local! {
    static PRIVATE_DP_DIAGNOSTICS_V1: RefCell<String> = const { RefCell::new(String::new()) };
}
const PRIVATE_DP_DIAGNOSTIC_MAX: usize = 65536;
struct PrivateDpDiagnosticText {
    text: String,
    truncated: bool,
}
impl std::fmt::Write for PrivateDpDiagnosticText {
    fn write_str(&mut self, value: &str) -> std::fmt::Result {
        // Reserve room for the final truncation marker. No partial UTF-8.
        let room = (PRIVATE_DP_DIAGNOSTIC_MAX - 64).saturating_sub(self.text.len());
        let mut keep = room.min(value.len());
        while !value.is_char_boundary(keep) { keep -= 1; }
        self.text.push_str(&value[..keep]);
        self.truncated |= keep != value.len();
        Ok(())
    }
}
fn private_dp_diagnostic_snapshot(
    workspace: &TensorWorkspace<f64>,
    reduced: &std::result::Result<iregi_rust::BatchResult<f64>, iregi_rust::ReductionError>,
) {
    use std::fmt::Write as _;
    let mut record = PrivateDpDiagnosticText { text: String::new(), truncated: false };
    let _ = writeln!(record, "private_dp_diagnostics_v1");
    let _ = writeln!(record, "reduce_batch_executed=true");
    let _ = writeln!(record, "precision_bits={}", workspace.binary_precision());
    let _ = writeln!(record, "embedder_options={:?}", workspace.embedder_options());
    match reduced {
        Ok(result) => {
            let _ = writeln!(record, "reduce_batch=Ok");
            let _ = writeln!(record, "workspace_metadata_fresh=true");
            let _ = writeln!(record, "lane_statuses={:?}", result.statuses);
            let _ = writeln!(record, "fallback_used={:?}", result.fallback_used);
            let _ = writeln!(record, "analytic_used={:?}", result.analytic_used);
            // No result values, lane() or best-effort tensor arrays are read.
        }
        Err(error) => {
            let _ = writeln!(record, "reduce_batch=Err({error:?})");
            // The upstream engine may return Err before resetting its last
            // metadata. Keep it labelled, never identify it as this call.
            let _ = writeln!(record, "workspace_metadata_fresh=unproven_partial_or_previous");
        }
    }
    let _ = writeln!(record, "last_lane_routes={:?}", workspace.last_lane_routes());
    let _ = writeln!(record, "engine_execution={:?}", workspace.engine_execution());
    let _ = writeln!(record, "last_diagnostics={:?}", workspace.last_diagnostics());
    record.text.push_str(if record.truncated { "truncated=true\n" } else { "truncated=false\n" });
    PRIVATE_DP_DIAGNOSTICS_V1.with(|target| *target.borrow_mut() = record.text);
}
/// Private read-only TLS copy. NULL/zero queries required bytes including NUL.
/// Empty means the DP call did not reach the embedded reduce_batch snapshot.
/// # Safety
/// Non-NULL out has capacity writable bytes. Returned pointers are never borrowed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ml_tirust_private_dp_diagnostics_v1(out: *mut u8, capacity: usize) -> usize {
    PRIVATE_DP_DIAGNOSTICS_V1.with(|record| {
        let record = record.borrow();
        if !out.is_null() && capacity > 0 {
            let n = record.len().min(capacity - 1);
            unsafe {
                write(out, &record.as_bytes()[..n]);
                out.add(n).write(0);
            }
        }
        record.len() + 1
    })
}
// END PRIVATE DP DIAGNOSTICS V1.
'''


def _entry_offset(text):
    if text.count(DP_SIGNATURE) != 1:
        raise ValueError("Ambiguous DP entry")
    start = text.index(DP_SIGNATURE)
    end = text.index(DP_ENTRY_END, start)
    return end + len(") -> i32 {\n")


def prepare(basebytes):
    if type(basebytes) is not bytes or hashlib.sha256(basebytes).hexdigest() != BASE_SHA256:
        raise ValueError("Expected exact unchanged installed EB3 bridge source")
    text = basebytes.decode("utf-8")
    if text.count(ANCHOR) != 1 or text.count(DP_CORE) != 1 or GETTER in text:
        raise ValueError("Unexpected bridge structure")
    text = text.replace(ANCHOR, PRIVATE_BLOCK + ANCHOR, 1)
    start = text.index(DP_CORE)
    end = text.index(DP_CORE_END, start)
    section = text[start:end]
    if section.count(REDUCE) != 1:
        raise ValueError("Expected exactly one full-DP reduction")
    text = text[:start] + section.replace(REDUCE, SNAPSHOT, 1) + text[end:]
    offset = _entry_offset(text)
    text = text[:offset] + ENTRY + text[offset:]
    answer = text.encode("utf-8")
    if restore(answer) != basebytes:
        raise ValueError("Diagnostic additions do not reverse exactly")
    return answer


def restore(candidatebytes):
    text = candidatebytes.decode("utf-8")
    if text.count(PRIVATE_BLOCK) != 1 or text.count(SNAPSHOT) != 1 or text.count(ENTRY) != 1:
        raise ValueError("Missing or ambiguous diagnostic additions")
    text = text.replace(PRIVATE_BLOCK, "", 1).replace(SNAPSHOT, REDUCE, 1).replace(ENTRY, "", 1)
    return text.encode("utf-8")


def patch(basebytes):
    return "".join(difflib.unified_diff(basebytes.decode().splitlines(True),
        prepare(basebytes).decode().splitlines(True), fromfile="a/src/lib.rs", tofile="b/src/lib.rs"))
