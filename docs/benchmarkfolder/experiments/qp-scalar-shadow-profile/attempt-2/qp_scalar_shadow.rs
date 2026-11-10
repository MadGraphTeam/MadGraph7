//! Private diagnostic only. Observe every kernel; never return cached values.
//! This file is copied into an isolated iregi-rust source tree by prepare.py.
use std::{any::{Any,TypeId}, cell::RefCell, sync::OnceLock, time::Instant};
use iregi_masters::NativeFloat;
use symbolica::domains::float::{Complex, Float};

// The core keeps forbid(unsafe_code). Only the bridge calls MPFR's C ABI.
static MPFR_READER: OnceLock<fn() -> u32> = OnceLock::new();
pub fn install_mpfr_reader(reader: fn() -> u32) -> bool {
    let installed = MPFR_READER.get_or_init(|| reader);
    std::ptr::fn_addr_eq(*installed, reader)
}
pub fn mpfr_reader_available() -> bool { MPFR_READER.get().is_some() }
// Explicit sentinel outside MPFR_FLAGS_ALL=63. Comparisons must also require
// mpfr_reader_available=true; two missing readers can never establish parity.
pub fn mpfr_flags() -> u32 { MPFR_READER.get().map(|reader| reader()).unwrap_or(u32::MAX) }

const MAX_BYTES: usize = 8 * 1024 * 1024;
const MAX_SCALARS: u64 = 4096;
const MAX_EVENTS: usize = 16384;
const NAMESPACE: &str = "TIRust854b79dc_oneloop_bd703d09_Symbolica_dfe6bd97_MP256_input113_native_pinnedconfig_v1";

#[derive(Default)]
struct Journal {
    enabled: bool,
    parent: u64,
    next_parent: u64,
    point: i32,
    group: i32,
    synthetic: bool,
    rows: Vec<String>,
    bytes: usize,
    scalar_requests: u64,
    scalar_completed: u64,
    scalar_errors: u64,
    nonfinite_outputs: u64,
    kernel_ns: u128,
    key_ns: u128,
    dropped_events: u64,
    key_flag_changes: u64,
    output_flag_changes: u64,
}
impl Journal {
    fn emit(&mut self, row: String) {
        if self.rows.len() >= MAX_EVENTS || row.len() > MAX_BYTES.saturating_sub(self.bytes) {
            self.dropped_events += 1;
        } else {
            self.bytes += row.len();
            self.rows.push(row);
        }
    }
}
thread_local! { static JOURNAL: RefCell<Journal> = RefCell::new(Journal::default()); }

pub fn control(enabled: bool) {
    JOURNAL.with(|j| *j.borrow_mut() = Journal { enabled, ..Journal::default() });
}
pub fn label(point: i32, group: i32, synthetic: bool) {
    JOURNAL.with(|j| { let mut j=j.borrow_mut(); j.point=point; j.group=group; j.synthetic=synthetic; });
}
pub fn parent_begin(context: usize, n: i32, rank: i32, poles: i32, finite: i32, check: i32, cache: i32) {
    JOURNAL.with(|j| {
        let mut j=j.borrow_mut(); if !j.enabled { return; }
        j.next_parent+=1; j.parent=j.next_parent;
        let row=format!("{{\"kind\":\"parent_begin\",\"parent\":{},\"context\":{},\"point\":{},\"group\":{},\"synthetic\":{},\"n\":{},\"rank\":{},\"poles\":{},\"finite\":{},\"check\":{},\"cache\":{},\"mpfr_flags\":{}}}",j.parent,context,j.point,j.group,j.synthetic,n,rank,poles,finite,check,cache,mpfr_flags());
        j.emit(row);
    });
}
pub fn parent_end(status: i32) {
    JOURNAL.with(|j| {
        let mut j=j.borrow_mut(); if !j.enabled { return; }
        let row=format!("{{\"kind\":\"parent_end\",\"parent\":{},\"status\":{},\"mpfr_flags\":{}}}",j.parent,status,mpfr_flags());
        j.emit(row); j.parent=0;
    });
}
pub fn store_state(budget: u64, qp_bytes: u64, dp_bytes: u64, pole_bytes: u64) {
    JOURNAL.with(|j| {
        let mut j=j.borrow_mut(); if !j.enabled { return; }
        let row=format!("{{\"kind\":\"store_state\",\"parent\":{},\"budget\":{},\"qp_bytes\":{},\"dp_bytes\":{},\"pole_bytes\":{}}}",j.parent,budget,qp_bytes,dp_bytes,pole_bytes);
        j.emit(row);
    });
}
pub fn event(context: usize, action: &str) {
    // action is an internal fixed identifier, never arbitrary foreign text.
    JOURNAL.with(|j| {
        let mut j=j.borrow_mut(); if !j.enabled { return; }
        let row=format!("{{\"kind\":\"lifecycle\",\"context\":{},\"action\":\"{}\"}}",context,action);
        j.emit(row);
    });
}
pub fn parent_note(action: &str) {
    JOURNAL.with(|j| {
        let mut j=j.borrow_mut(); if !j.enabled { return; }
        let row=format!("{{\"kind\":\"parent_note\",\"parent\":{},\"action\":\"{}\"}}",j.parent,action);
        j.emit(row);
    });
}
fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8]=b"0123456789abcdef";
    let mut out=String::with_capacity(bytes.len()*2);
    for &b in bytes { out.push(DIGITS[(b>>4) as usize] as char); out.push(DIGITS[(b&15) as usize] as char); }
    out
}
fn float_key<T: NativeFloat>(x: &T, out: &mut Vec<u8>) -> Option<(bool,usize)> {
    let x=(x as &dyn Any).downcast_ref::<Float>()?;
    out.extend_from_slice(&x.prec().to_le_bytes());
    out.push(u8::from(x.is_sign_negative()));
    let mut value=Vec::new(); x.serialize_into(&mut value);
    out.extend_from_slice(&(value.len() as u32).to_le_bytes()); out.extend_from_slice(&value);
    // Payload estimate only: MPFR limbs plus conservative scalar metadata.
    Some((x.is_finite(), ((x.prec() as usize+63)/64)*8+32))
}
pub struct Ticket { id: u64, parent: u64, started: Instant, retained: bool }
pub fn scalar_begin<T: NativeFloat>(family: usize, partial: bool, input: &[Complex<T>]) -> Option<Ticket> {
    if TypeId::of::<T>() != TypeId::of::<Float>() { return None; }
    let (id,parent,retain)=JOURNAL.with(|j| {
        let mut j=j.borrow_mut(); if !j.enabled || j.parent==0 { return (0,0,false); }
        j.scalar_requests+=1;
        let retain=j.scalar_requests<=MAX_SCALARS && j.bytes<MAX_BYTES && j.rows.len()<MAX_EVENTS;
        if !retain { j.dropped_events+=1; }
        (j.scalar_requests,j.parent,retain)
    });
    if id==0 { return None; }
    if retain {
        let flags_before=mpfr_flags(); let started=Instant::now();
        let mut key=Vec::new(); key.extend_from_slice(NAMESPACE.as_bytes()); key.push(0);
        key.push(family as u8); key.push(u8::from(partial));
        key.extend_from_slice(&(input.len() as u32).to_le_bytes());
        let mut finite=true;
        for z in input {
            let (f,_)=float_key(&z.re,&mut key)?; finite&=f;
            let (f,_)=float_key(&z.im,&mut key)?; finite&=f;
        }
        let elapsed=started.elapsed().as_nanos(); let flags_after=mpfr_flags();
        JOURNAL.with(|j| {
            let mut j=j.borrow_mut(); j.key_ns+=elapsed;
            if flags_before!=flags_after { j.key_flag_changes+=1; }
            j.emit(format!("{{\"kind\":\"scalar_begin\",\"id\":{},\"parent\":{},\"family\":{},\"partial\":{},\"input_finite\":{},\"key\":\"{}\",\"key_bytes\":{},\"key_ns\":{},\"flags_before_key\":{},\"flags_after_key\":{}}}",id,parent,family,partial,finite,hex(&key),key.len(),elapsed,flags_before,flags_after));
        });
    }
    Some(Ticket { id,parent,started:Instant::now(),retained:retain })
}
pub fn scalar_end<T: NativeFloat>(ticket: Option<Ticket>, success: bool, output: &[Complex<T>]) {
    let Some(ticket)=ticket else { return; };
    let elapsed=ticket.started.elapsed().as_nanos(); let flags_after_kernel=mpfr_flags();
    let mut finite=true; let mut output_bytes=0;
    if ticket.retained {
        for z in output {
            let mut unused=Vec::new();
            if let Some((f,b))=float_key(&z.re,&mut unused) { finite&=f; output_bytes+=b; } else { finite=false; }
            unused.clear();
            if let Some((f,b))=float_key(&z.im,&mut unused) { finite&=f; output_bytes+=b; } else { finite=false; }
        }
    }
    let flags_after_observation=mpfr_flags();
    JOURNAL.with(|j| {
        let mut j=j.borrow_mut(); j.scalar_completed+=1; j.kernel_ns+=elapsed;
        if !success { j.scalar_errors+=1; }
        if ticket.retained && !finite { j.nonfinite_outputs+=1; }
        if flags_after_kernel!=flags_after_observation { j.output_flag_changes+=1; }
        if ticket.retained {
            j.emit(format!("{{\"kind\":\"scalar_end\",\"id\":{},\"parent\":{},\"success\":{},\"output_finite\":{},\"output_payload_estimate\":{},\"kernel_ns\":{},\"flags_after_kernel\":{},\"flags_after_observation\":{}}}",ticket.id,ticket.parent,success,finite,output_bytes,elapsed,flags_after_kernel,flags_after_observation));
        }
    });
}
pub fn snapshot() -> String {
    JOURNAL.with(|j| {
        let j=j.borrow();
        format!("{{\"schema\":1,\"mpfr_reader_available\":{},\"enabled\":{},\"namespace\":\"{}\",\"max_bytes\":{},\"max_scalar_events\":{},\"retained_json_bytes\":{},\"scalar_requests\":{},\"scalar_completed\":{},\"scalar_errors\":{},\"nonfinite_outputs_observed\":{},\"kernel_ns\":{},\"key_ns\":{},\"dropped_events\":{},\"uniqueness_complete\":{},\"key_flag_changes\":{},\"output_flag_changes\":{},\"events\":[{}]}}",mpfr_reader_available(),j.enabled,NAMESPACE,MAX_BYTES,MAX_SCALARS,j.bytes,j.scalar_requests,j.scalar_completed,j.scalar_errors,j.nonfinite_outputs,j.kernel_ns,j.key_ns,j.dropped_events,j.dropped_events==0,j.key_flag_changes,j.output_flag_changes,j.rows.join(","))
    })
}
