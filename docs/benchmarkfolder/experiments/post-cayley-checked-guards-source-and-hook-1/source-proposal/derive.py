#!/usr/bin/env python3
"""Private text derivation only; no compiler/native invocation."""
from pathlib import Path
import difflib, json, hashlib, shutil, sys
ROOT=Path(__file__).resolve().parent
FEATURE='checked-dilog-rounding'
HOOK=r'''
// Optional, scoped, thread-local evidence supplied by an embedding host.
// The probe must only inspect current floating-point controls; it must neither
// change the environment nor evaluate scientific kernels. Unknown means false.
#[cfg(feature = "checked-dilog-rounding")]
mod checked_rounding {
    use std::cell::Cell;
    std::thread_local! {
        static PROBE: Cell<Option<fn() -> bool>> = const { Cell::new(None) };
    }
    struct Restore(Option<fn() -> bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            PROBE.with(|slot| slot.set(self.0));
        }
    }
    /// Install a read-only, fresh-control probe only for this thread and scope.
    /// No pointer is retained after normal return or Rust panic unwinding.
    /// Returning true promises binary64 nearest rounding and no FTZ/DAZ.
    /// The host owns this promise; false/absent probes retain the old arithmetic.
    pub fn with_probe<R>(probe: fn() -> bool, action: impl FnOnce() -> R) -> R {
        let previous = PROBE.with(|slot| slot.replace(Some(probe)));
        let _restore = Restore(previous);
        action()
    }
    pub(super) fn now() -> bool {
        // No Cell access or borrow remains active while calling host code.
        // Reentry from a probe sees no probe and therefore the old path.
        let previous = PROBE.with(|slot| slot.replace(None));
        let _restore = Restore(previous);
        previous.is_some_and(|probe| probe())
    }
    #[cfg(test)]
    mod tests {
        use super::*;
        fn yes() -> bool { true }
        fn no() -> bool { false }
        fn reentrant() -> bool { !now() }
        fn panic_probe() -> bool { panic!("probe unwind"); }
        #[test]
        fn default_nesting_reentry_and_thread_isolation() {
            assert!(!now());
            with_probe(yes, || {
                assert!(now());
                with_probe(no, || assert!(!now()));
                assert!(now());
                std::thread::spawn(|| assert!(!now())).join().unwrap();
                with_probe(reentrant, || assert!(now()));
            });
            assert!(!now());
        }
        #[test]
        fn restores_action_and_probe_panics() {
            with_probe(yes, || {
                let result = std::panic::catch_unwind(|| with_probe(no, || panic!("action unwind")));
                assert!(result.is_err());
                assert!(now());
                let result = std::panic::catch_unwind(|| with_probe(panic_probe, now));
                assert!(result.is_err());
                assert!(now());
            });
            assert!(!now());
        }
    }
}
#[cfg(feature = "checked-dilog-rounding")]
pub use checked_rounding::with_probe as with_checked_dilog_rounding_probe;

'''
BRIDGE=r'''
// Private, feature-disabled by default. The probe object is separately compiled
// and hidden-linked in a future admitted build, never loaded from another DSO.
#[cfg(feature = "checked-dilog-rounding")]
unsafe extern "C" {
    fn ml_private_kinematic_environment_token() -> i64;
}
#[cfg(feature = "checked-dilog-rounding")]
fn current_dilog_rounding_is_checked() -> bool {
    // This C routine only reads MXCSR/x87 control words; it changes no controls
    // or status flags. Unsupported targets return zero and use the old path.
    let token = unsafe { ml_private_kinematic_environment_token() } as u64;
    if token >> 32 != 0x4b45_0000 { return false; }
    let csr = (token >> 16) & 0xffff;
    let cw = token & 0xffff;
    csr & 0x003f == 0 && cw & 0xe0c0 == 0 && csr & 0xe040 == 0
        && cw & 0x0c00 == 0 && cw & 0x0300 == 0x0300
}
fn numerical_scope<T>(action: impl FnOnce() -> T) -> T {
    #[cfg(feature = "checked-dilog-rounding")]
    {
        iregi_rust::with_checked_dilog_rounding_probe(current_dilog_rounding_is_checked, action)
    }
    #[cfg(not(feature = "checked-dilog-rounding"))]
    { action() }
}

'''
def replace(s, old, new):
    assert s.count(old)==1, (old[:120],s.count(old))
    return s.replace(old,new)
def derive(out):
    shutil.copytree(ROOT/'original',out)
    def edit(name, fn):
        p=out/'sources'/name;p.write_text(fn(p.read_text()))
    edit('pole-development/crates/iregi-rust/src/direct/exec/gram.rs',lambda s:replace(s,
        '    let w = largest / cnorm2(&delta).to_f64().sqrt();',
        '    let denominator = cnorm2(&delta).to_f64().sqrt();\n    if denominator.to_bits() & 0x7fff_ffff_ffff_ffff == 0 {\n        return None;\n    }\n    let w = largest / denominator;'))
    edit('oneloopmaster/Cargo.toml', lambda s:replace(s,'rust-codegen = []','rust-codegen = []\n# Private opt-in; a missing/false host probe leaves the original estimator.\nchecked-dilog-rounding = []'))
    def dilog(s):
        s=replace(s,'/// Series terms needed at `bits`',HOOK+'/// Series terms needed at `bits`')
        old='    let ratio = 4.0 * std::f64::consts::PI * std::f64::consts::PI / modulus;'
        new='''    #[cfg(feature = "checked-dilog-rounding")]
    let ratio = if modulus.to_bits() > 0 && modulus.to_bits() <= 0x0030_0000_0000_0000
        && checked_rounding::now()
    {
        // Under the fresh checked contract the old binary64 ratio overflows
        // to positive infinity. Preserve the old masked coefficient count.
        // This intentionally suppresses old overflow/inexact status effects.
        f64::INFINITY
    } else {
        4.0 * std::f64::consts::PI * std::f64::consts::PI / modulus
    };
    #[cfg(not(feature = "checked-dilog-rounding"))]
'''+old
        return replace(s,old,new)
    edit('oneloopmaster/src/native/dilog.rs',dilog)
    export='#[cfg(feature = "checked-dilog-rounding")]\npub use dilog::with_checked_dilog_rounding_probe;\n'
    edit('oneloopmaster/src/native/mod.rs',lambda s:replace(s,'mod dilog;','mod dilog;\n'+export))
    edit('oneloopmaster/src/lib.rs',lambda s:replace(s,'mod native;','mod native;\n#[cfg(feature = "checked-dilog-rounding")]\npub use native::with_checked_dilog_rounding_probe;'))
    edit('pole-development/crates/iregi-rust/Cargo.toml',lambda s:replace(s,'native-f64-abc = ["iregi-masters/native-f64-abc"]','native-f64-abc = ["iregi-masters/native-f64-abc"]\nchecked-dilog-rounding = ["oneloop/checked-dilog-rounding"]'))
    edit('pole-development/crates/iregi-rust/src/lib.rs',lambda s:replace(s,'#![forbid(unsafe_code)]','#![forbid(unsafe_code)]\n\n#[cfg(feature = "checked-dilog-rounding")]\npub use oneloop::with_checked_dilog_rounding_probe;'))
    def cargo(s):
        s=replace(s,'poles-only = []','poles-only = []\nchecked-dilog-rounding = ["iregi-rust/checked-dilog-rounding"]')
        return replace(s,'[profile.release]','[patch."https://github.com/alphal00p/oneloopmaster.git"]\noneloop = { path = "../oneloopmaster" }\n\n[profile.release]')
    edit('adaptive-qp-rescue-prototype3/Cargo.toml',cargo)
    def bridge(s):
        s=replace(s,'fn guard(f: impl FnOnce() -> Result<i32>) -> i32 {',BRIDGE+'fn guard(f: impl FnOnce() -> Result<i32>) -> i32 {')
        return replace(s,'match catch_unwind(AssertUnwindSafe(f)) {','match catch_unwind(AssertUnwindSafe(|| numerical_scope(f))) {')
    edit('adaptive-qp-rescue-prototype3/src/lib.rs',bridge)
    return out
if __name__=='__main__':
    out=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'candidate'
    derive(out)
    if len(sys.argv)==1:
        patches=ROOT/'patches';patches.mkdir()
        pins={}
        for f in sorted((ROOT/'original').rglob('*')):
            if not f.is_file():continue
            rel=f.relative_to(ROOT/'original');g=out/rel
            pins[str(rel)]={'original':hashlib.sha256(f.read_bytes()).hexdigest(),'candidate':hashlib.sha256(g.read_bytes()).hexdigest()}
            if f.read_bytes()!=g.read_bytes():
                patch=''.join(difflib.unified_diff(f.read_text().splitlines(True),g.read_text().splitlines(True),fromfile='a/'+str(rel),tofile='b/'+str(rel)))
                (patches/(str(rel).replace('/','__')+'.patch')).write_text(patch)
        (ROOT/'derived-source-pins.json').write_text(json.dumps(pins,indent=2)+'\n')
