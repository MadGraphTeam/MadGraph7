#![forbid(unsafe_code)]

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
