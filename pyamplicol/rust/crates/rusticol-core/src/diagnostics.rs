// SPDX-License-Identifier: 0BSD

//! Symbolica's diagnostics on standard error.
//!
//! Symbolica installs a default `tracing` subscriber writing to standard
//! *output* at its first message, for example when a saved evaluator of
//! Python Symbolica 3.0.3 (which saves the producer's whole state) imports a
//! symbol without the user-defined functions this process gave it. In an
//! extension module that output interleaves with the program's own
//! (`pyamplicol self-test --json` stopped being JSON). The extension installs
//! the same subscriber (`RUST_LOG`, default level INFO) writing to standard
//! error at import instead, and Symbolica's own is never installed.

use std::io::IsTerminal;

/// Route this library's `tracing` events (Symbolica's diagnostics) to
/// standard error. Idempotent; a subscriber already installed in this library
/// is kept.
pub fn symbolica_diagnostics_on_stderr() {
    use tracing_subscriber::{EnvFilter, filter::LevelFilter};
    let subscriber = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_ansi(std::io::stderr().is_terminal())
        .with_env_filter(
            EnvFilter::builder()
                .with_default_directive(LevelFilter::INFO.into())
                .from_env_lossy(),
        )
        .finish();
    let _ = tracing::subscriber::set_global_default(subscriber);
    symbolica::GLOBAL_SETTINGS
        .initialize_tracing
        .store(false, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    const CHILD: &str = "RUSTICOL_DIAGNOSTICS_CHILD";

    /// A Symbolica warning (log(0)) after the routing reaches standard error
    /// and leaves standard output empty (checked in a child process, which
    /// owns its global subscriber).
    #[test]
    fn symbolica_warnings_go_to_standard_error() {
        if std::env::var_os(CHILD).is_some() {
            super::symbolica_diagnostics_on_stderr();
            let _unlock = crate::licence::unlock();
            let value = symbolica::parse!("log(0)");
            assert!(!value.to_string().is_empty());
            return;
        }
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "diagnostics::tests::symbolica_warnings_go_to_standard_error",
                "--test-threads=1",
                "--quiet",
            ])
            .env(CHILD, "1")
            .env_remove("RUST_LOG")
            .output()
            .expect("run the test binary as a child");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(output.status.success(), "child failed: {stdout}\n{stderr}");
        assert!(
            stderr.contains("Created infinity by log(0)"),
            "the warning is on standard error: {stderr}"
        );
        assert!(
            !stdout.contains("log(0)"),
            "standard output carries no diagnostics: {stdout}"
        );
    }
}
