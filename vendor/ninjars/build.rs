fn main() {
    println!("cargo:rerun-if-changed=src/guarded_fenv.c");
    cc::Build::new().file("src/guarded_fenv.c").compile("ninjars_t3_fenv");
    if std::env::var_os("CARGO_FEATURE_FENV_TEST_SUPPORT").is_some() {
        println!("cargo:rerun-if-changed=tests/fenv_helpers.c");
        cc::Build::new().file("tests/fenv_helpers.c").compile("ninjars_test_fenv");
    }
}
