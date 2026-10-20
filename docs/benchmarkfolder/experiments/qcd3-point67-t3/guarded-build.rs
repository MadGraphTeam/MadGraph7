fn main() {
    println!("cargo:rerun-if-changed=src/guarded_fenv.c");
    cc::Build::new().file("src/guarded_fenv.c").compile("ninjars_t3_fenv");
}
