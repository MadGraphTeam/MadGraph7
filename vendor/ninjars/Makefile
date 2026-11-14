SHELL := bash
FC = gfortran
AR ?= ar
CARGO ?= cargo
PYTHON ?= python3
PREFIX ?= $(CURDIR)/install
CARGO_TARGET_DIR ?= $(CURDIR)/target
OEM_TEST_OUTPUT ?= $(CARGO_TARGET_DIR)/oem-check
export CARGO_TARGET_DIR
FFLAGS ?= -O3 -fPIC

.PHONY: all rust preflight compiler-runtime install check check-oem clean
all: libninjars.a

preflight: compiler-runtime
	@command -v $(CARGO) >/dev/null 2>&1 || { echo 'Installing NinjaRS requires Cargo/Rust 1.96 or later on PATH'; exit 1; }

rust: preflight
	@set -o pipefail; $(CARGO) rustc --release --locked --lib -- --print native-static-libs 2>&1 | tee build.log
	@sed -n 's/^.*native-static-libs: //p' build.log | tail -n 1 > ninjars-native-libs.txt
	@test -s ninjars-native-libs.txt || { echo 'Rust did not report its required native linker flags'; exit 1; }

mninjars.o: mninjars.f90 | preflight
	$(FC) $(FFLAGS) -c $< -o $@

compiler-runtime:
	@test "$$(uname -s)" = Linux || { echo 'NinjaRS is currently supported on Linux with GNU Fortran'; exit 1; }
	@$(FC) --version | head -n 1 | grep -q 'GNU Fortran' || { echo 'NinjaRS currently validates compiler-runtime isolation with GNU Fortran on Linux'; exit 1; }
	@case "$$($(FC) -dumpmachine)" in *-linux-*) printf '%s\n' '-lgcc_s' > ninjars-link-before.txt ;; *) echo 'NinjaRS compiler-runtime isolation is currently validated on Linux'; exit 1 ;; esac

libninjars.a: rust mninjars.o
	cp "$(CARGO_TARGET_DIR)/release/libmadloop_ninjars.a" $@
	$(AR) rcs $@ mninjars.o

install: libninjars.a
	mkdir -p "$(PREFIX)/lib" "$(PREFIX)/include"
	cp libninjars.a mninjars.mod ninjars-native-libs.txt ninjars-link-before.txt "$(PREFIX)/lib/"
	cp ninjars.h mninjars.f90 "$(PREFIX)/include/"

check: libninjars.a
	$(CARGO) test --locked --features fenv-test-support -- --test-threads=1
	$(FC) $(FFLAGS) smoke.f90 -I. $$(cat ninjars-link-before.txt) -L. -lninjars $$(cat ninjars-native-libs.txt) -o smoke
	./smoke

check-oem: libninjars.a
	$(PYTHON) tests/oem.py --lib "$(CURDIR)" --output "$(OEM_TEST_OUTPUT)"

clean:
	rm -f mninjars.o mninjars.mod libninjars.a smoke build.log ninjars-native-libs.txt ninjars-link-before.txt
