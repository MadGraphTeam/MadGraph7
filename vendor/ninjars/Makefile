SHELL := bash
FC = gfortran
AR ?= ar
CARGO ?= cargo
PREFIX ?= $(CURDIR)/install
CARGO_TARGET_DIR ?= $(CURDIR)/target
export CARGO_TARGET_DIR
FFLAGS ?= -O3 -fPIC

.PHONY: all rust compiler-runtime install check clean
all: libninjars.a

rust:
	@set -o pipefail; $(CARGO) rustc --release --locked --lib -- --print native-static-libs 2>&1 | tee build.log
	@sed -n 's/^.*native-static-libs: //p' build.log | tail -n 1 > ninjars-native-libs.txt
	@test -s ninjars-native-libs.txt || { echo 'Rust did not report its required native linker flags'; exit 1; }

mninjars.o: mninjars.f90
	$(FC) $(FFLAGS) -c $< -o $@

compiler-runtime:
	@$(FC) --version | head -n 1 | grep -q 'GNU Fortran' || { echo 'NinjaRS currently validates compiler-runtime isolation with GNU Fortran on Linux'; exit 1; }
	@case "$$($(FC) -dumpmachine)" in *-linux-*) printf '%s\n' '-lgcc_s' > ninjars-link-before.txt ;; *) echo 'NinjaRS compiler-runtime isolation is currently validated on Linux'; exit 1 ;; esac

libninjars.a: rust mninjars.o compiler-runtime
	cp "$(CARGO_TARGET_DIR)/release/libmadloop_ninjars.a" $@
	$(AR) rcs $@ mninjars.o

install: libninjars.a
	mkdir -p "$(PREFIX)/lib" "$(PREFIX)/include"
	cp libninjars.a mninjars.mod ninjars-native-libs.txt ninjars-link-before.txt "$(PREFIX)/lib/"
	cp ninjars.h "$(PREFIX)/include/"

check: libninjars.a
	$(CARGO) test --locked -- --test-threads=1
	$(FC) $(FFLAGS) smoke.f90 -I. $$(cat ninjars-link-before.txt) -L. -lninjars $$(cat ninjars-native-libs.txt) -o smoke
	./smoke

clean:
	rm -f mninjars.o mninjars.mod libninjars.a smoke build.log ninjars-native-libs.txt ninjars-link-before.txt
