SHELL := bash
FC = gfortran
AR ?= ar
CARGO ?= cargo
PREFIX ?= $(CURDIR)/install
CARGO_TARGET_DIR ?= $(CURDIR)/target
export CARGO_TARGET_DIR
FFLAGS ?= -O3 -fPIC

.PHONY: all rust install check clean
all: libninjars.a

rust:
	@set -o pipefail; $(CARGO) rustc --release --locked --lib -- --print native-static-libs 2>&1 | tee build.log
	@sed -n 's/^.*native-static-libs: //p' build.log | tail -n 1 > ninjars-native-libs.txt
	@test -s ninjars-native-libs.txt || { echo 'Rust did not report its required native linker flags'; exit 1; }

mninjars.o: mninjars.f90
	$(FC) $(FFLAGS) -c $< -o $@

libninjars.a: rust mninjars.o
	cp "$(CARGO_TARGET_DIR)/release/libmadloop_ninjars.a" $@
	$(AR) rcs $@ mninjars.o

install: libninjars.a
	mkdir -p "$(PREFIX)/lib" "$(PREFIX)/include"
	cp libninjars.a mninjars.mod ninjars-native-libs.txt "$(PREFIX)/lib/"
	cp ninjars.h "$(PREFIX)/include/"

check: libninjars.a
	$(CARGO) test --locked -- --test-threads=1
	$(FC) $(FFLAGS) smoke.f90 -I. -L. -lninjars $$(cat ninjars-native-libs.txt) -o smoke
	./smoke

clean:
	rm -f mninjars.o mninjars.mod libninjars.a smoke build.log ninjars-native-libs.txt
