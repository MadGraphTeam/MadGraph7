SHELL := bash
FC = gfortran
AR ?= ar
CARGO ?= cargo
PREFIX ?= $(CURDIR)/install
CARGO_TARGET_DIR ?= $(CURDIR)/target
export CARGO_TARGET_DIR
FFLAGS ?= -O3 -fPIC
CARGO_FEATURES ?= poles-only
FEATURE_ARGS = $(if $(strip $(CARGO_FEATURES)),--features $(CARGO_FEATURES),)

.PHONY: all rust install check clean exports
all: libtirust.a libmadloop_tirust.so tirust-native-libs.txt tirust-link-before.txt

rust:
	$(CARGO) rustc --release --locked --lib $(FEATURE_ARGS) -- -C link-arg=-Wl,-soname,libmadloop_tirust.so -C link-arg=-Wl,--exclude-libs,ALL

libmadloop_tirust.so: rust
	cp "$(CARGO_TARGET_DIR)/release/libmadloop_tirust.so" $@

mtirust.o mtirust.mod &: mtirust.f90
	$(FC) $(FFLAGS) -c $< -o mtirust.o

libtirust.a: mtirust.o
	$(AR) rcs $@ mtirust.o

tirust-native-libs.txt:
	printf '%s\n' '-lmadloop_tirust' > $@

tirust-link-before.txt:
	@$(FC) --version | head -n 1 | grep -q 'GNU Fortran' || { echo 'TIRust currently qualifies GNU Fortran on Linux'; exit 1; }
	@case "$$($(FC) -dumpmachine)" in *-linux-*) printf '%s\n' '-lgcc_s' > $@ ;; *) echo 'TIRust currently qualifies Linux'; exit 1 ;; esac

exports: libmadloop_tirust.so
	@nm -D --defined-only $< > exported-symbols.txt
	@awk '{if ($$3 !~ /^ml_tirust_/) print $$3}' exported-symbols.txt > exported-unexpected.txt; test ! -s exported-unexpected.txt || { cat exported-unexpected.txt; echo 'Unexpected shared-library exports'; exit 1; }

install: all exports
	mkdir -p "$(PREFIX)/lib" "$(PREFIX)/include"
	cp libtirust.a libmadloop_tirust.so mtirust.mod tirust-native-libs.txt tirust-link-before.txt "$(PREFIX)/lib/"
	cp tirust.h mtirust.f90 "$(PREFIX)/include/"

check: all exports
	$(CARGO) test --locked $(FEATURE_ARGS) -- --test-threads=1
	$(FC) $(FFLAGS) smoke.f90 -I. $$(cat tirust-link-before.txt) -L. -ltirust $$(cat tirust-native-libs.txt) -Wl,-rpath,'$$ORIGIN' -o smoke
	./smoke

clean:
	rm -f mtirust.o mtirust.mod libtirust.a libmadloop_tirust.so smoke tirust-native-libs.txt tirust-link-before.txt exported-symbols.txt exported-unexpected.txt
