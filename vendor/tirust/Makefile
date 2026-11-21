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

# Linux (qualified) uses GNU ld; macOS uses Apple ld and Mach-O dylibs.
ifeq ($(shell uname -s),Darwin)
SHLIB = libmadloop_tirust.dylib
RUST_LINK_ARGS = -C link-arg=-Wl,-install_name,@rpath/$(SHLIB)
EXPORTED_SYMBOLS = nm -gU
EXPORT_PATTERN = ^_ml_tirust_
SMOKE_RPATH = -Wl,-rpath,@loader_path
else
SHLIB = libmadloop_tirust.so
RUST_LINK_ARGS = -C link-arg=-Wl,-soname,$(SHLIB) -C link-arg=-Wl,--exclude-libs,ALL
EXPORTED_SYMBOLS = nm -D --defined-only
EXPORT_PATTERN = ^ml_tirust_
SMOKE_RPATH = -Wl,-rpath,'$$ORIGIN'
endif

.PHONY: all rust install check clean exports
all: tirust-link-before.txt libtirust.a $(SHLIB) tirust-native-libs.txt

# The platform and compiler check runs before the Rust build.
rust: tirust-link-before.txt
	$(CARGO) rustc --release --locked --lib $(FEATURE_ARGS) -- $(RUST_LINK_ARGS)

$(SHLIB): rust
	cp "$(CARGO_TARGET_DIR)/release/$(SHLIB)" $@

mtirust.o mtirust.mod &: mtirust.f90
	$(FC) $(FFLAGS) -c $< -o mtirust.o

libtirust.a: mtirust.o
	$(AR) rcs $@ mtirust.o

tirust-native-libs.txt:
	printf '%s\n' '-lmadloop_tirust' > $@

# Linux links libgcc_s before the Rust library so that the Fortran runtime
# resolves its unwinder there; macOS's two-level namespace needs nothing.
tirust-link-before.txt:
	@$(FC) --version | head -n 1 | grep -q 'GNU Fortran' || { echo 'TIRust requires GNU Fortran (gfortran)'; exit 1; }
	@case "$$($(FC) -dumpmachine)" in *-linux-*) printf '%s\n' '-lgcc_s' > $@ ;; *-darwin*) : > $@ ;; *) echo 'TIRust builds on Linux and macOS'; exit 1 ;; esac

exports: $(SHLIB)
	@$(EXPORTED_SYMBOLS) $< > exported-symbols.txt
	@awk '{if ($$3 !~ /$(EXPORT_PATTERN)/) print $$3}' exported-symbols.txt > exported-unexpected.txt; test ! -s exported-unexpected.txt || { cat exported-unexpected.txt; echo 'Unexpected shared-library exports'; exit 1; }

install: all exports
	mkdir -p "$(PREFIX)/lib" "$(PREFIX)/include"
	cp libtirust.a $(SHLIB) mtirust.mod tirust-native-libs.txt tirust-link-before.txt "$(PREFIX)/lib/"
	cp tirust.h mtirust.f90 "$(PREFIX)/include/"

check: all exports
	$(CARGO) test --locked $(FEATURE_ARGS) -- --test-threads=1
	$(FC) $(FFLAGS) smoke.f90 -I. $$(cat tirust-link-before.txt) -L. -ltirust $$(cat tirust-native-libs.txt) $(SMOKE_RPATH) -o smoke
	./smoke

clean:
	rm -f mtirust.o mtirust.mod libtirust.a libmadloop_tirust.so libmadloop_tirust.dylib smoke tirust-native-libs.txt tirust-link-before.txt exported-symbols.txt exported-unexpected.txt
