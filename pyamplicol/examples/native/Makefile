# SPDX-License-Identifier: 0BSD

CC ?= cc
CXX ?= c++
RUSTC ?= rustc
ifeq ($(origin FC), default)
FC := gfortran
endif
FC ?= gfortran
CFLAGS ?= -O2
CXXFLAGS ?= -O2
FFLAGS ?= -O2
RUSTICOL_CONFIG ?= rusticol-config
RUSTICOL_CONFIG_PATH := $(shell \
	if command -v "$(RUSTICOL_CONFIG)" >/dev/null 2>&1; then \
		command -v "$(RUSTICOL_CONFIG)"; \
	elif [ "$(RUSTICOL_CONFIG)" = rusticol-config ]; then \
		directory="$(CURDIR)"; \
		while [ "$$directory" != / ]; do \
			candidate="$$directory/.venv/bin/rusticol-config"; \
			if [ -x "$$candidate" ]; then printf '%s\n' "$$candidate"; break; fi; \
			directory=$$(dirname "$$directory"); \
		done; \
	fi)
ifeq ($(strip $(RUSTICOL_CONFIG_PATH)),)
ifeq ($(strip $(LOOP_SDK_LIBS)),)
$(error rusticol-config is unavailable; activate the pyAmpliCol environment or set RUSTICOL_CONFIG=/absolute/path)
endif
else
RUSTICOL_CFLAGS := $(shell "$(RUSTICOL_CONFIG_PATH)" --cflags)
RUSTICOL_LIBS := $(shell "$(RUSTICOL_CONFIG_PATH)" --libs)
RUSTICOL_FORTRAN := $(shell "$(RUSTICOL_CONFIG_PATH)" --fortran-source)
RUSTICOL_RUST := $(shell "$(RUSTICOL_CONFIG_PATH)" --rust-source)
RUSTICOL_RUSTFLAGS := $(shell "$(RUSTICOL_CONFIG_PATH)" --rustflags)
endif
# The one-loop client links the installed SDK's loop C ABI (rusticol_virtual.h).
# A source checkout without the wheel's SDK passes the loop SDK instead:
#   make loop LOOP_SDK_CFLAGS=-I<checkout>/rust/loop-reduction/sdk/include \
#     LOOP_SDK_LIBS="-L<native build> -lrusticol_loop_sdk -Wl,-rpath,<native build>"
LOOP_SDK_CFLAGS ?= $(RUSTICOL_CFLAGS)
LOOP_SDK_LIBS ?= $(RUSTICOL_LIBS)

.PHONY: all correlated loop
all: runtime_cpp runtime_fortran
correlated: correlated_c correlated_cpp correlated_fortran correlated_rust
loop: loop_client

runtime_cpp: runtime.cpp
	$(CXX) $(CXXFLAGS) -std=c++17 $< $(RUSTICOL_CFLAGS) \
		$(RUSTICOL_LIBS) -o $@

runtime_fortran: runtime.f90
	$(FC) $(FFLAGS) -std=f2008 "$(RUSTICOL_FORTRAN)" $< \
		$(RUSTICOL_LIBS) -o $@

correlated_c: correlated.c
	$(CC) $(CFLAGS) -std=c11 $< $(RUSTICOL_CFLAGS) $(RUSTICOL_LIBS) -o $@

correlated_cpp: correlated.cpp
	$(CXX) $(CXXFLAGS) -std=c++17 $< $(RUSTICOL_CFLAGS) $(RUSTICOL_LIBS) -o $@

correlated_fortran: correlated.f90
	$(FC) $(FFLAGS) -std=f2008 "$(RUSTICOL_FORTRAN)" $< $(RUSTICOL_LIBS) -o $@

correlated_rust: correlated.rs
	RUSTICOL_RUST_SOURCE="$(RUSTICOL_RUST)" $(RUSTC) --edition=2021 \
		-C opt-level=2 $< $(RUSTICOL_RUSTFLAGS) -o $@

loop_client: loop_client.c
	$(CC) $(CFLAGS) -std=c11 $< $(LOOP_SDK_CFLAGS) $(LOOP_SDK_LIBS) -lm -o $@
