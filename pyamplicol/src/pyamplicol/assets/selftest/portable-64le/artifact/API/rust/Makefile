# SPDX-License-Identifier: 0BSD
RUSTICOL_CONFIG ?= rusticol-config
RUSTC ?= rustc
RUST_SCRIPT ?= rust-script
ARTIFACT_DIR := $(abspath ../..)
BUILD_DIR ?= $(abspath ../../..)/.pyamplicol-api-build/$(notdir $(ARTIFACT_DIR))/rust
TARGET := $(BUILD_DIR)/check_standalone
RUSTFLAGS ?= -C opt-level=2
RUSTICOL_RUST_SOURCE := $(shell $(RUSTICOL_CONFIG) --rust-source)
RUSTICOL_RUSTFLAGS := $(shell $(RUSTICOL_CONFIG) --rustflags)

.PHONY: all check_standalone clean run run-script

all: $(TARGET)

check_standalone: $(TARGET)

$(BUILD_DIR):
	mkdir -p "$@"

$(TARGET): check_standalone.rs | $(BUILD_DIR)
	RUSTICOL_RUST_SOURCE="$(RUSTICOL_RUST_SOURCE)" \
		$(RUSTC) --edition=2021 $(RUSTFLAGS) "$<" -o "$@" $(RUSTICOL_RUSTFLAGS)

run: $(TARGET)
	cd "$(ARTIFACT_DIR)" && "$(TARGET)" $(ARGS)

run-script:
	cd "$(ARTIFACT_DIR)" && \
		RUSTICOL_RUST_SOURCE="$(RUSTICOL_RUST_SOURCE)" \
		CARGO_ENCODED_RUSTFLAGS="$$($(RUSTICOL_CONFIG) --cargo-rustflags)" \
		$(RUST_SCRIPT) "API/rust/check_standalone.rs" -- $(ARGS)

clean:
	$(RM) "$(TARGET)"
