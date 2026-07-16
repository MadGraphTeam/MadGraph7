# SPDX-License-Identifier: 0BSD
RUSTICOL_CONFIG ?= rusticol-config
RUSTC = rustc
ARTIFACT_DIR := $(abspath ../..)
BUILD_DIR ?= $(abspath ../../..)/.pyamplicol-api-build/$(notdir $(ARTIFACT_DIR))/rust
TARGET := $(BUILD_DIR)/check_standalone
RUSTFLAGS ?= -C opt-level=2
RUSTFLAGS += --edition=2021
RUSTICOL_RUSTFLAGS := $(shell $(RUSTICOL_CONFIG) --rustflags)

.PHONY: all check_standalone clean run

all: $(TARGET)

check_standalone: $(TARGET)

$(BUILD_DIR):
	mkdir -p "$@"

$(TARGET): check_standalone.rs | $(BUILD_DIR)
	$(RUSTC) $(RUSTFLAGS) "$<" -o "$@" $(RUSTICOL_RUSTFLAGS)

run: $(TARGET)
	cd "$(ARTIFACT_DIR)" && "$(TARGET)" $(ARGS)

clean:
	$(RM) "$(TARGET)"
