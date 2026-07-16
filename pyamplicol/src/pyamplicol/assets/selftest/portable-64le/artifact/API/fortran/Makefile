# SPDX-License-Identifier: 0BSD
RUSTICOL_CONFIG ?= rusticol-config
FC = gfortran
ARTIFACT_DIR := $(abspath ../..)
BUILD_DIR ?= $(abspath ../../..)/.pyamplicol-api-build/$(notdir $(ARTIFACT_DIR))/fortran
TARGET := $(BUILD_DIR)/check_standalone
RUSTICOL_FORTRAN := $(shell $(RUSTICOL_CONFIG) --fortran-source)
FFLAGS ?= -O2
FFLAGS += -std=f2008 -ffree-line-length-none -Wall -Wextra
LDLIBS += $(shell $(RUSTICOL_CONFIG) --libs)

.PHONY: all check_standalone clean run

all: $(TARGET)

check_standalone: $(TARGET)

$(BUILD_DIR):
	mkdir -p "$@"

$(TARGET): check_standalone.f90 $(RUSTICOL_FORTRAN) | $(BUILD_DIR)
	$(FC) $(FFLAGS) -J"$(BUILD_DIR)" "$(RUSTICOL_FORTRAN)" "$<" -o "$@" $(LDLIBS)

run: $(TARGET)
	cd "$(ARTIFACT_DIR)" && "$(TARGET)" $(ARGS)

clean:
	$(RM) "$(TARGET)" "$(BUILD_DIR)/rusticol.mod"
