# SPDX-License-Identifier: 0BSD
RUSTICOL_CONFIG ?= rusticol-config
CC ?= cc
ARTIFACT_DIR := $(abspath ../..)
BUILD_DIR ?= $(abspath ../../..)/.pyamplicol-api-build/$(notdir $(ARTIFACT_DIR))/c
TARGET := $(BUILD_DIR)/check_standalone
CPPFLAGS += $(shell $(RUSTICOL_CONFIG) --cflags)
CFLAGS ?= -O2
CFLAGS += -std=c11 -Wall -Wextra -pedantic
LDLIBS += $(shell $(RUSTICOL_CONFIG) --libs)

.PHONY: all check_standalone clean run

all: $(TARGET)

check_standalone: $(TARGET)

$(BUILD_DIR):
	mkdir -p "$@"

$(TARGET): check_standalone.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) "$<" -o "$@" $(LDLIBS)

run: $(TARGET)
	cd "$(ARTIFACT_DIR)" && "$(TARGET)" $(ARGS)

clean:
	$(RM) "$(TARGET)"
