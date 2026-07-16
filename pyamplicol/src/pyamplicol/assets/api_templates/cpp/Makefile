# SPDX-License-Identifier: 0BSD
RUSTICOL_CONFIG ?= rusticol-config
CXX = c++
ARTIFACT_DIR := $(abspath ../..)
BUILD_DIR ?= $(abspath ../../..)/.pyamplicol-api-build/$(notdir $(ARTIFACT_DIR))/cpp
TARGET := $(BUILD_DIR)/check_standalone
CPPFLAGS += $(shell $(RUSTICOL_CONFIG) --cflags)
CXXFLAGS ?= -O2
CXXFLAGS += -std=c++17 -Wall -Wextra -pedantic
LDLIBS += $(shell $(RUSTICOL_CONFIG) --libs)

.PHONY: all check_standalone clean run

all: $(TARGET)

check_standalone: $(TARGET)

$(BUILD_DIR):
	mkdir -p "$@"

$(TARGET): check_standalone.cpp | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) "$<" -o "$@" $(LDLIBS)

run: $(TARGET)
	cd "$(ARTIFACT_DIR)" && "$(TARGET)" $(ARGS)

clean:
	$(RM) "$(TARGET)"
