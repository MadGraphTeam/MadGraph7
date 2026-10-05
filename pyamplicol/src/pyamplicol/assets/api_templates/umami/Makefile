# SPDX-License-Identifier: 0BSD
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
$(error rusticol-config is unavailable; activate the pyAmpliCol environment or set RUSTICOL_CONFIG=/absolute/path)
endif
include providers.mk
CC ?= cc
ARTIFACT_DIR := $(abspath ../..)
BUILD_DIR ?= $(abspath ../../..)/.pyamplicol-api-build/$(notdir $(ARTIFACT_DIR))/umami
CFLAGS ?= -O2
CFLAGS += -std=c11 -Wall -Wextra -pedantic
SDK_CFLAGS := $(shell "$(RUSTICOL_CONFIG_PATH)" --cflags)
SDK_LIBS := $(shell "$(RUSTICOL_CONFIG_PATH)" --libs)
SDK_LIBRARY := $(shell "$(RUSTICOL_CONFIG_PATH)" --library)
HOST_SYSTEM := $(shell uname -s)
ifeq ($(HOST_SYSTEM),Darwin)
SHARED_FLAGS = -dynamiclib -Wl,-install_name,@rpath/$(@F) -Wl,-exported_symbols_list,umami.exports.macos
DRIVER_FLAGS = -Wl,-rpath,@loader_path
else ifeq ($(HOST_SYSTEM),Linux)
SHARED_FLAGS = -shared -Wl,-z,defs -Wl,-soname,$(@F) -Wl,--version-script=umami.exports
DRIVER_FLAGS = -Wl,-rpath,'$$ORIGIN'
else
$(error UMAMI shared-library examples support Linux and macOS)
endif
ifeq ($(UMAMI_SINGLE_PROVIDER),1)
LIBRARIES := $(BUILD_DIR)/libumami.so
DRIVERS := $(BUILD_DIR)/umami_driver
RUN_DRIVER := $(BUILD_DIR)/umami_driver
else
LIBRARIES := $(addprefix $(BUILD_DIR)/libumami_,$(addsuffix .so,$(UMAMI_PROVIDERS)))
DRIVERS := $(addprefix $(BUILD_DIR)/umami_driver_,$(UMAMI_PROVIDERS))
PROVIDER ?= $(firstword $(UMAMI_PROVIDERS))
RUN_DRIVER := $(BUILD_DIR)/umami_driver_$(PROVIDER)
endif

.PHONY: all shared run clean
all: $(LIBRARIES) $(DRIVERS)
shared: $(LIBRARIES)

$(BUILD_DIR):
	mkdir -p "$@"

$(BUILD_DIR)/libumami.so: umami.c provider_p0.h umami_provider.h umami.h umami.exports umami.exports.macos $(SDK_LIBRARY) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(SDK_CFLAGS) $(CFLAGS) -fPIC -DUMAMI_PROVIDER_HEADER='"provider_p0.h"' $(SHARED_FLAGS) "$<" -o "$@" $(LDFLAGS) $(SDK_LIBS)

$(BUILD_DIR)/libumami_%.so: umami.c provider_%.h umami_provider.h umami.h umami.exports umami.exports.macos $(SDK_LIBRARY) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(SDK_CFLAGS) $(CFLAGS) -fPIC -DUMAMI_PROVIDER_HEADER='"provider_$*.h"' $(SHARED_FLAGS) "$<" -o "$@" $(LDFLAGS) $(SDK_LIBS)

$(BUILD_DIR)/umami_driver: umami_driver.c $(BUILD_DIR)/libumami.so provider_p0.h umami_provider.h umami.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DUMAMI_PROVIDER_HEADER='"provider_p0.h"' "$<" "$(BUILD_DIR)/libumami.so" -o "$@" $(LDFLAGS) $(DRIVER_FLAGS)

$(BUILD_DIR)/umami_driver_%: umami_driver.c $(BUILD_DIR)/libumami_%.so provider_%.h umami_provider.h umami.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DUMAMI_PROVIDER_HEADER='"provider_$*.h"' "$<" "$(BUILD_DIR)/libumami_$*.so" -o "$@" $(LDFLAGS) $(DRIVER_FLAGS)

run: $(RUN_DRIVER)
	cd "$(ARTIFACT_DIR)" && "$(RUN_DRIVER)" "$(ARTIFACT_DIR)" $(ARGS)

clean:
	$(RM) $(LIBRARIES) $(DRIVERS)
