# SPDX-License-Identifier: 0BSD

CXX ?= c++
FC ?= gfortran
CXXFLAGS ?= -O2
FFLAGS ?= -O2
RUSTICOL_CONFIG ?= rusticol-config

RUSTICOL_CFLAGS := $(shell $(RUSTICOL_CONFIG) --cflags)
RUSTICOL_LIBS := $(shell $(RUSTICOL_CONFIG) --libs)
RUSTICOL_FORTRAN := $(shell $(RUSTICOL_CONFIG) --fortran-source)

.PHONY: all
all: runtime_cpp runtime_fortran

runtime_cpp: runtime.cpp
	$(CXX) $(CXXFLAGS) -std=c++17 $< $(RUSTICOL_CFLAGS) \
		$(RUSTICOL_LIBS) -o $@

runtime_fortran: runtime.f90
	$(FC) $(FFLAGS) -std=f2008 "$(RUSTICOL_FORTRAN)" $< \
		$(RUSTICOL_LIBS) -o $@
