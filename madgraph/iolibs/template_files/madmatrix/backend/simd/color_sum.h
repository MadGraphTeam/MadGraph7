// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Sep 2025) for the MG5aMC CUDACPP plugin.
// Further modified by: A. Valassi (2025).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef COLOR_SUM_H
#define COLOR_SUM_H 1

#include "mgOnGpuConfig.h"

#include "mgOnGpuVectors.h"

#include "ProcessData.h"

namespace madmatrix
{
  //--------------------------------------------------------------------------

  // No-op on cpu/simd: the normalized color matrix is already a compile-time
  // constexpr there (see color_sum.cc); only gpu needs a runtime push to device memory.
  inline void createNormalizedColorMatrix() {}

  //--------------------------------------------------------------------------


  //--------------------------------------------------------------------------

  void
  color_sum_cpu( fptype* allMEs,              // output: allMEs[nevt], add |M|^2 for one specific helicity
                 const cxtype_sv* allJamp_sv, // input: jamp_sv[ncolor] (float/double) or jamp_sv[2*ncolor] (mixed) for one specific helicity
                 const int ievt0 );           // input: first event number in current C++ event page (for CUDA, ievt depends on threadid)

  //--------------------------------------------------------------------------


  //--------------------------------------------------------------------------


  //--------------------------------------------------------------------------
}

#endif // COLOR_SUM_H
