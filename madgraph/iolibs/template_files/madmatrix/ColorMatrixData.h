// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Integrated with the MadGraph7 project in Feb 2026.
//
// Process-specific color matrix, generated once per subprocess. Kept as a
// header (not a .cc) so the backend-owned color_sum_cpu/color_sum_gpu
// (backend/{cpu,simd,gpu}/color_sum.cc) can #include it and still constexpr-
// evaluate the normalized color matrix at compile time.

#ifndef COLORMATRIXDATA_H
#define COLORMATRIXDATA_H 1

#include "mgOnGpuConfig.h"
#include "ProcessData.h"

namespace ColorMatrixData
{
  constexpr int ncolor = ProcessData::ncolor;

%(color_matrix_lines)s
}

#endif // COLORMATRIXDATA_H
