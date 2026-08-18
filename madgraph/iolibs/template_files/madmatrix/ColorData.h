// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Integrated with the MadGraph7 project in Feb 2026.
//
// Process-specific color structure, generated once per subprocess: the
// normalized color matrix (for color_sum_cpu/color_sum_gpu) and the
// diagram/channel/config maps for multichannel color selection.

#ifndef COLORDATA_H
#define COLORDATA_H 1

#include "mgOnGpuConfig.h"
#include "ProcessData.h"

namespace ColorMatrixData
{
  constexpr int ncolor = ProcessData::ncolor;

%(color_matrix_lines)s
}

namespace mgOnGpu
{
  // Diagram: C-indexed [0,ndiagrams). Channel (channelId): F-indexed [1,nchannels],
  // not all diagrams have one (#919); channelId-1 indexes channel2iconfig. Config
  // (iconfig): F-indexed [1,nconfigSDE]; iconfig-1 indexes icolamp.
  constexpr unsigned int nchannels = %(nb_diag)i; // may be < ndiagrams, see #919
  static_assert( nchannels <= ProcessData::ndiagrams, "nchannels should be <= ndiagrams" ); // #910 #919

  // Map channel (C-indexed) to iconfig (F-indexed); -1 = no associated iconfig (#917)
  __device__ constexpr int channel2iconfig[%(nb_diag)i] = {
%(channelc2iconfig_lines)s
  };

  // #configs with an associated iconfig for single-diagram enhancement (#917)
  constexpr unsigned int nconfigSDE = %(nb_channel)s;

  // Map iconfig (C-indexed) to the mask of allowed colors
  __device__ constexpr bool icolamp[%(nb_channel)s][%(nb_color)s] = {
%(is_LC)s
  };
}

#endif // COLORDATA_H
