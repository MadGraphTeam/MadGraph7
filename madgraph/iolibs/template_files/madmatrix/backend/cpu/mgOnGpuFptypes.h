// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Jan 2022) for the MG5aMC CUDACPP plugin.
// Further modified by: J. Teig, A. Valassi (2022-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MGONGPUFPTYPES_H
#define MGONGPUFPTYPES_H 1

#include "mgOnGpuConfig.h"

#include <algorithm>
#include <cmath>

// NB: namespaces mg5amcGpu and mg5amcCpu includes types which are defined in different ways for CPU and GPU builds (see #318 and #725)
namespace mg5amcCpu
{
  //==========================================================================


  //==========================================================================


  //------------------------------
  // Floating point types - C++
  //------------------------------

  inline const fptype&
  fpmax( const fptype& a, const fptype& b )
  {
    return std::max( a, b );
  }

  inline const fptype&
  fpmin( const fptype& a, const fptype& b )
  {
    return std::min( a, b );
  }

  inline fptype
  fpsqrt( const fptype& f )
  {
    return std::sqrt( f );
  }


  //==========================================================================

} // end namespace mg5amcGpu/mg5amcCpu

#endif // MGONGPUFPTYPES_H
