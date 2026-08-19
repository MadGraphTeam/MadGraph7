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

//One namespace. Split ber backend.
namespace madmatrix
{
  //==========================================================================


  //------------------------------
  // Floating point types - Cuda
  //------------------------------

  /*
  inline __host__ __device__ fptype
  fpmax( const fptype& a, const fptype& b )
  {
    return max( a, b );
  }

  inline __host__ __device__ fptype
  fpmin( const fptype& a, const fptype& b )
  {
    return min( a, b );
  }
  */

  inline __host__ __device__ const fptype&
  fpmax( const fptype& a, const fptype& b )
  {
    return ( ( b < a ) ? a : b );
  }

  inline __host__ __device__ const fptype&
  fpmin( const fptype& a, const fptype& b )
  {
    return ( ( a < b ) ? a : b );
  }

  inline __host__ __device__ fptype
  fpsqrt( const fptype& f )
  {
#if defined MGONGPU_FPTYPE_FLOAT
    // See https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html
    return sqrtf( f );
#else
    // See https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html
    return sqrt( f );
#endif
  }


  //==========================================================================


  //==========================================================================

} // end namespace madmatrix

#endif // MGONGPUFPTYPES_H
