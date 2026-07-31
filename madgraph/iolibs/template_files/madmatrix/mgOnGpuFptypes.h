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
#ifdef MGONGPUCPP_GPUIMPL // cuda
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
  //==========================================================================

#ifdef MGONGPUCPP_GPUIMPL // cuda

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

  template<typename FP>
  inline __host__ __device__ const FP&
  fpmax( const FP& a, const FP& b )
  {
    return ( ( b < a ) ? a : b );
  }

  template<typename FP>
  inline __host__ __device__ const FP&
  fpmin( const FP& a, const FP& b )
  {
    return ( ( a < b ) ? a : b );
  }

  template<typename FP>
  inline __host__ __device__ FP
  fpsqrt( const FP& f )
  {
#if defined MGONGPU_FPTYPE_FLOAT
    // See https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html
    return sqrtf( f );
#else
    // See https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html
    return sqrt( f );
#endif
  }

#endif // #ifdef MGONGPUCPP_GPUIMPL

  //==========================================================================

#ifndef MGONGPUCPP_GPUIMPL

  //------------------------------
  // Floating point types - C++
  //------------------------------

  template<typename FP>
  inline const FP&
  fpmax( const FP& a, const FP& b )
  {
    return std::max( a, b );
  }

  template<typename FP>
  inline const FP&
  fpmin( const FP& a, const FP& b )
  {
    return std::min( a, b );
  }

  template<typename FP>
  inline FP
  fpsqrt( const FP& f )
  {
    return std::sqrt( f );
  }

#endif // #ifndef MGONGPUCPP_GPUIMPL

  //==========================================================================

} // end namespace mg5amcGpu/mg5amcCpu

#endif // MGONGPUFPTYPES_H
