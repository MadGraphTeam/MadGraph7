// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created for the MadGraph7 madmatrix propagator-invariant (FIXP2) feature.

#ifndef MemoryAccessInvariants_H
#define MemoryAccessInvariants_H 1

#include "mgOnGpuConfig.h"

#include "CPPProcess.h" // for CPPProcess::npar

// NB: namespaces mg5amcGpu and mg5amcCpu include types defined differently for CPU and GPU builds (see #318 and #725)
#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
  //----------------------------------------------------------------------------

  // Per-event dense table of sampled propagator p^2 indexed by external-leg bitmask (bit i =
  // external leg i). record[mask] == 0 => not sampled (recompute p^2 from momenta).
  class MemoryAccessInvariants
  {
  public:

    static constexpr int npar = CPPProcess::npar;
    static constexpr int maskDim = 1 << npar; // one slot per external-leg bitmask

    // Per-event record base (a dense table of maskDim entries).
    static __host__ __device__ inline const fptype_invmass*
    ieventAccessRecordConst( const fptype_invmass* buffer, const int ievt )
    {
      return buffer + ievt * maskDim;
    }

    // p^2 of the propagator built from external legs `mask` (0 if not sampled).
    static __host__ __device__ inline fptype_invmass
    kernelAccessConst( const fptype_invmass* record, const int mask )
    {
      return record[mask];
    }
  };

  //----------------------------------------------------------------------------

} // end namespace mg5amcGpu/mg5amcCpu

#endif // MemoryAccessInvariants_H
