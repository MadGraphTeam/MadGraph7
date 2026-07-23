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

  //  5 4 3 2 1 
  //  0 0 1 0 1 | means t channel ex. par. 13	
  class MemoryAccessInvariants
  {
  public:
	  
	

    static constexpr int npar = CPPProcess::npar;
    static constexpr int fullMask = ( 1 << npar ) - 1; // (1<<7)-1=0b10000000-1=0b0111111

    // canonical; 2nd always 0; s 00 | t 01
    static constexpr int maskDim = 1 << ( npar - 1 );

    static __host__ __device__ inline constexpr int
    slotOfMask( int mask )
    {
      // canonicalise: bit 1 must be clear
      //         10         flip the end npar-1 bits    	    
      if( mask & 2 ) mask = fullMask ^ mask; 

      //      keep bit 0      drop bit 1 and make space for bit 0 
      //		combine
      return ( mask & 1 ) | ( ( mask >> 2 ) << 1 ); 
    }

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
#ifdef MGONGPU_INVP2_DEBUG
      if( s_invUsed != nullptr ) s_invUsed[mask] = true;
#endif
      return record[mask];
    }

#ifdef MGONGPU_INVP2_DEBUG
    // usage debug bookkeeping
    static inline thread_local bool* s_invUsed = nullptr;
#endif
  };

  //----------------------------------------------------------------------------

} // end namespace mg5amcGpu/mg5amcCpu

#endif // MemoryAccessInvariants_H
