// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created for the MG5aMC standalone_mg7 (madmatrix) FIXP2 feature.

#ifndef MemoryAccessMij_H
#define MemoryAccessMij_H 1

#include "mgOnGpuConfig.h"

#include "CPPProcess.h"
#include "MemoryAccessMomenta.h" // reuse MemoryAccessMomentaBase::neppM (events page identically to momenta)
#include "MemoryAccessVectors.h"

// NB: namespaces mg5amcGpu and mg5amcCpu include types defined differently for CPU and GPU builds (see #318 and #725)
#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
  //----------------------------------------------------------------------------

  // Access to the per-event matrix of (signed) invariant mass^2 of external leg
  // pairs: m_ij[i][j] = the propagator virtuality p^2 to use for an offshell
  // current built from external legs i and j (the caller provides the correct
  // s-/t-channel sign). When a buffer/record pointer is null the offshell
  // propagator routines fall back to recomputing p^2 from the momenta.
  //
  // The buffer is an AOSOA[npagM][npar*npar][neppM] (same neppM paging as the
  // momenta AOSOA, so the two index events identically). NB: the SIMD bulk-load
  // path below mirrors MemoryAccessMomenta and is only used in C++ builds; the
  // GPU path currently never reads this buffer (it is passed as nullptr).
  class MemoryAccessMij
  {
  public:

    static constexpr int neppM = MemoryAccessMomentaBase::neppM;
    static constexpr int npar = CPPProcess::npar;
    static constexpr int nmij = npar * npar; // #components of the m_ij matrix

    // Locate the event record (page) base pointer for event ievt (component 0,0)
    static __host__ __device__ inline const fptype_invmass*
    ieventAccessRecordConst( const fptype_invmass* buffer,
                             const int ievt )
    {
      const int ipagM = ievt / neppM; // #event "M-page"
      const int ieppM = ievt % neppM; // #event in the current event M-page
      return &( buffer[ipagM * nmij * neppM + ieppM] ); // AOSOA[ipagM][0][ieppM]
    }

    // Access element (i,j) of the m_ij matrix within an event record (page),
    // returning the SIMD vector (C++) or scalar (GPU/no-SIMD) of values.
    // The buffer element type is fptype_invmass (the offshell-propagator virtuality
    // precision); in SIMD builds fptype_invmass==fptype (enforced in mgOnGpuConfig.h).
    static __host__ __device__ inline fptype_invmass_sv
    kernelAccessConst( const fptype_invmass* record,
                       const int i,
                       const int j )
    {
      const fptype_invmass& out = record[( i * npar + j ) * neppM]; // AOSOA[ipagM][i*npar+j][0]
#ifndef MGONGPU_CPPSIMD
      return out;
#else
      // neppM==neppV (default) => neppV contiguous fptype starting at &out form the SIMD vector
      if( (size_t)( &out ) % mgOnGpu::cppAlign == 0 )
        return mg5amcCpu::fptypevFromAlignedArray( out );   // SIMD bulk load, reinterpret_cast
      else
        return mg5amcCpu::fptypevFromUnalignedArray( out ); // SIMD bulk load, no reinterpret_cast
#endif
    }
  };

  //----------------------------------------------------------------------------

} // end namespace mg5amcGpu/mg5amcCpu

#endif // MemoryAccessMij_H
