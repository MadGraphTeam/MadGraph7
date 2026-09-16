// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Integrated with the MadGraph7 project in Feb 2026.
//
// Backend-owned driver: sigmaKin, calculate_jamps, good-helicity filtering.
// The diagram/vertex-call sequence (helas_calls) is process-specific and
// lives in the P1-generated EvaluateDiagrams.inc, #include'd below.
//
// This is a first, foundational slice: the process-independent storage the
// rest of this file operates on (cHel/cFlavors/cIPD/cIPC/cIPF/bsmIndepParam),
// its setters, and the two helpers (getChannelId, computeDependentCouplings)
// that don't depend on calculate_jamps/sigmaKin, which land in a later slice.

#include "SigmaKin.h"

#include "CPPProcess.h" // ProcessData.h, Parameters.h, HelAmps_<model>.h transitively
#include "ProcessTables.h"

#include "GpuRuntime.h"
#include "MemoryAccessChannelIds.h"
#include "MemoryAccessCouplings.h"
#include "MemoryAccessGs.h"

namespace mg5amcCpu
{
  using namespace ProcessData;

  // Helicity/flavor tables and SM parameter/coupling storage, populated once
  // by CPPProcess's constructor/initProc via the setters below.
  static short cHel[ncomb][npar];
  static short cFlavors[nmaxflavor][npar];
  static int cNGoodHel;
  static int cGoodHel[ncomb];
  static fptype cIPD[nIPD > 0 ? nIPD : 1];
  static fptype cIPC[nIPC > 0 ? nIPC * 2 : 1];
  static int cIPF_partner1[ProcessTables::nMF * nIPF > 0 ? ProcessTables::nMF * nIPF : 1];
  static int cIPF_partner2[ProcessTables::nMF * nIPF > 0 ? ProcessTables::nMF * nIPF : 1];
  static fptype cIPF_value[ProcessTables::nMF * nIPF * 2 > 0 ? ProcessTables::nMF * nIPF * 2 : 1];
  static double bsmIndepParam[Parameters::nBsmIndepParam > 0 ? Parameters::nBsmIndepParam : 1];

  void setHelicitiesAndFlavors( const short* tHel, const short* tFlavors )
  {
    memcpy( cHel, tHel, ncomb * npar * sizeof( short ) );
    memcpy( cFlavors, tFlavors, nmaxflavor * npar * sizeof( short ) );
  }

  void setIndependentParams( const fptype* tIPD )
  {
    if( nIPD > 0 ) memcpy( cIPD, tIPD, nIPD * sizeof( fptype ) );
  }

  void setIndependentCouplings( const cxtype* tIPC )
  {
    if( nIPC > 0 ) memcpy( cIPC, tIPC, nIPC * sizeof( cxtype ) );
  }

  void setFlavorCouplings( const int* tIPF_partner1, const int* tIPF_partner2, const cxtype* tIPF_value )
  {
    if( nIPF == 0 ) return;
    memcpy( cIPF_partner1, tIPF_partner1, ProcessTables::nMF * nIPF * sizeof( int ) );
    memcpy( cIPF_partner2, tIPF_partner2, ProcessTables::nMF * nIPF * sizeof( int ) );
    memcpy( cIPF_value, tIPF_value, ProcessTables::nMF * nIPF * sizeof( cxtype ) );
  }

  void setBsmIndepParam( const double* values, int n )
  {
    if( n > 0 ) memcpy( bsmIndepParam, values, n * sizeof( double ) );
  }

  //--------------------------------------------------------------------------

  // SCALAR channelId for the whole SIMD neppV2 event page (C++), i.e. one or two neppV event page(s)
  // The cudacpp implementation ASSUMES (and checks! #898) that all channelIds are the same in a neppV2 SIMD event page
  // **NB! in "mixed" precision, using SIMD, calculate_wavefunctions computes MEs for TWO neppV pages with a single channelId! #924
  __device__ INLINE unsigned int
  getChannelId( const unsigned int* allChannelIds, const int ievt00, bool sanityCheckMixedPrecision = true )
  {
    unsigned int channelId = 0; // disable multichannel single-diagram enhancement unless allChannelIds != nullptr
    using CID_ACCESS = HostAccessChannelIds; // non-trivial access: buffer includes all events
    if( allChannelIds != nullptr )
    {
      // First - and/or only - neppV page of channels (iParity=0 => ievt0 = ievt00 + 0 * neppV)
      const unsigned int* channelIds = CID_ACCESS::ieventAccessRecordConst( allChannelIds, ievt00 ); // fix bug #899/#911
      uint_sv channelIds_sv = CID_ACCESS::kernelAccessConst( channelIds );                           // fix #895 (compute this only once for all diagrams)
#ifndef MGONGPU_CPPSIMD
      // NB: channelIds_sv is a scalar in no-SIMD C++
      channelId = channelIds_sv;
#else
      // NB: channelIds_sv is a vector in SIMD C++
      channelId = channelIds_sv[0];    // element[0]
      for( int i = 1; i < neppV; ++i ) // elements[1...neppV-1]
      {
        assert( channelId == channelIds_sv[i] ); // SANITY CHECK #898: check that all events in a SIMD vector have the same channelId
      }
#endif
      assert( channelId > 0 ); // SANITY CHECK: scalar channelId must be > 0 if multichannel is enabled (allChannelIds != nullptr)
      if( sanityCheckMixedPrecision )
      {
#if defined MGONGPU_CPPSIMD and defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
        // Second neppV page of channels (iParity=1 => ievt0 = ievt00 + 1 * neppV)
        const unsigned int* channelIds2 = CID_ACCESS::ieventAccessRecordConst( allChannelIds, ievt00 + neppV ); // fix bug #899/#911
        uint_v channelIds2_v = CID_ACCESS::kernelAccessConst( channelIds2 );                                    // fix #895 (compute this only once for all diagrams)
        // **NB! in "mixed" precision, using SIMD, calculate_wavefunctions computes MEs for TWO neppV pages with a single channelId! #924
        for( int i = 0; i < neppV; ++i )
        {
          assert( channelId == channelIds2_v[i] ); // SANITY CHECKS #898 #924: all events in the 2nd SIMD vector have the same channelId as that of the 1st SIMD vector
        }
#else
        (void)sanityCheckMixedPrecision; // no second SIMD page to cross-check outside mixed precision
#endif
      }
    }
    return channelId;
  }

  //--------------------------------------------------------------------------

  __global__ void
  computeDependentCouplings( const fptype* allgs, fptype* allcouplings, const int nevt )
  {
    using G_ACCESS = HostAccessGs;
    using C_ACCESS = HostAccessCouplings;
    for( int ipagV = 0; ipagV < nevt / neppV; ++ipagV )
    {
      const int ievt0 = ipagV * neppV;
      const fptype* gs = MemoryAccessGs::ieventAccessRecordConst( allgs, ievt0 );
      fptype* couplings = MemoryAccessCouplings::ieventAccessRecord( allcouplings, ievt0 );
      G2COUP<G_ACCESS, C_ACCESS>( gs, couplings, bsmIndepParam );
    }
  }

  //--------------------------------------------------------------------------

} // end namespace
