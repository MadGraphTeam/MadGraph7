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
// its setters, and the two helpers (gpu_channelId, computeDependentCouplings)
// that don't depend on calculate_jamps/sigmaKin, which land in a later slice.

#include "SigmaKin.h"

#include "CPPProcess.h" // ProcessData.h, Parameters.h, HelAmps_<model>.h transitively
#include "ProcessTables.h"

#include "GpuRuntime.h"
#include "MemoryAccessChannelIds.h"
#include "MemoryAccessCouplings.h"
#include "MemoryAccessGs.h"

namespace mg5amcGpu
{
  using namespace ProcessData;

  // Helicity/flavor tables and SM parameter/coupling storage, populated once
  // by CPPProcess's constructor/initProc via the setters below.
  __device__ __constant__ short cHel[ncomb][npar];
  __device__ __constant__ int dcNGoodHel;
  __device__ __constant__ int dcGoodHel[ncomb];
  __device__ __constant__ short cFlavors[nmaxflavor][npar];
  static int cNGoodHel;
  static int cGoodHel[ncomb];
  __device__ __constant__ fptype cIPD[nIPD > 0 ? nIPD : 1];
  __device__ __constant__ fptype cIPC[nIPC > 0 ? nIPC * 2 : 1];
  __device__ __constant__ int cIPF_partner1[ProcessTables::nMF * nIPF > 0 ? ProcessTables::nMF * nIPF : 1];
  __device__ __constant__ int cIPF_partner2[ProcessTables::nMF * nIPF > 0 ? ProcessTables::nMF * nIPF : 1];
  __device__ __constant__ fptype cIPF_value[ProcessTables::nMF * nIPF * 2 > 0 ? ProcessTables::nMF * nIPF * 2 : 1];
  __device__ __constant__ double bsmIndepParam[Parameters::nBsmIndepParam > 0 ? Parameters::nBsmIndepParam : 1];

  void setHelicitiesAndFlavors( const short* tHel, const short* tFlavors )
  {
    gpuMemcpyToSymbol( cHel, tHel, ncomb * npar * sizeof( short ) );
    gpuMemcpyToSymbol( cFlavors, tFlavors, nmaxflavor * npar * sizeof( short ) );
  }

  void setIndependentParams( const fptype* tIPD )
  {
    if( nIPD > 0 ) gpuMemcpyToSymbol( cIPD, tIPD, nIPD * sizeof( fptype ) );
  }

  void setIndependentCouplings( const cxtype* tIPC )
  {
    if( nIPC > 0 ) gpuMemcpyToSymbol( cIPC, tIPC, nIPC * sizeof( cxtype ) );
  }

  void setFlavorCouplings( const int* tIPF_partner1, const int* tIPF_partner2, const cxtype* tIPF_value )
  {
    if( nIPF == 0 ) return;
    gpuMemcpyToSymbol( cIPF_partner1, tIPF_partner1, ProcessTables::nMF * nIPF * sizeof( int ) );
    gpuMemcpyToSymbol( cIPF_partner2, tIPF_partner2, ProcessTables::nMF * nIPF * sizeof( int ) );
    gpuMemcpyToSymbol( cIPF_value, tIPF_value, ProcessTables::nMF * nIPF * sizeof( cxtype ) );
  }

  void setBsmIndepParam( const double* values, int n )
  {
    if( n > 0 ) gpuMemcpyToSymbol( bsmIndepParam, values, n * sizeof( double ) );
  }

  //--------------------------------------------------------------------------

  // SCALAR channelId for the current event (CUDA)
  __device__ INLINE unsigned int
  gpu_channelId( const unsigned int* allChannelIds )
  {
    unsigned int channelId = 0; // disable multichannel single-diagram enhancement unless allChannelIds != nullptr
    using CID_ACCESS = DeviceAccessChannelIds; // non-trivial access: buffer includes all events
    if( allChannelIds != nullptr )
    {
      const unsigned int* channelIds = allChannelIds;                            // fix #899 (distinguish channelIds and allChannelIds)
      const uint_sv channelIds_sv = CID_ACCESS::kernelAccessConst( channelIds ); // fix #895 (compute this only once for all diagrams)
      // NB: channelIds_sv is a scalar in CUDA
      channelId = channelIds_sv;
      assert( channelId > 0 ); // SANITY CHECK: scalar channelId must be > 0 if multichannel is enabled (allChannelIds != nullptr)
    }
    return channelId;
  }

  //--------------------------------------------------------------------------

  __global__ void
  computeDependentCouplings( const fptype* allgs, fptype* allcouplings )
  {
    using G_ACCESS = DeviceAccessGs;
    using C_ACCESS = DeviceAccessCouplings;
    G2COUP<G_ACCESS, C_ACCESS>( allgs, allcouplings, bsmIndepParam );
  }

  //--------------------------------------------------------------------------

} // end namespace
