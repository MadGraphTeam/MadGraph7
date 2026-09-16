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
#include "MemoryAccessAmplitudes.h"
#include "MemoryAccessChannelIds.h"
#include "MemoryAccessCouplings.h"
#include "MemoryAccessCouplingsFixed.h"
#include "MemoryAccessGs.h"
#include "MemoryAccessIflavorVec.h"
#include "MemoryAccessMomenta.h"
#include "MemoryAccessNumerators.h"
#include "MemoryAccessWavefunctions.h"
#include "color_sum.h" // for DeviceAccessJamp

namespace mg5amcGpu
{
  using namespace ProcessData;
  using namespace ProcessTables;
  using Parameters_dependentCouplings::ndcoup;   // #couplings that vary event by event (depend on running alphas QCD)
  using Parameters_independentCouplings::nicoup; // #couplings that are fixed for all events (do not depend on running alphas QCD)

  // Per-color running sum of |jamp|^2 over helicities, for event-by-event color choice.
  class DeviceAccessJamp2
  {
  public:
    static __device__ inline fptype_amp&
    kernelAccessIcol( fptype_amp* buffer, const int icol )
    {
      const int nevt = gridDim.x * blockDim.x;
      const int ievt = blockDim.x * blockIdx.x + threadIdx.x;
      return buffer[icol * nevt + ievt];
    }
  };

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

  // Accumulate a multichannel numerator contribution in place. All good-helicity
  // blocks/streams for a given event race on the same numerator slot (the helicity
  // dimension has been removed to save memory), so an atomicAdd is mandatory.
#define NUM_ATOMIC_ADD( DST, VAL ) atomicAdd( &( DST ), VAL )

  // Evaluate QCD partial amplitudes jamps for this given helicity from Feynman diagrams.
  // Also compute running sums over helicities adding jamp2, numerator, denominator
  // (NB: this function no longer handles matrix elements as the color sum has now been
  // moved to a separate function/kernel). This is a kernel function: it processes a
  // single event (the CUDA thread), and takes a channelId array as input.
  __global__ void /* clang-format off */
  calculate_jamps( int ihel,
                   const fptype_momenta* allmomenta,   // input: momenta[nevt*npar*4]
                   const fptype* allcouplings,         // input: couplings[nevt*ndcoup*2]
                   const unsigned int* iflavorVec,     // input: indices of the flavor combinations
                   fptype_amp* allJamps,               // output: jamp[2*ncolor*nevt] buffer for one helicity _within a super-buffer for dcNGoodHel helicities_
                   bool storeChannelWeights,
                   fptype_amp* allNumerators,          // input/output: multichannel numerators[nevt], add helicity ihel
                   fptype_amp* allDenominators,        // input/output: multichannel denominators[nevt], add helicity ihel
                   fptype_amp* colAllJamp2s,           // output: allJamp2s[ncolor_flow][nevt] super-buffer, sum over col/hel (nullptr to disable)
                   const int nevt,                     // input: #events (nevt == ndim == gpublocks*gputhreads)
                   const bool processAllHelicities )   // input: if true, use blockIdx.y to index helicities
  /* clang-format on */
  {
    using M_ACCESS = DeviceAccessMomenta;         // non-trivial access: buffer includes all events
    using W_ACCESS = DeviceAccessWavefunctions;   // TRIVIAL ACCESS (no kernel splitting yet): buffer for one event
    using A_ACCESS = DeviceAccessAmplitudes;      // TRIVIAL ACCESS (no kernel splitting yet): buffer for one event
    using CD_ACCESS = DeviceAccessCouplings;      // non-trivial access (dependent couplings): buffer includes all events
    using CI_ACCESS = DeviceAccessCouplingsFixed; // TRIVIAL access (independent couplings): buffer for one event
    using F_ACCESS = DeviceAccessIflavorVec;      // non-trivial access: buffer includes all events
    using NUM_ACCESS = DeviceAccessNumerators;    // non-trivial access: buffer includes all events
    mgDebug( 0, __FUNCTION__ );

    if( processAllHelicities )
    {
      int ighel = blockIdx.y;
      ihel = dcGoodHel[ighel];
      allJamps = allJamps + ighel * nevt;
      // NB: the numerators buffer has NO helicity dimension anymore: all good-helicity blocks
      // for a given event accumulate in place into the same [nevt][ndiagrams] slot via atomicAdd.
      // The denominators are no longer accumulated here (derived as the sum of numerators later).
    }

    // Local TEMPORARY variables for a subset of Feynman diagrams in the given CUDA event
    // (ievt) [NB these variables are reused several times (and re-initialised each time)
    // within the same event]. Create memory for both momenta and wavefunctions separately,
    // and later wrap them in ALOHAOBJ.
    fptype_momenta_sv pvec_sv[nwf][np4];
    cxtype_amp_sv w_sv[nwf][nw6]; // particle wavefunctions within Feynman diagrams
    cxtype_amp_sv amp_sv[1];      // invariant amplitude for one given Feynman diagram
    ALOHAOBJ aloha_obj[nwf];
    for( int iwf = 0; iwf < nwf; iwf++ ) aloha_obj[iwf] = ALOHAOBJ{ pvec_sv[iwf], w_sv[iwf] };
    fptype_amp* amp_fp = reinterpret_cast<fptype_amp*>( amp_sv );

    // special temporary ALOHAOBJ to hold F/Vtmp values in the combined vertex functions
    // while using the FD gauge (harmless, unused, when the model doesn't need it)
    fptype_momenta_sv pvec_sv_tmp[1][np4];
    cxtype_amp_sv w_sv_tmp[1][nw6];
    ALOHAOBJ aloha_obj_tmp[1];
    aloha_obj_tmp[0] = ALOHAOBJ{ pvec_sv_tmp[0], w_sv_tmp[0] };
    cxtype_amp_sv amp_tmp_sv[1]; // to ensure proper aligment for vector instructions
    fptype_amp* amp_tmp_fp = reinterpret_cast<fptype_amp*>( amp_tmp_sv );

    // jamp: sum (for one event) of the invariant amplitudes for all Feynman diagrams in a
    // given color combination (NB: vector cxtype_v IS initialized to 0, but scalar cxtype
    // is NOT, if "= {}" is missing!)
    cxtype_amp_sv jamp_sv[ncolor] = {};
    // jampTmp: partial sums of amplitudes that several color flows share, so that they are
    // computed only once (see MadMatrixUFOHelasCallWriter.build_jamp_plan); no "= {}", each
    // one is assigned before it is ever read.
    cxtype_amp_sv jampTmp_sv[ProcessTables::nb_tmp_jamp > 0 ? ProcessTables::nb_tmp_jamp : 1];

    // === Calculate wavefunctions and amplitudes for all diagrams in all processes
#include "EvaluateDiagrams.inc"

    // *** COLOR CHOICE BELOW ***
    // Store the leading color flows for choice of color
    if( colAllJamp2s ) // disable color choice if nullptr
    {
      using J2_ACCESS = DeviceAccessJamp2;
      for( int icol = 0; icol < ncolor; icol++ )
        J2_ACCESS::kernelAccessIcol( colAllJamp2s, icol ) += cxabs2( jamp_sv[icol] ); // may underflow #831
    }

    // *** PREPARE OUTPUT JAMPS ***
    // allJamps already points at this helicity's slot in the dcNGoodHel super-buffer
    // (see processAllHelicities above), so this is nhel=1 from that slot's own view.
    {
      using J_ACCESS = DeviceAccessJamp;
      for( int icol = 0; icol < ncolor; icol++ )
        J_ACCESS::kernelAccessIcolIhelNhel( allJamps, icol, 0, 1 ) = jamp_sv[icol];
    }

    mgDebug( 1, __FUNCTION__ );
  }

#undef NUM_ATOMIC_ADD

  //--------------------------------------------------------------------------

  void /* clang-format off */
  sigmaKin_getGoodHel( const fptype_momenta* allmomenta, // input: momenta[nevt*npar*4]
                       const fptype* allcouplings,       // input: couplings[nevt*ndcoup*2]
                       const unsigned int* iflavorVec,   // input: indices of the flavor combinations
                       fptype* allMEs,                   // output: allMEs[nevt], |M|^2 final_avg_over_helicities
                       fptype_amp* allNumerators,        // output: multichannel numerators[nevt], running_sum_over_helicities
                       fptype_amp* allDenominators,      // output: multichannel denominators[nevt], running_sum_over_helicities
                       fptype_amp_sv* allJamps,          // tmp: jamp[ncolor*2*nevt] _for one helicity_ (reused in the getGoodHel helicity loop)
                       bool* isGoodHel,                  // output: isGoodHel[ncomb] - host array
                       const int nevt )                  // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
  { /* clang-format on */
    const int maxtry0 = 16;
    fptype hstMEs[maxtry0];
    const int maxtry = std::min( maxtry0, nevt ); // 16, but at most nevt (avoid invalid memory access if nevt<maxtry0)
    // Per-flavor good-helicity union (merged flavors, e.g. PDG=81): sample every
    // flavor combination on the same momenta and OR the result, so cGoodHel is
    // the union over all flavors (see the C++ branch below for the rationale).
    for( int ihel = 0; ihel < ncomb; ihel++ ) isGoodHel[ihel] = false;
    (void)iflavorVec; // flavor is forced below to scan every flavor combination
    unsigned int hstFlavorVec[maxtry0] = {};
    unsigned int* devFlavorVec = nullptr;
    gpuMalloc( (void**)&devFlavorVec, maxtry * sizeof( unsigned int ) );
    for( int iflav = 0; iflav < nmaxflavor; ++iflav )
    {
    for( int i = 0; i < maxtry; ++i ) hstFlavorVec[i] = (unsigned int)iflav;
    gpuMemcpy( devFlavorVec, hstFlavorVec, maxtry * sizeof( unsigned int ), gpuMemcpyHostToDevice );
    for( int ihel = 0; ihel < ncomb; ihel++ )
    {
      const int gpublocks = 1;
      const int gputhreads = maxtry;
      constexpr int nOneHel = 1; // use a jamp buffer for a single helicity
      gpuMemcpyToSymbol( dcNGoodHel, &nOneHel, sizeof( int ) );
      // NEW IMPLEMENTATION OF GETGOODHEL (#630): RESET THE RUNNING SUM OVER HELICITIES TO 0 BEFORE ADDING A NEW HELICITY
      gpuMemset( allMEs, 0, maxtry * sizeof( fptype ) );
      // NB: color_sum ADDS |M|^2 for one helicity to the running sum of |M|^2 over helicities for the given event(s)
      constexpr fptype_amp_sv* allJamp2s = nullptr; // no need for color selection during helicity filtering
      gpuLaunchKernel( calculate_jamps, gpublocks, gputhreads, ihel, allmomenta, allcouplings, devFlavorVec, allJamps, false, allNumerators, allDenominators, allJamp2s, gpublocks * gputhreads, false );
      gpuLaunchKernel( color_sum_kernel, gpublocks, gputhreads, allMEs, allJamps, nOneHel, 0 );
      gpuMemcpy( hstMEs, allMEs, maxtry * sizeof( fptype ), gpuMemcpyDeviceToHost );
      for( int ievt = 0; ievt < maxtry; ++ievt )
      {
        if( hstMEs[ievt] != 0 ) // NEW IMPLEMENTATION OF GETGOODHEL (#630): COMPARE EACH HELICITY CONTRIBUTION TO 0
        {
          isGoodHel[ihel] = true;
        }
      }
    }
    } // end loop over flavor combinations (per-flavor good-helicity union)
    gpuFree( devFlavorVec );
  }

  //--------------------------------------------------------------------------

  int                                          // output: nGoodHel (the number of good helicity combinations out of ncomb)
  sigmaKin_setGoodHel( const bool* isGoodHel ) // input: isGoodHel[ncomb] - host array (CUDA and C++)
  {
    int nGoodHel = 0;
    int goodHel[ncomb] = { 0 }; // all zeros https://en.cppreference.com/w/c/language/array_initialization#Notes
    for( int ihel = 0; ihel < ncomb; ihel++ )
    {
      if( isGoodHel[ihel] )
      {
        goodHel[nGoodHel] = ihel;
        nGoodHel++;
      }
    }
    gpuMemcpyToSymbol( dcNGoodHel, &nGoodHel, sizeof( int ) );
    gpuMemcpyToSymbol( dcGoodHel, goodHel, ncomb * sizeof( int ) );
    cNGoodHel = nGoodHel;
    for( int ihel = 0; ihel < ncomb; ihel++ ) cGoodHel[ihel] = goodHel[ihel];
    return nGoodHel;
  }

  //--------------------------------------------------------------------------

} // end namespace
