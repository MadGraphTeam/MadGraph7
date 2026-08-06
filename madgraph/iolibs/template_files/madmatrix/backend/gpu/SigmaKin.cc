// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Integrated with the MadGraph7 project in Feb 2026.
//
// Backend-owned driver: sigmaKin, calculate_jamps, good-helicity filtering.
// The diagram/vertex-call sequence (helas_calls) is process-specific and
// lives in the P1-generated EvaluateDiagrams.inc, #include'd below.

#include "SigmaKin.h"

#include "CPPProcess.h" // ProcessData.h, Parameters.h, HelAmps_<model>.h transitively
#include "ProcessTables.h"

#include "GpuRuntime.h"
#include "MemoryAccessAmplitudes.h"
#include "MemoryAccessChannelIds.h"
#include "MemoryAccessCouplings.h"
#include "MemoryAccessCouplingsFixed.h"
#include "MemoryAccessDenominators.h"
#include "MemoryAccessGs.h"
#include "MemoryAccessIflavorVec.h"
#include "MemoryAccessMatrixElements.h"
#include "MemoryAccessMomenta.h"
#include "MemoryAccessNumerators.h"
#include "MemoryAccessWavefunctions.h"
#include "color_sum.h"
#include "ColorData.h"

namespace mg5amcGpu
{
  using namespace ProcessData;
  using namespace ProcessTables;
  using Parameters_dependentCouplings::ndcoup;   // #couplings that vary event by event (depend on running alphas QCD)
  using Parameters_independentCouplings::nicoup; // #couplings that are fixed for all events (do not depend on running alphas QCD)
  constexpr int nParity = 1; // CUDA/HIP process one event per thread, no host-side SIMD paging

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
    gpuMemcpyToSymbol( cIPF_partner1, tIPF_partner1, nMF * nIPF * sizeof( int ) );
    gpuMemcpyToSymbol( cIPF_partner2, tIPF_partner2, nMF * nIPF * sizeof( int ) );
    gpuMemcpyToSymbol( cIPF_value, tIPF_value, nMF * nIPF * sizeof( cxtype ) );
  }

  void setBsmIndepParam( const double* values, int n )
  {
    if( n > 0 ) gpuMemcpyToSymbol( bsmIndepParam, values, n * sizeof( double ) );
  }

  //--------------------------------------------------------------------------

  class DeviceAccessJamp2
  {
  public:
    static __device__ inline fptype&
    kernelAccessIcol( fptype* buffer, const int icol )
    {
      const int nevt = gridDim.x * blockDim.x;
      const int ievt = blockDim.x * blockIdx.x + threadIdx.x;
      return buffer[icol * nevt + ievt];
    }
    static __device__ inline const fptype&
    kernelAccessIcolConst( const fptype* buffer, const int icol )
    {
      const int nevt = gridDim.x * blockDim.x;
      const int ievt = blockDim.x * blockIdx.x + threadIdx.x;
      return buffer[icol * nevt + ievt];
    }
  };

  //--------------------------------------------------------------------------

  __device__ INLINE unsigned int
  gpu_channelId( const unsigned int* allChannelIds )
  {
    unsigned int channelId = 0; // disable multichannel single-diagram enhancement unless allChannelIds != nullptr
    using CID_ACCESS = DeviceAccessChannelIds; // non-trivial access: buffer includes all events
    // SCALAR channelId for the current event (CUDA)
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

  __global__ void /* clang-format off */
  computeDependentCouplings( const fptype* allgs, // input: Gs[nevt]
                             fptype* allcouplings // output: couplings[nevt*ndcoup*2]
  ) /* clang-format on */
  {
    using G_ACCESS = DeviceAccessGs;
    using C_ACCESS = DeviceAccessCouplings;
    G2COUP<G_ACCESS, C_ACCESS>( allgs, allcouplings, bsmIndepParam );
  }

  //--------------------------------------------------------------------------

  // Evaluate QCD partial amplitudes jamps for this given helicity from Feynman diagrams.
  // This function processes a single event (one CUDA thread).
  __global__ void /* clang-format off */
  calculate_jamps( int ihel,
                   const fptype* allmomenta,          // input: momenta[nevt*npar*4]
                   const fptype* allcouplings,        // input: couplings[nevt*ndcoup*2]
                   const unsigned int* iflavorVec,    // input: indices of the flavor combinations
                   fptype* allJamps,                  // output: jamp[2*ncolor*nevt] buffer for one helicity _within a super-buffer for dcNGoodHel helicities_
                   bool storeChannelWeights,
                   fptype* allNumerators,             // input/output: multichannel numerators[nevt], add helicity ihel
                   fptype* allDenominators,           // input/output: multichannel denominators[nevt], add helicity ihel
                   fptype* colAllJamp2s,              // output: allJamp2s[ncolor][nevt] super-buffer, sum over col/hel (nullptr to disable)
                   const int nevt,                    // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
                   const bool processAllHelicities    // input: if true, use blockIdx.y to index helicities
                   ) /* clang-format on */
  {
    using M_ACCESS = DeviceAccessMomenta;         // non-trivial access: buffer includes all events
    using W_ACCESS = DeviceAccessWavefunctions;   // TRIVIAL ACCESS (no kernel splitting yet): buffer for one event
    using A_ACCESS = DeviceAccessAmplitudes;      // TRIVIAL ACCESS (no kernel splitting yet): buffer for one event
    using CD_ACCESS = DeviceAccessCouplings;      // non-trivial access (dependent couplings): buffer includes all events
    using CI_ACCESS = DeviceAccessCouplingsFixed; // TRIVIAL access (independent couplings): buffer for one event
    using F_ACCESS = DeviceAccessIflavorVec;      // non-trivial access: buffer includes all events
    using NUM_ACCESS = DeviceAccessNumerators;    // non-trivial access: buffer includes all events
    using DEN_ACCESS = DeviceAccessDenominators;  // non-trivial access: buffer includes all events
    mgDebug( 0, __FUNCTION__ );
    if( processAllHelicities )
    {
      int ighel = blockIdx.y;
      ihel = dcGoodHel[ighel];
      allJamps = allJamps + ighel * nevt;
      allNumerators = allNumerators + ighel * nevt * ndiagrams;
      allDenominators = allDenominators + ighel * nevt;
    }

    fptype_sv pvec_sv[nwf][np4];
    cxtype_sv w_sv[nwf][nw6]; // particle wavefunctions within Feynman diagrams
    cxtype_sv amp_sv[1];      // invariant amplitude for one given Feynman diagram
    ALOHAOBJ aloha_obj[nwf];
    for( int iwf = 0; iwf < nwf; iwf++ ) aloha_obj[iwf] = ALOHAOBJ{ pvec_sv[iwf], w_sv[iwf] };
    fptype* amp_fp = reinterpret_cast<fptype*>( amp_sv );

    // jamp: sum (for one event) of the invariant amplitudes for all Feynman diagrams in a given color combination
    cxtype_sv jamp_sv[ncolor] = {}; // all zeros

    for( int iParity = 0; iParity < nParity; ++iParity )
    {
      constexpr size_t nxcoup = ndcoup + nIPC; // both dependent and independent couplings
      const fptype* allCOUPs[nxcoup];
#ifdef __CUDACC__ // this must be __CUDACC__
#pragma nv_diagnostic push
#pragma nv_diag_suppress 186 // e.g. <<warning #186-D: pointless comparison of unsigned integer with zero>>
#endif
      for( size_t idcoup = 0; idcoup < ndcoup; idcoup++ )
        allCOUPs[idcoup] = CD_ACCESS::idcoupAccessBufferConst( allcouplings, idcoup ); // dependent couplings, vary event-by-event
      for( size_t iicoup = 0; iicoup < nIPC; iicoup++ )
        allCOUPs[ndcoup + iicoup] = CI_ACCESS::iicoupAccessBufferConst( cIPC, iicoup ); // independent couplings, fixed for all events
#ifdef __CUDACC__ // this must be __CUDACC__
#pragma nv_diagnostic pop
#endif
      // CUDA kernels take input/output buffers with momenta/MEs for all events
      const fptype* momenta = allmomenta;
      const fptype* COUPs[nxcoup];
      for( size_t ixcoup = 0; ixcoup < nxcoup; ixcoup++ ) COUPs[ixcoup] = allCOUPs[ixcoup];
      const int ievt = blockDim.x * blockIdx.x + threadIdx.x; // index of event (thread) in grid
      fptype* numerators = &allNumerators[ievt * ndiagrams];
      fptype* denominators = allDenominators;
      // Create an array of views over the Flavor Couplings
      FLV_COUPLING_ARRAY<nIPF, nMF> flvCOUPs{ cIPF_partner1, cIPF_partner2, cIPF_value };

      // Dependent (event-by-event, running-alphas) flavor couplings (Step 3): the per-flavor
      // values are NOT baked in (they run per event). Gather the current values of the
      // underlying dependent couplings for this event page into an AOSOA buffer dpf_value
      // (one nx2*neppC SIMD record per (coupling,flavor) slot, matching CD_ACCESS), then build
      // an ordinary value-based view over it. The flavor index is constant across a SIMD lane
      // (guaranteed by the phase-space integrator), so each lane gets its own running value
      // while sharing the same flavor selection. This is the direct analogue of Fortran's
      // FLV_xx%VAL(k)%P => GC_yyy(J). The vertex routines are instantiated with CD_ACCESS so
      // get_coupling_def reads dpf_value with the right per-flavor stride (CD_ACCESS::flv_stride).
      constexpr int ndpfbuf = ( nDPF > 0 ? nDPF * nMF * CD_ACCESS::flv_stride : 1 );
      fptype dpf_value[ndpfbuf]{};
      for( int idpf = 0; idpf < nDPF; idpf++ )
        for( int imf = 0; imf < nMF; imf++ )
        {
          const int idc = cDPF_idcoup[idpf * nMF + imf];
          if( idc >= 0 )
            CD_ACCESS::kernelAccess( dpf_value + ( idpf * nMF + imf ) * CD_ACCESS::flv_stride ) =
              CD_ACCESS::kernelAccessConst( COUPs[idc] );
        }
      FLV_COUPLING_ARRAY<nDPF, nMF, CD_ACCESS::flv_stride> flvCOUPs_dep{ cDPF_partner1, cDPF_partner2, dpf_value };

      // Reset color flows (reset jamp_sv) at the beginning of a new event or event page
      for( int i = 0; i < ncolor; i++ ) { jamp_sv[i] = cxzero_sv(); }

      // Numerators and denominators for the current event (CUDA)
      fptype_sv* numerators_sv = NUM_ACCESS::kernelAccessP( numerators );
      fptype_sv& denominators_sv = DEN_ACCESS::kernelAccess( denominators );
      // Scalar iflavor for the current event
      const unsigned int iflavor = F_ACCESS::kernelAccessConst( iflavorVec );
#include "EvaluateDiagrams.inc"

      // *** COLOR CHOICE BELOW ***
      // Store the leading color flows for choice of color
      assert( iParity == 0 ); // sanity check for J2_ACCESS
      using J2_ACCESS = DeviceAccessJamp2;
      if( colAllJamp2s ) // disable color choice if nullptr
      {
        for( int icol = 0; icol < ncolor; icol++ )
          // NB: atomicAdd is needed after moving to cuda streams with one helicity per stream!
          atomicAdd( &J2_ACCESS::kernelAccessIcol( colAllJamp2s, icol ), cxabs2( jamp_sv[icol] ) );
      }

      // *** PREPARE OUTPUT JAMPS ***
      // In CUDA, copy the local jamp to the output global-memory jamp
      constexpr int ihel0 = 0; // the allJamps buffer already points to a specific helicity _within a super-buffer for dcNGoodHel helicities_
      using J_ACCESS = DeviceAccessJamp;
      for( int icol = 0; icol < ncolor; icol++ )
        J_ACCESS::kernelAccessIcolIhelNhel( allJamps, icol, ihel0, dcNGoodHel ) = jamp_sv[icol];
    }
    // END LOOP ON IPARITY

    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  void /* clang-format off */
  sigmaKin_getGoodHel( const fptype* allmomenta,       // input: momenta[nevt*npar*4]
                       const fptype* allcouplings,     // input: couplings[nevt*ndcoup*2]
                       const unsigned int* iflavorVec, // input: index of the flavor combination
                       fptype* allMEs,                 // output: allMEs[nevt], |M|^2 final_avg_over_helicities
                       fptype* allNumerators,          // output: multichannel numerators[nevt], running_sum_over_helicities
                       fptype* allDenominators,        // output: multichannel denominators[nevt], running_sum_over_helicities
                       fptype_sv* allJamps,            // tmp: jamp[ncolor*2*nevt] _for one helicity_ (reused in the getGoodHel helicity loop)
                       bool* isGoodHel,                // output: isGoodHel[ncomb] - host array
                       const int nevt )                // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
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
      constexpr fptype_sv* allJamp2s = nullptr; // no need for color selection during helicity filtering
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
    int goodHel[ncomb] = { 0 }; // all zeros
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

  // Decay-aware identical-particle (broken-)symmetry factor, shared with the
  // Fortran / standalone_cpp exporters (_get_broken_symmetry_data). Two
  // entries contribute to the over-counting factor only when they have the
  // same top-level PID AND the same full decay/flavour block, so e.g. two Z
  // bosons decaying to different families are correctly distinguished.
  __device__ int
  broken_symmetry_factor( const int iflavor )
  {
    int pid_work[broken_sym_nentries];
    for( int i = 0; i < broken_sym_nentries; i++ )
      pid_work[i] = broken_sym_pid_list[i];

    int total_factor = 1;
    for( int icomp = 0; icomp < broken_sym_ncomponents; icomp++ )
    {
      int old_factor = broken_sym_component_old_factors[icomp];
      if( broken_sym_component_old_factors[icomp] > 1 )
      {
        for( int i = broken_sym_component_starts[icomp] - 1; i < broken_sym_component_ends[icomp]; i++ )
        {
          if( pid_work[i] == 0 )
            continue;
          int n_tot = 1;
          for( int j = i + 1; j < broken_sym_component_ends[icomp]; j++ )
          {
            if( pid_work[i] != pid_work[j] )
              continue;
            bool same_block = ( broken_sym_block_lengths[i] == broken_sym_block_lengths[j] );
            for( int k = 0; same_block && k < broken_sym_block_lengths[i]; k++ )
            {
              if( cFlavors[iflavor][broken_sym_block_starts[i] - 1 + k] != cFlavors[iflavor][broken_sym_block_starts[j] - 1 + k] )
                same_block = false;
            }
            if( same_block )
            {
              pid_work[j] = 0;
              n_tot = n_tot + 1;
              old_factor = old_factor / n_tot;
            }
          }
        }
      }
      total_factor = total_factor * old_factor;
    }
    return total_factor;
  }

  //--------------------------------------------------------------------------

  __global__ void
  normalise_output( fptype* allMEs,                    // output: allMEs[nevt], |M|^2 running_sum_over_helicities
                    const unsigned int* iflavorVec,
                    fptype* ghelAllNumerators,         // input/tmp: allNumerators super-buffer for nGoodHel <= ncomb individual helicities (index is ighel)
                    fptype* ghelAllDenominators,       // input/tmp: allNumerators super-buffer for nGoodHel <= ncomb individual helicities (index is ighel)
                    const unsigned int* allChannelIds, // input: multichannel channelIds[nevt] (1 to #diagrams); nullptr to disable SDE enhancement (fix #899/#911)
                    bool storeChannelWeights,           // if true, compute final multichannel weights
                    bool mulChannelWeight,             // if true, multiply matrix element by channel weight
                    const fptype globaldenom)
  {
    const int ievt = blockDim.x * blockIdx.x + threadIdx.x; // index of event (thread)
    allMEs[ievt] = allMEs[ievt] * broken_symmetry_factor(iflavorVec[ievt]) / globaldenom;
    const int nevt = gridDim.x * blockDim.x;
    if( storeChannelWeights ) // fix segfault #892 (not 'channelIds[0] != 0')
    {
      fptype* totAllNumerators = ghelAllNumerators;     // reuse "helicity #0" buffer to compute the total over all helicities
      fptype* totAllDenominators = ghelAllDenominators; // reuse "helicity #0" buffer to compute the total over all helicities
      for( int ighel = 1; ighel < dcNGoodHel; ighel++ ) // NB: the loop starts at ighel=1
      {
        fptype* hAllDenominators = ghelAllDenominators + ighel * nevt;
        totAllDenominators[ievt] += hAllDenominators[ievt];
        fptype* hAllNumerators = ghelAllNumerators + ( ievt + ighel * nevt ) * ndiagrams;
        fptype* firstNumerator = ghelAllNumerators + ievt * ndiagrams;
        for( int idiag = 0; idiag < ndiagrams; ++idiag )
        {
          firstNumerator[idiag] += hAllNumerators[idiag];
        }
      }
      if( mulChannelWeight )
      {
        unsigned int channelId = allChannelIds[ievt];
        allMEs[ievt] *= totAllNumerators[channelId - 1 + ievt * ndiagrams] / totAllDenominators[ievt];
      }
    }
    return;
  }

  //--------------------------------------------------------------------------

  __global__ void
  add_and_select_hel( int* allselhel,          // output: helicity selection[nevt]
                      const fptype* allrndhel, // input: random numbers[nevt] for helicity selection
                      fptype* ghelAllMEs,      // input/tmp: allMEs for nGoodHel <= ncomb individual/runningsum helicities (index is ighel)
                      fptype* allMEs,          // output: allMEs[nevt], final sum over helicities
                      const int nevt )         // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
  {
    const int ievt = blockDim.x * blockIdx.x + threadIdx.x; // index of event (thread)
    // Compute the sum of MEs over all good helicities (defer this after the helicity loop to avoid breaking streams parallelism)
    for( int ighel = 0; ighel < dcNGoodHel; ighel++ )
    {
      allMEs[ievt] += ghelAllMEs[ighel * nevt + ievt];
      ghelAllMEs[ighel * nevt + ievt] = allMEs[ievt]; // reuse the buffer to store the running sum for helicity selection
    }
    // Event-by-event random choice of helicity #403
    for( int ighel = 0; ighel < dcNGoodHel; ighel++ )
    {
      if( allrndhel[ievt] < ( ghelAllMEs[ighel * nevt + ievt] / allMEs[ievt] ) )
      {
        const int ihelF = dcGoodHel[ighel] + 1; // NB Fortran [1,ncomb], cudacpp [0,ncomb-1]
        allselhel[ievt] = ihelF;
        break;
      }
    }
    return;
  }

  //--------------------------------------------------------------------------

  __global__ void
  select_col_and_diag( int* allselcol,                    // output: color selection[nevt]
                       unsigned int* allDiagramIdsOut,    // output: sampled diagram ids
                       const fptype* allrndcol,           // input: random numbers[nevt] for color selection
                       const fptype* allrnddiagram,       // input: random numbers[nevt] for diagram selection
                       const unsigned int* allChannelIds, // input: multichannel channelIds[nevt] (1 to #diagrams); nullptr to disable SDE enhancement (fix #899/#911)
                       const fptype_sv* allJamp2s,        // input: jamp2[ncolor][nevt] for color choice (nullptr if disabled)
                       const fptype* allNumerators,       // input: all numerators
                       const fptype* allDenominators,     // input: all denominators
                       const int nevt )                   // input: #events (for cuda: nevt == ndim == gpublocks*gputhreads)
  {
    const int ievt = blockDim.x * blockIdx.x + threadIdx.x; // index of event (thread)
    // SCALAR channelId for the current event (CUDA)
    unsigned int channelId = gpu_channelId( allChannelIds );
    // Event-by-event random choice of channel
    if( allrnddiagram != nullptr )
    {
      fptype numerator_sum = 0., normalization = 0.;
      for( unsigned int ichan = 0; ichan < mgOnGpu::nchannels; ichan++ )
      {
        if( mgOnGpu::channel2iconfig[ichan] == -1 ) continue;
        normalization += allNumerators[ievt * ndiagrams + ichan];
      }
      channelId = mgOnGpu::nchannels;
      for( unsigned int ichan = 0; ichan < mgOnGpu::nchannels; ichan++ )
      {
        if( mgOnGpu::channel2iconfig[ichan] == -1 ) continue;
        numerator_sum += allNumerators[ievt * ndiagrams + ichan];
        if( allrnddiagram[ievt] < numerator_sum / normalization )
        {
          channelId = ichan + 1;
          break;
        }
      }
      allDiagramIdsOut[ievt] = channelId;
    }

    if( channelId != 0 ) // no event-by-event choice of color if channelId == 0 (fix FPE #783)
    {
      if( channelId > mgOnGpu::nchannels )
      {
        printf( "INTERNAL ERROR! Cannot choose an event-by-event random color for channelId=%d which is greater than nchannels=%d\n", channelId, mgOnGpu::nchannels );
        assert( channelId <= mgOnGpu::nchannels ); // SANITY CHECK #919 #910
      }
      // Determine the jamp2 for this event
      fptype_sv jamp2_sv[ncolor] = { 0 };
      assert( allJamp2s != nullptr ); // sanity check
      using J2_ACCESS = DeviceAccessJamp2;
      for( int icolC = 0; icolC < ncolor; icolC++ )
        jamp2_sv[icolC] = J2_ACCESS::kernelAccessIcolConst( allJamp2s, icolC );
      const int iconfig = mgOnGpu::channel2iconfig[channelId - 1]; // map N_diagrams to N_config <= N_diagrams configs (fix LHE color mismatch #856: see also #826, #852, #853)
      if( iconfig <= 0 )
      {
        printf( "INTERNAL ERROR! Cannot choose an event-by-event random color for channelId=%d which has no associated SDE iconfig\n", channelId );
        assert( iconfig > 0 ); // SANITY CHECK #917
      }
      else if( iconfig > (int)mgOnGpu::nconfigSDE )
      {
        printf( "INTERNAL ERROR! Cannot choose an event-by-event random color for channelId=%d (invalid SDE iconfig=%d\n > nconfig=%d)", channelId, iconfig, mgOnGpu::nconfigSDE );
        assert( iconfig <= (int)mgOnGpu::nconfigSDE ); // SANITY CHECK #917
      }
      fptype targetamp[ncolor] = { 0 };
      for( int icolC = 0; icolC < ncolor; icolC++ )
      {
        if( icolC == 0 )
          targetamp[icolC] = 0;
        else
          targetamp[icolC] = targetamp[icolC - 1];
        if( mgOnGpu::icolamp[iconfig - 1][icolC] ) targetamp[icolC] += jamp2_sv[icolC];
      }
      for( int icolC = 0; icolC < ncolor; icolC++ )
      {
        if( allrndcol[ievt] < ( targetamp[icolC] / targetamp[ncolor - 1] ) )
        {
          allselcol[ievt] = icolC + 1; // NB Fortran [1,ncolor], cudacpp [0,ncolor-1]
          break;
        }
      }
    }
    else
    {
      allselcol[ievt] = 0; // no color selected in Fortran range [1,ncolor] if channelId == 0 (see #931)
    }
    return;
  }

  //--------------------------------------------------------------------------
  // Evaluate |M|^2, part independent of incoming flavour

  void /* clang-format off */
  sigmaKin( const fptype* allmomenta,           // input: momenta[nevt*npar*4]
            const fptype* allcouplings,         // input: couplings[nevt*ndcoup*2]
            const unsigned int* iflavorVec,     // input: indices of the flavor combinations
            const fptype* allrndhel,            // input: random numbers[nevt] for helicity selection
            const fptype* allrndcol,            // input: random numbers[nevt] for color selection
            const unsigned int* allChannelIds,  // input: multichannel channelIds[nevt] (1 to #diagrams); nullptr to disable single-diagram enhancement (fix #899/#911)
            const fptype* allrnddiagram,        // input: random numbers[nevt] for channel sampling
            fptype* allMEs,                     // output: allMEs[nevt], |M|^2 final_avg_over_helicities
            int* allselhel,                     // output: helicity selection[nevt]
            int* allselcol,                     // output: helicity selection[nevt]
            fptype* colAllJamp2s,               // tmp: allJamp2s super-buffer for ncolor individual colors, running sum over colors and helicities
            fptype* ghelAllNumerators,          // tmp: allNumerators super-buffer for nGoodHel <= ncomb individual helicities (index is ighel)
            fptype* ghelAllDenominators,        // tmp: allDenominators super-buffer for nGoodHel <= ncomb individual helicities (index is ighel)
            unsigned int* allDiagramIdsOut,     // output: multichannel channelIds[nevt] (1 to #diagrams)
            bool mulChannelWeight,              // if true, multiply channel weight to ME output
            fptype* ghelAllMEs,                 // tmp: allMEs super-buffer for nGoodHel <= ncomb individual helicities (index is ighel)
            fptype* ghelAllJamps,               // tmp: allJamps super-buffer[2][ncol][nGoodHel][nevt] for nGoodHel <= ncomb individual helicities
            fptype2* ghelAllBlasTmp,            // tmp: allBlasTmp super-buffer for nGoodHel <= ncomb individual helicities
            gpuBlasHandle_t* pBlasHandle,       // input: cuBLAS/hipBLAS handle
            gpuStream_t* ghelStreams,           // input: cuda streams (index is ighel: only the first nGoodHel <= ncomb are non-null)
            const bool async,                   // input: if true, run everything asynchronously in first stream in ghelStreams
            const int gpublocks,                // input: cuda gpublocks
            const int gputhreads                // input: cuda gputhreads
            ) /* clang-format on */
  {
    mgDebugInitialise();

    // SANITY CHECKS for cudacpp code generation (see issues #272 and #343 and PRs #619, #626, #360, #396 and #754)
    {
      // nprocesses == 2 may happen for "mirror processes" such as P0_uux_ttx within pp_tt012j (see PR #754)
      static_assert( nproc == 1 || nproc == 2, "Assume nprocesses == 1 or 2" );
      static_assert( proc_id == 1, "Assume process_id == 1" );
    }

    const int nevt = gpublocks * gputhreads;
    gpuMemset( allMEs, 0, nevt * sizeof( fptype ) );
    gpuMemset( ghelAllJamps, 0, cNGoodHel * ncolor * mgOnGpu::nx2 * nevt * sizeof( fptype ) );
    gpuMemset( colAllJamp2s, 0, ncolor * nevt * sizeof( fptype ) );
    gpuMemset( ghelAllNumerators, 0, cNGoodHel * ndiagrams * nevt * sizeof( fptype ) );
    gpuMemset( ghelAllDenominators, 0, cNGoodHel * nevt * sizeof( fptype ) );
    gpuMemset( ghelAllMEs, 0, cNGoodHel * nevt * sizeof( fptype ) );

    // *** HELICITY LOOP: CALCULATE WAVEFUNCTIONS (one event per GPU thread) ***
    // Use CUDA/HIP streams to process different helicities in parallel (one good helicity per stream)
    // (1) First, within each helicity stream, compute the QCD partial amplitudes jamp's for each helicity
    // In multichannel mode, also compute the running sums over helicities of numerators, denominators and squared jamp2s
    bool storeChannelWeights = allChannelIds != nullptr || allrnddiagram != nullptr;
    if( async )
    {
      gpuLaunchKernel2D( calculate_jamps, gpublocks, cNGoodHel, gputhreads, ghelStreams[0], 0, allmomenta, allcouplings, iflavorVec, ghelAllJamps, storeChannelWeights, ghelAllNumerators, ghelAllDenominators, colAllJamp2s, nevt, true );
      color_sum_gpu( ghelAllMEs, ghelAllJamps, ghelAllBlasTmp, pBlasHandle, ghelStreams, cNGoodHel, gpublocks, gputhreads, true );
    }
    else
    {
      for( int ighel = 0; ighel < cNGoodHel; ighel++ )
      {
        const int ihel = cGoodHel[ighel];
        fptype* hAllJamps = ghelAllJamps + ighel * nevt; // HACK: bypass DeviceAccessJamp (consistent with layout defined there)
        fptype* hAllNumerators = ghelAllNumerators + ighel * nevt * ndiagrams;
        fptype* hAllDenominators = ghelAllDenominators + ighel * nevt;
        gpuLaunchKernelStream( calculate_jamps, gpublocks, gputhreads, ghelStreams[ighel], ihel, allmomenta, allcouplings, iflavorVec, hAllJamps, storeChannelWeights, hAllNumerators, hAllDenominators, colAllJamp2s, nevt, false );
      }
      // (2) Then compute the ME for that helicity from the color sum of QCD partial amplitudes jamps
      color_sum_gpu( ghelAllMEs, ghelAllJamps, ghelAllBlasTmp, pBlasHandle, ghelStreams, cNGoodHel, gpublocks, gputhreads, false );
      checkGpu( gpuDeviceSynchronize() ); // do not start helicity/color selection until the loop over helicities has completed
      // (3) Wait for all helicity streams to complete, then finally compute the ME sum over all helicities and choose one helicity and one color
    }
    // Event-by-event random choice of helicity #403 and ME sum over helicities (defer this after the helicity loop to avoid breaking streams parallelism)
    gpuLaunchKernel( add_and_select_hel, gpublocks, gputhreads, allselhel, allrndhel, ghelAllMEs, allMEs, gpublocks * gputhreads );

    gpuLaunchKernel( normalise_output, gpublocks, gputhreads, allMEs, iflavorVec, ghelAllNumerators, ghelAllDenominators, allChannelIds, storeChannelWeights, mulChannelWeight, helcolDenominators[0] );

    // Event-by-event random choice of color and diagram #402
    gpuLaunchKernel( select_col_and_diag, gpublocks, gputhreads, allselcol, allDiagramIdsOut, allrndcol, allrnddiagram, allChannelIds, colAllJamp2s, ghelAllNumerators, ghelAllDenominators, gpublocks * gputhreads );

    mgDebugFinalise();
  }
}
