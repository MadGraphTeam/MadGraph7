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
#include "coloramps.h"

namespace mg5amcCpu
{
  using namespace ProcessData;
  using namespace ProcessTables;
  using Parameters_dependentCouplings::ndcoup;   // #couplings that vary event by event (depend on running alphas QCD)
  using Parameters_independentCouplings::nicoup; // #couplings that are fixed for all events (do not depend on running alphas QCD)

  // The number of SIMD vectors of events processed by calculate_jamps
  constexpr int nParity = 1;

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
    memcpy( cIPF_partner1, tIPF_partner1, nMF * nIPF * sizeof( int ) );
    memcpy( cIPF_partner2, tIPF_partner2, nMF * nIPF * sizeof( int ) );
    memcpy( cIPF_value, tIPF_value, nMF * nIPF * sizeof( cxtype ) );
  }

  void setBsmIndepParam( const double* values, int n )
  {
    if( n > 0 ) memcpy( bsmIndepParam, values, n * sizeof( double ) );
  }

  //--------------------------------------------------------------------------

  __device__ INLINE unsigned int
  getChannelId( const unsigned int* allChannelIds, const int ievt00, bool sanityCheckMixedPrecision = true )
  {
    unsigned int channelId = 0; // disable multichannel single-diagram enhancement unless allChannelIds != nullptr
    using CID_ACCESS = HostAccessChannelIds; // non-trivial access: buffer includes all events
    // SCALAR channelId for the whole SIMD neppV2 event page (C++), i.e. one or two neppV event page(s)
    // The cudacpp implementation ASSUMES (and checks! #898) that all channelIds are the same in a neppV2 SIMD event page
    // **NB! in "mixed" precision, using SIMD, calculate_wavefunctions computes MEs for TWO neppV pages with a single channelId! #924
    if( allChannelIds != nullptr )
    {
      // First - and/or only - neppV page of channels (iParity=0 => ievt0 = ievt00 + 0 * neppV)
      const unsigned int* channelIds = CID_ACCESS::ieventAccessRecordConst( allChannelIds, ievt00 ); // fix bug #899/#911
      uint_sv channelIds_sv = CID_ACCESS::kernelAccessConst( channelIds );                           // fix #895 (compute this only once for all diagrams)
      // NB: channelIds_sv is a scalar in no-SIMD C++
      channelId = channelIds_sv;
      assert( channelId > 0 ); // SANITY CHECK: scalar channelId must be > 0 if multichannel is enabled (allChannelIds != nullptr)
      (void)sanityCheckMixedPrecision; // no second SIMD page to cross-check in no-SIMD C++
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

  // Evaluate QCD partial amplitudes jamps for this given helicity from Feynman diagrams.
  // This function processes a single event "page" or SIMD vector (or for two in "mixed"
  // precision mode, nParity=2). Accepts a SCALAR channelId because it is GUARANTEED that
  // all events in a SIMD vector have the same channelId #898.
  void
  calculate_jamps( int ihel,
                   const fptype* allmomenta,
                   const fptype* allcouplings,
                   const unsigned int* iflavorVec,
                   cxtype_sv* allJamp_sv,
                   bool storeChannelWeights,
                   fptype* allNumerators,
                   fptype* allDenominators,
                   fptype_sv* jamp2_sv,
                   const int ievt00 )
  {
    using M_ACCESS = HostAccessMomenta;
    using W_ACCESS = HostAccessWavefunctions;
    using A_ACCESS = HostAccessAmplitudes;
    using CD_ACCESS = HostAccessCouplings;
    using CI_ACCESS = HostAccessCouplingsFixed;
    using F_ACCESS = HostAccessIflavorVec;
    using NUM_ACCESS = HostAccessNumerators;
    using DEN_ACCESS = HostAccessDenominators;
    mgDebug( 0, __FUNCTION__ );

    fptype_sv pvec_sv[nwf][np4];
    cxtype_sv w_sv[nwf][nw6]; // particle wavefunctions within Feynman diagrams
    cxtype_sv amp_sv[1];      // invariant amplitude for one given Feynman diagram
    ALOHAOBJ aloha_obj[nwf];
    for( int iwf = 0; iwf < nwf; iwf++ ) aloha_obj[iwf] = ALOHAOBJ{ pvec_sv[iwf], w_sv[iwf] };
    fptype* amp_fp = reinterpret_cast<fptype*>( amp_sv );

    // jamp: sum (for one event or event page) of the invariant amplitudes for
    // all Feynman diagrams in a given color combination
    cxtype_sv jamp_sv[ncolor] = {}; // all zeros

    for( int iParity = 0; iParity < nParity; ++iParity )
    {
      const int ievt0 = ievt00 + iParity * neppV;

      constexpr size_t nxcoup = ndcoup + nIPC; // both dependent and independent couplings
      const fptype* allCOUPs[nxcoup];
      for( size_t idcoup = 0; idcoup < ndcoup; idcoup++ )
        allCOUPs[idcoup] = CD_ACCESS::idcoupAccessBufferConst( allcouplings, idcoup ); // dependent couplings, vary event-by-event
      for( size_t iicoup = 0; iicoup < nIPC; iicoup++ )
        allCOUPs[ndcoup + iicoup] = CI_ACCESS::iicoupAccessBufferConst( cIPC, iicoup ); // independent couplings, fixed for all events
      // C++ kernels take input/output buffers with momenta/MEs for one specific event (the first in the current event page)
      const fptype* momenta = M_ACCESS::ieventAccessRecordConst( allmomenta, ievt0 );
      const fptype* COUPs[nxcoup];
      for( size_t idcoup = 0; idcoup < ndcoup; idcoup++ )
        COUPs[idcoup] = CD_ACCESS::ieventAccessRecordConst( allCOUPs[idcoup], ievt0 ); // dependent couplings, vary event-by-event
      for( size_t iicoup = 0; iicoup < nIPC; iicoup++ )
        COUPs[ndcoup + iicoup] = allCOUPs[ndcoup + iicoup]; // independent couplings, fixed for all events
      fptype* numerators = NUM_ACCESS::ieventAccessRecord( allNumerators, ievt0 * ndiagrams );
      fptype* denominators = DEN_ACCESS::ieventAccessRecord( allDenominators, ievt0 );
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
      alignas( mgOnGpu::cppAlign ) fptype dpf_value[ndpfbuf]{};
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

      // Numerators and denominators for the current event (CUDA) or SIMD event page (C++)
      fptype_sv* numerators_sv = NUM_ACCESS::kernelAccessP( numerators );
      fptype_sv& denominators_sv = DEN_ACCESS::kernelAccess( denominators );
      // Scalar iflavor for the current event (constant across the SIMD vector)
      const unsigned int* iflavor_rec = F_ACCESS::ieventAccessRecordConst( iflavorVec, ievt0 );
      const uint_sv iflavor_sv = F_ACCESS::kernelAccessConst( iflavor_rec );
      const unsigned int iflavor = reinterpret_cast<const unsigned int*>(&iflavor_sv)[0];
#include "EvaluateDiagrams.inc"

      // *** COLOR CHOICE BELOW ***
      // Store the leading color flows for choice of color
      if( jamp2_sv ) // disable color choice if nullptr
      {
        for( int icol = 0; icol < ncolor; icol++ )
          jamp2_sv[ncolor * iParity + icol] += cxabs2( jamp_sv[icol] ); // may underflow #831
      }

      // *** PREPARE OUTPUT JAMPS ***
      // In C++, copy the local jamp to the output array passed as function argument
      for( int icol = 0; icol < ncolor; icol++ )
        allJamp_sv[iParity * ncolor + icol] = jamp_sv[icol];
    }
    // END LOOP ON IPARITY

    mgDebug( 1, __FUNCTION__ );
    return;
  }

  //--------------------------------------------------------------------------

  void
  sigmaKin_getGoodHel( const fptype* allmomenta,
                       const fptype* allcouplings,
                       const unsigned int* iflavorVec,
                       fptype* allMEs,
                       fptype* allNumerators,
                       fptype* allDenominators,
                       bool* isGoodHel,
                       const int nevt )
  {
    // Allocate arrays at build time to contain at least 16 events (or at least neppV events if neppV>16, e.g. in future VPUs)
    constexpr int maxtry0 = std::max( 16, neppV ); // 16, but at least neppV (otherwise the npagV loop does not even start)
    assert( nevt >= neppV );
    const int maxtry = std::min( maxtry0, nevt ); // 16, but at most nevt (avoid invalid memory access if nevt<maxtry0)
    const int npagV = maxtry / neppV;
    const int npagV2 = npagV; // loop on one SIMD page (neppV events) at a time
    // Per-flavor good-helicity union (merged flavors, e.g. PDG=81): a helicity
    // that vanishes for the sampled flavor may be non-zero for another merged
    // flavor and must not be dropped. Sample every flavor combination on the
    // same momenta and OR the result, so cGoodHel becomes the union over all
    // flavors (extra helicities simply contribute 0 for a given flavor at run
    // time, exactly as in the scalar standalone_cpp per-flavor good-hel filter).
    for( int ihel = 0; ihel < ncomb; ihel++ ) isGoodHel[ihel] = false;
    (void)iflavorVec; // flavor is forced below to scan every flavor combination
    unsigned int hgFlavorVec[maxtry0] = {}; // forced single-flavor index buffer
    for( int iflav = 0; iflav < nmaxflavor; ++iflav )
    {
    for( int i = 0; i < maxtry0; ++i ) hgFlavorVec[i] = (unsigned int)iflav;
    for( int ipagV2 = 0; ipagV2 < npagV2; ++ipagV2 )
    {
      const int ievt00 = ipagV2 * neppV; // loop on one SIMD page (neppV events) at a time
      for( int ihel = 0; ihel < ncomb; ihel++ )
      {
        // NEW IMPLEMENTATION OF GETGOODHEL (#630): RESET THE RUNNING SUM OVER HELICITIES TO 0 BEFORE ADDING A NEW HELICITY
        for( int ieppV = 0; ieppV < neppV; ++ieppV )
        {
          const int ievt = ievt00 + ieppV;
          allMEs[ievt] = 0;
        }
        constexpr fptype_sv* jamp2_sv = nullptr; // no need for color selection during helicity filtering
        cxtype_sv jamp_sv[ncolor] = {};  // all zeros
        calculate_jamps( ihel, allmomenta, allcouplings, hgFlavorVec, jamp_sv, false, allNumerators, allDenominators, jamp2_sv, ievt00 );
        color_sum_cpu( allMEs, jamp_sv, ievt00 );
        for( int ieppV = 0; ieppV < neppV; ++ieppV )
        {
          const int ievt = ievt00 + ieppV;
          if( allMEs[ievt] != 0 ) // NEW IMPLEMENTATION OF GETGOODHEL (#630): COMPARE EACH HELICITY CONTRIBUTION TO 0
          {
            isGoodHel[ihel] = true;
          }
        }
      }
    }
    } // end loop over flavor combinations (per-flavor good-helicity union)
  }

  //--------------------------------------------------------------------------

  int // output: nGoodHel (the number of good helicity combinations out of ncomb)
  sigmaKin_setGoodHel( const bool* isGoodHel ) // input: isGoodHel[ncomb] - host array
  {
    int nGoodHel = 0;
    int goodHel[ncomb] = { 0 };
    for( int ihel = 0; ihel < ncomb; ihel++ )
    {
      if( isGoodHel[ihel] )
      {
        goodHel[nGoodHel] = ihel;
        nGoodHel++;
      }
    }
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
  // Evaluate |M|^2, part independent of incoming flavour

  void
  sigmaKin( const fptype* allmomenta,
            const fptype* allcouplings,
            const unsigned int* iflavorVec,
            const fptype* allrndhel,
            const fptype* allrndcol,
            const unsigned int* allChannelIds,
            const fptype* allrnddiagram,
            fptype* allMEs,
            int* allselhel,
            int* allselcol,
            fptype* allNumerators,
            fptype* allDenominators,
            unsigned int* allDiagramIdsOut,
            bool mulChannelWeight,
            const int nevt )
  {
    mgDebugInitialise();

    // SANITY CHECKS for cudacpp code generation (see issues #272 and #343 and PRs #619, #626, #360, #396 and #754)
    {
      // nprocesses == 2 may happen for "mirror processes" such as P0_uux_ttx within pp_tt012j (see PR #754)
      static_assert( nproc == 1 || nproc == 2, "Assume nprocesses == 1 or 2" );
      static_assert( proc_id == 1, "Assume process_id == 1" );
    }

    using E_ACCESS = HostAccessMatrixElements; // non-trivial access: buffer includes all events
    using NUM_ACCESS = HostAccessNumerators;   // non-trivial access: buffer includes all events
    using DEN_ACCESS = HostAccessDenominators; // non-trivial access: buffer includes all events

    // Reset the "matrix elements" - running sums of |M|^2 over helicities for the given event
    const int npagV = nevt / neppV;
    for( int ipagV = 0; ipagV < npagV; ++ipagV )
    {
      const int ievt0 = ipagV * neppV;
      fptype* MEs = E_ACCESS::ieventAccessRecord( allMEs, ievt0 );
      fptype_sv& MEs_sv = E_ACCESS::kernelAccess( MEs );
      MEs_sv = fptype_sv{ 0 };
      fptype* numerators = NUM_ACCESS::ieventAccessRecord( allNumerators, ievt0 * ndiagrams );
      fptype* denominators = DEN_ACCESS::ieventAccessRecord( allDenominators, ievt0 );
      fptype_sv* numerators_sv = NUM_ACCESS::kernelAccessP( numerators );
      fptype_sv& denominators_sv = DEN_ACCESS::kernelAccess( denominators );
      for( int i = 0; i < ndiagrams; ++i )
      {
        numerators_sv[i] = fptype_sv{ 0 };
      }
      denominators_sv = fptype_sv{ 0 };
    }

    // HELICITY LOOP: CALCULATE WAVEFUNCTIONS (using precomputed good helicities)
    const int npagV2 = npagV;            // loop on one SIMD page (neppV events) at a time
#ifdef _OPENMP
    // OMP multithreading #575 (NB: tested only with gcc11 so far)
#define _OMPLIST0 allcouplings, allMEs, allmomenta, allrndcol, allrndhel, allselcol, allselhel, cGoodHel, cNGoodHel, npagV2
#define _OMPLIST1 , allDenominators, allNumerators, allChannelIds, mgOnGpu::icolamp, mgOnGpu::channel2iconfig
#pragma omp parallel for default( none ) shared( _OMPLIST0 _OMPLIST1 )
#undef _OMPLIST0
#undef _OMPLIST1
#endif // _OPENMP
    for( int ipagV2 = 0; ipagV2 < npagV2; ++ipagV2 )
    {
      const int ievt00 = ipagV2 * neppV; // loop on one SIMD page (neppV events) at a time
      // Running sum of partial amplitudes squared for event by event color selection (#402)
      fptype_sv jamp2_sv[nParity * ncolor] = {};
      fptype_sv MEs_ighel[ncomb] = {};  // sum of MEs for all good helicities up to ighel (for the first - and/or only - neppV page)
      for( int ighel = 0; ighel < cNGoodHel; ighel++ )
      {
        const int ihel = cGoodHel[ighel];
        cxtype_sv jamp_sv[nParity * ncolor] = {}; // fixed nasty bug (omitting 'nParity' caused memory corruptions after calling calculate_jamps)
        bool storeChannelWeights = allChannelIds != nullptr || allrnddiagram != nullptr;
        calculate_jamps( ihel, allmomenta, allcouplings, iflavorVec, jamp_sv, storeChannelWeights, allNumerators, allDenominators, jamp2_sv, ievt00 );
        color_sum_cpu( allMEs, jamp_sv, ievt00 );
        MEs_ighel[ighel] = E_ACCESS::kernelAccess( E_ACCESS::ieventAccessRecord( allMEs, ievt00 ) );
      }
      // Event-by-event random choice of helicity #403
      for( int ieppV = 0; ieppV < neppV; ++ieppV )
      {
        const int ievt = ievt00 + ieppV;
        for( int ighel = 0; ighel < cNGoodHel; ighel++ )
        {
          const bool okhel = allrndhel[ievt] < ( MEs_ighel[ighel] / MEs_ighel[cNGoodHel - 1] );
          if( okhel )
          {
            const int ihelF = cGoodHel[ighel] + 1; // NB Fortran [1,ncomb], cudacpp [0,ncomb-1]
            allselhel[ievt] = ihelF;
            break;
          }
        }
      }
      const int vecsize = neppV;
      unsigned int channelIdVec[vecsize];
      if( allChannelIds != nullptr )
      {
        for( int ieppV = 0; ieppV < vecsize; ++ieppV )
        {
          const int ievt = ievt00 + ieppV;
          channelIdVec[ieppV] = allChannelIds[ievt];
        }
      }

      // Event-by-event random choice of channel
      if( allrnddiagram != nullptr )
      {
        for( int ieppV = 0; ieppV < vecsize; ++ieppV )
        {
          const int ievt = ievt00 + ieppV;
          fptype numerator_sum = 0., normalization = 0.;
          for( unsigned int ichan = 0; ichan < mgOnGpu::nchannels; ichan++ )
          {
            if( mgOnGpu::channel2iconfig[ichan] == -1 ) continue;
            normalization += allNumerators[ievt / neppV * neppV * ndiagrams +
                                           ichan * neppV + ieppV % neppV];
          }
          channelIdVec[ieppV] = mgOnGpu::nchannels;
          for( unsigned int ichan = 0; ichan < mgOnGpu::nchannels; ichan++ )
          {
            if( mgOnGpu::channel2iconfig[ichan] == -1 ) continue;
            numerator_sum += allNumerators[ievt / neppV * neppV * ndiagrams +
                                           ichan * neppV + ieppV % neppV];
            if( allrnddiagram[ievt] < numerator_sum / normalization )
            {
              channelIdVec[ieppV] = ichan + 1;
              break;
            }
          }
          allDiagramIdsOut[ievt] = channelIdVec[ieppV];
        }
      }

      // Event-by-event random choice of color #402
      if( allChannelIds != nullptr || allrnddiagram != nullptr ) // no event-by-event choice of color if channelId == 0 (fix FPE #783)
      {
        for( int ieppV = 0; ieppV < vecsize; ++ieppV )
        {
          unsigned int channelId = channelIdVec[ieppV];
          if( channelId > mgOnGpu::nchannels )
          {
            printf( "INTERNAL ERROR! Cannot choose an event-by-event random color for channelId=%d which is greater than nchannels=%d\n", channelId, mgOnGpu::nchannels );
            assert( channelId <= mgOnGpu::nchannels ); // SANITY CHECK #919 #910
          }
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
            if( mgOnGpu::icolamp[iconfig - 1][icolC] ) targetamp[icolC] +=
              jamp2_sv[icolC + ncolor * ( ieppV / neppV )];
          }
          const int ievt = ievt00 + ieppV;
          for( int icolC = 0; icolC < ncolor; icolC++ )
          {
            if( allrndcol[ievt] < ( targetamp[icolC] / targetamp[ncolor - 1] ) )
            {
              allselcol[ievt] = icolC + 1; // NB Fortran [1,ncolor], cudacpp [0,ncolor-1]
              break;
            }
          }
        }
      }
      else
      {
        for( int ieppV = 0; ieppV < neppV; ++ieppV )
        {
          const int ievt = ievt00 + ieppV;
          allselcol[ievt] = 0; // no color selected in Fortran range [1,ncolor] if channelId == 0 (see #931)
        }
      }
    }
    // *** END OF PART 1b - C++ (loop on event pages)

    // PART 2 - FINALISATION (after calculate_jamps)
    // Get the final |M|^2 as an average over helicities/colors of the running sum of |M|^2 over helicities for the given event
    for( int ipagV = 0; ipagV < npagV; ++ipagV )
    {
      const int ievt0 = ipagV * neppV;
      fptype* MEs = E_ACCESS::ieventAccessRecord( allMEs, ievt0 );
      fptype_sv& MEs_sv = E_ACCESS::kernelAccess( MEs );
      MEs_sv = MEs_sv * broken_symmetry_factor( iflavorVec[ievt0] ) / helcolDenominators[0];
      if( mulChannelWeight && allChannelIds != nullptr ) // fix segfault #892 (not 'channelIds[0] != 0')
      {
        const unsigned int channelId = getChannelId( allChannelIds, ievt0, false );
        fptype* numerators = NUM_ACCESS::ieventAccessRecord( allNumerators, ievt0 * ndiagrams );
        fptype* denominators = DEN_ACCESS::ieventAccessRecord( allDenominators, ievt0 );
        fptype_sv* numerators_sv = NUM_ACCESS::kernelAccessP( numerators );
        fptype_sv& denominators_sv = DEN_ACCESS::kernelAccess( denominators );
        MEs_sv *= numerators_sv[channelId - 1] / denominators_sv;
      }
    }
    mgDebugFinalise();
  }
}
