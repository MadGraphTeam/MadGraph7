// Copyright (C) 2010 The MadGraph5_aMC@NLO development team and contributors.
// Created by: J. Alwall (Oct 2010) for the MG5aMC CPP backend.
//==========================================================================
// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Modified originally by: O. Mattelaer (Nov 2020) for the MG5aMC CUDACPP plugin.
// Further modified by: S. Hageboeck, D. Massaro, O. Mattelaer, S. Roiser, J. Teig, A. Thete, A. Valassi (2020-2026).
// Integrated with the MadGraph7 project in Feb 2026.
//==========================================================================
//
// Standalone script for MadGraph7 standalone mode.
// Generates phase-space points with RAMBO and evaluates the matrix element
// through the UMAMI interface (umami.h).
//
// Two run modes:
//   * matrix (default): runs 8 events on every flavor combination and prints
//                       the first event's phase-space point + matrix element
//                       for each flavor.
//   * perf            : runs nblocks*nthreads*niter events on a single flavor
//                       and prints performance counters.
//
//==========================================================================

#include "mgOnGpuConfig.h"

#include "CPPProcess.h"
#include "GpuAbstraction.h"
#include "GpuRuntime.h"
#include "MemoryAccessMomenta.h"
#include "MemoryBuffers.h"
#include "RamboSamplingKernels.h"
#include "RandomNumberKernels.h"
#include "epoch_process_id.h"
#include "timermap.h"
#include "umami.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#define STRINGIFY( s ) #s
#define XSTRINGIFY( s ) STRINGIFY( s )
#define SEP79 79

namespace
{
#ifdef MGONGPUCPP_GPUIMPL
  using namespace mg5amcGpu;
#else
  using namespace mg5amcCpu;
#endif

  // Fixed physics inputs
  fptype kEnergy = 1500.;                  // Ecms = 1.5 TeV and changed for the matrix mode to 1TeV
  constexpr unsigned long long kSeed = 20200805ULL;  // reproducible RAMBO seed

  // Matrix-mode always runs 8 events on a single flavor index.
  constexpr unsigned int kMatrixBlocks = 1;
  constexpr unsigned int kMatrixThreads = 8;

  // Power of GeV of the matrix-element output; depends only on the number of external legs.
  constexpr int kMEGeVExponent = -( 2 * CPPProcess::npar - 8 );

  bool is_number( const char* s )
  {
    const char* t = s;
    while( *t != '\0' && isdigit( *t ) ) ++t;
    return (int)strlen( s ) == t - s;
  }

  enum Mode { MODE_MATRIX, MODE_PERF };

  enum RamboType { RAMBO_MASSIVE, RAMBO_MASSLESS };

  int usage( const char* argv0, int ret = 1 )
  {
    std::cout
      << "Usage:\n"
      << "  " << argv0 << " [matrix] [-v|--verbose]\n"
      << "  " << argv0 << " perf [-v|--verbose] [-f|--flavor <int>] [--rambo-massless]"
      << " [<#blocksPerGrid> <#threadsPerBlock>] <#iterations>\n"
      << "  " << argv0 << " -p [opts]   (legacy alias for `perf`)\n"
      << "\n"
      << "Subcommands:\n"
      << "  matrix (default)  Run " << kMatrixBlocks << " events on every flavor combination\n"
      << "                    and print the first event's phase-space point and\n"
      << "                    matrix element for each flavor.\n"
      << "                    With -v also prints backend/fptype/hardcodePARAM header.\n"
      << "  perf              Run #blocks*#threads events over #iterations iterations\n"
      << "                    on a single flavor index, then print performance counters.\n"
      << "                    Always prints inputs + backend/fptype header.\n"
      << "                    With -v also dumps every event's phase-space point and ME.\n"
      << "\n"
      << "perf-mode defaults if positional args are omitted:\n"
      << "  #blocksPerGrid = 64, #threadsPerBlock = 256, #iterations = 1.\n";
    return ret;
  }

  // AOSOA -> UMAMI SoA single-event helper. Layout reminder:
  //   AOSOA: aosoa[i_page * npar*4*neppM + ipar*4*neppM + ip4*neppM + i_vector]
  //   UMAMI: soa[ip4 * npar*nevt + ipar*nevt + ievt]
  __host__ __device__ inline void
  aosoa_to_umami_one( const fptype* aosoa,
                      double* soa,
                      std::size_t ievt,
                      std::size_t nevt )
  {
    constexpr int npar = CPPProcess::npar;
    for( int ipar = 0; ipar < npar; ++ipar )
    {
      for( int ip4 = 0; ip4 < 4; ++ip4 )
      {
        soa[(std::size_t)ip4 * npar * nevt + (std::size_t)ipar * nevt + ievt] =
          (double)MemoryAccessMomenta::ieventAccessIp4IparConst( aosoa, ievt, ip4, ipar );
      }
    }
  }

#ifdef MGONGPUCPP_GPUIMPL
  __global__ void
  aosoa_to_umami_kernel( const fptype* aosoa,
                         double* soa,
                         std::size_t nevt )
  {
    std::size_t ievt = blockDim.x * blockIdx.x + threadIdx.x;
    if( ievt >= nevt ) return;
    aosoa_to_umami_one( aosoa, soa, ievt, nevt );
  }
#endif

  const char* backend_label()
  {
#ifdef __CUDACC__
    return "CUDA";
#elif defined( __HIPCC__ )
    return "HIP";
#else
    return "CPP";
#endif
  }

  const char* fp_label()
  {
#if defined MGONGPU_FPTYPE_DOUBLE and defined MGONGPU_FPTYPE2_FLOAT
    return "MIXED";
#elif defined MGONGPU_FPTYPE_DOUBLE
    return "DOUBLE";
#elif defined MGONGPU_FPTYPE_FLOAT
    return "FLOAT";
#else
    return "UNKNOWN";
#endif
  }

  void print_run_header( std::ostream& os )
  {
    os << "Process                     = " << XSTRINGIFY( MG_EPOCH_PROCESS_ID ) << "_" << backend_label()
#ifdef MGONGPU_HARDCODE_PARAM
       << " [hardcodePARAM=1]" << std::endl
#else
       << " [hardcodePARAM=0]" << std::endl
#endif
       << "FP precision                = " << fp_label() << std::endl
       << "Random number generation    = COMMON RANDOM HOST" << std::endl;
  }

  void print_momenta_table( std::ostream& os, const fptype* aosoa, unsigned int ievt )
  {
    constexpr int npar = CPPProcess::npar;
    os << std::string( SEP79, '-' ) << std::endl
       << " n        E             px             py              pz" << std::endl;
    for( int ipar = 0; ipar < npar; ++ipar )
    {
      double E  = (double)MemoryAccessMomenta::ieventAccessIp4IparConst( aosoa, ievt, 0, ipar );
      double px = (double)MemoryAccessMomenta::ieventAccessIp4IparConst( aosoa, ievt, 1, ipar );
      double py = (double)MemoryAccessMomenta::ieventAccessIp4IparConst( aosoa, ievt, 2, ipar );
      double pz = (double)MemoryAccessMomenta::ieventAccessIp4IparConst( aosoa, ievt, 3, ipar );
      os << std::scientific << std::setprecision( 7 )
         << std::setw( 2 ) << ipar + 1
         << std::setw( 16 ) << E
         << std::setw( 16 ) << px
         << std::setw( 16 ) << py
         << std::setw( 16 ) << pz
         << std::endl
         << std::defaultfloat;
    }
    os << std::string( SEP79, '-' ) << std::endl;
  }

  // Run sigmaKin via UMAMI for `nevt` events and copy back the MEs.
  // Both the momenta (UMAMI SoA layout) and the per-event flavor buffer must be set
  // by the caller. On GPU the buffers are device pointers and `hstMEs` receives the
  // host-side copy; on CPU `umamiMEs` is the output buffer.
  bool run_umami(
    UmamiHandle handle,
    unsigned int nevt,
    mgOnGpu::TimerMap& timermap,
    double& wavetime,
#ifdef MGONGPUCPP_GPUIMPL
    const DeviceBufferBase<double>& devUmamiMomenta,
    const DeviceBufferBase<unsigned int>& devFlv,
    DeviceBufferBase<double>& devUmamiMEs,
    std::vector<double>& hstMEs
#else
    const std::vector<double>& umamiMomenta,
    const std::vector<unsigned int>& flvVec,
    std::vector<double>& umamiMEs
#endif
  )
  {
    constexpr unsigned int UmamiInKeyNum = 2;
    timermap.start( "3a SigmaKin" );
    UmamiInputKey in_keys[UmamiInKeyNum] = { UMAMI_IN_MOMENTA, UMAMI_IN_FLAVOR_INDEX };
    UmamiOutputKey out_keys[1] = { UMAMI_OUT_MATRIX_ELEMENT };
#ifdef MGONGPUCPP_GPUIMPL
    const void* inputs[UmamiInKeyNum] = { devUmamiMomenta.data(), devFlv.data() };
    void* outputs[1] = { devUmamiMEs.data() };
#else
    const void* inputs[UmamiInKeyNum] = { umamiMomenta.data(), flvVec.data() };
    void* outputs[1] = { umamiMEs.data() };
#endif
    UmamiStatus st = umami_matrix_element(
      handle, nevt, nevt, 0, UmamiInKeyNum, in_keys, inputs, 1, out_keys, outputs );
    wavetime += timermap.stop();
    if( st != UMAMI_SUCCESS )
    {
      std::cerr << "ERROR! umami_matrix_element failed (status=" << st << ")" << std::endl;
      return false;
    }

#ifdef MGONGPUCPP_GPUIMPL
    timermap.start( "3b CpDTHmes" );
    gpuMemcpy( hstMEs.data(), devUmamiMEs.data(), nevt * sizeof( double ), gpuMemcpyDeviceToHost );
    wavetime += timermap.stop();
#endif
    return true;
  }

  // --------------------------------------------------------------------------
  // matrix mode: same PS point fed to every flavor combination, print event 0.
  // --------------------------------------------------------------------------
  int run_matrix_mode( bool verbose )
  {
    constexpr unsigned int nevt = kMatrixBlocks * kMatrixThreads;
    const unsigned int nFlavors = CPPProcess::nmaxflavor;

    mgOnGpu::TimerMap timermap;

#ifdef MGONGPUCPP_GPUIMPL
    timermap.start( "00 GpuInit" );
    GpuRuntime gpuRuntime( false );

    PinnedHostBufferRndNumMomenta hstRndmom( nevt );
    PinnedHostBufferMomenta hstMomenta( nevt );
    PinnedHostBufferWeights hstWeights( nevt );
    DeviceBufferRndNumMomenta devRndmom( nevt );
    DeviceBufferMomenta devMomenta( nevt );
    DeviceBufferWeights devWeights( nevt );
    DeviceBufferBase<double> devUmamiMomenta( (std::size_t)4 * CPPProcess::npar * nevt );
    DeviceBufferBase<double> devUmamiMEs( nevt );
    DeviceBufferBase<unsigned int> devFlv( nevt );
    std::vector<unsigned int> flvVec( nevt );
    std::vector<double> hstUmamiMEs( nevt );
#else
    HostBufferRndNumMomenta hstRndmom( nevt );
    HostBufferMomenta hstMomenta( nevt );
    HostBufferWeights hstWeights( nevt );
    std::vector<double> umamiMomenta( (std::size_t)4 * CPPProcess::npar * nevt );
    std::vector<double> umamiMEs( nevt );
    std::vector<unsigned int> flvVec( nevt );
#endif

    UmamiHandle umami_handle = nullptr;
    if( umami_initialize( &umami_handle, "../../Cards/param_card.dat" ) != UMAMI_SUCCESS )
    {
      std::cerr << "ERROR! umami_initialize failed" << std::endl;
      return 2;
    }

    // Retrieve masses
    int npar_meta = 0;
    if( umami_get_meta( UMAMI_META_PARTICLE_COUNT, &npar_meta ) != UMAMI_SUCCESS || npar_meta != CPPProcess::npar )
    {
      std::cerr << "ERROR! umami_get_meta(UMAMI_META_PARTICLE_COUNT) failed" << std::endl;
      umami_free( umami_handle );
      return 2;
    }
    std::vector<double> massesD( npar_meta );
    if( umami_get_meta( UMAMI_META_MASSES, massesD.data() ) != UMAMI_SUCCESS )
    {
      std::cerr << "ERROR! umami_get_meta(UMAMI_META_MASSES) failed" << std::endl;
      umami_free( umami_handle );
      return 2;
    }
    const std::vector<fptype> masses( massesD.begin(), massesD.end() );

    // Always massive RAMBO
    std::unique_ptr<SamplingKernelBase> prsk(
      new RamboSamplingKernelHost( kEnergy, hstRndmom, masses, CPPProcess::npari, nevt, hstMomenta, hstWeights ) );
    prsk->getMomentaInitial();
    prsk->getMomentaFinal();

#ifdef MGONGPUCPP_GPUIMPL
    // Host only implementation now (copy)
    copyDeviceFromHost( devMomenta, hstMomenta );
    gpuLaunchKernel( aosoa_to_umami_kernel, kMatrixBlocks, kMatrixThreads, devMomenta.data(), devUmamiMomenta.data(), (std::size_t)nevt );
    checkGpu( gpuPeekAtLastError() );
#else
    for( std::size_t ievt = 0; ievt < nevt; ++ievt )
      aosoa_to_umami_one( hstMomenta.data(), umamiMomenta.data(), ievt, nevt );
#endif

    if( verbose )
    {
      std::cout << std::string( SEP79, '*' ) << std::endl;
      print_run_header( std::cout );
      std::cout << std::string( SEP79, '*' ) << std::endl;
    }

    std::cout << "Phase space point:" << std::endl;
    print_momenta_table( std::cout, hstMomenta.data(), 0 );

    for( unsigned int iflav = 0; iflav < nFlavors; ++iflav )
    {
      std::fill( flvVec.begin(), flvVec.end(), iflav );
#ifdef MGONGPUCPP_GPUIMPL
      gpuMemcpy( devFlv.data(), flvVec.data(), nevt * sizeof( unsigned int ), gpuMemcpyHostToDevice );
#endif
      double wavetime = 0;
      if( !run_umami( umami_handle, nevt, timermap, wavetime,
#ifdef MGONGPUCPP_GPUIMPL
                      devUmamiMomenta, devFlv, devUmamiMEs, hstUmamiMEs
#else
                      umamiMomenta, flvVec, umamiMEs
#endif
                      ) )
      {
        umami_free( umami_handle );
        return 3;
      }
#ifdef MGONGPUCPP_GPUIMPL
      const double* mes = hstUmamiMEs.data();
#else
      const double* mes = umamiMEs.data();
#endif

      std::cout << " PDG";
      for( int ipar = 0; ipar < CPPProcess::npar; ++ipar )
        std::cout << std::setw( 12 ) << CPPProcess::flavorPDG( iflav, ipar );
      std::cout << std::endl
                << " Matrix element = " << std::scientific << std::setprecision( 16 )
                << mes[0] << " GeV^" << kMEGeVExponent << std::endl
                << std::defaultfloat
                << std::string( SEP79, '-' ) << std::endl;
    }

    umami_free( umami_handle );
    return 0;
  }

  // --------------------------------------------------------------------------
  // perf mode: nblocks*nthreads events per iteration on a single flavor.
  // --------------------------------------------------------------------------
  int run_perf_mode( bool verbose,
                     unsigned int gpublocks,
                     unsigned int gputhreads,
                     unsigned int niter,
                     unsigned int flavorID,
                     RamboType ramboType )
  {
    const unsigned int nevt = gpublocks * gputhreads;

    mgOnGpu::TimerMap timermap;

#ifdef MGONGPUCPP_GPUIMPL
    timermap.start( "00 GpuInit" );
    GpuRuntime gpuRuntime( false );

    PinnedHostBufferRndNumMomenta hstRndmom( nevt );
    PinnedHostBufferMomenta hstMomenta( nevt );
    PinnedHostBufferWeights hstWeights( nevt );
    DeviceBufferRndNumMomenta devRndmom( nevt );
    DeviceBufferMomenta devMomenta( nevt );
    DeviceBufferWeights devWeights( nevt );
    DeviceBufferBase<double> devUmamiMomenta( (std::size_t)4 * CPPProcess::npar * nevt );
    DeviceBufferBase<double> devUmamiMEs( nevt );
    DeviceBufferBase<unsigned int> devFlv( nevt );
    std::vector<unsigned int> flvVec( nevt, flavorID );
    std::vector<double> hstUmamiMEs( nevt );
    // perf-mode runs a single flavor, so the device-side flavor buffer is filled once.
    gpuMemcpy( devFlv.data(), flvVec.data(), nevt * sizeof( unsigned int ), gpuMemcpyHostToDevice );
#else
    HostBufferRndNumMomenta hstRndmom( nevt );
    HostBufferMomenta hstMomenta( nevt );
    HostBufferWeights hstWeights( nevt );
    std::vector<double> umamiMomenta( (std::size_t)4 * CPPProcess::npar * nevt );
    std::vector<double> umamiMEs( nevt );
    std::vector<unsigned int> flvVec( nevt, flavorID );
#endif

    std::unique_ptr<RandomNumberKernelBase> prnk(
      new CommonRandomNumberKernel( hstRndmom ) );

    UmamiHandle umami_handle = nullptr;
    if( umami_initialize( &umami_handle, "../../Cards/param_card.dat" ) != UMAMI_SUCCESS )
    {
      std::cerr << "ERROR! umami_initialize failed" << std::endl;
      return 2;
    }

    // Retrieve masses
    std::vector<fptype> masses;
    if( ramboType != RAMBO_MASSLESS)
    {
      int npar_meta = 0;
      if( umami_get_meta( UMAMI_META_PARTICLE_COUNT, &npar_meta ) != UMAMI_SUCCESS || npar_meta != CPPProcess::npar )
      {
        std::cerr << "ERROR! umami_get_meta(UMAMI_META_PARTICLE_COUNT) failed" << std::endl;
        umami_free( umami_handle );
        return 2;
      }
      std::vector<double> massesD( npar_meta );
      if( umami_get_meta( UMAMI_META_MASSES, massesD.data() ) != UMAMI_SUCCESS )
      {
        std::cerr << "ERROR! umami_get_meta(UMAMI_META_MASSES) failed" << std::endl;
        umami_free( umami_handle );
        return 2;
      }
      masses.assign( massesD.begin(), massesD.end() );
    }

    std::unique_ptr<SamplingKernelBase> prsk;
    if( ramboType != RAMBO_MASSLESS )
    {
      // Massive host only (copy) 
      prsk.reset( new RamboSamplingKernelHost( kEnergy, hstRndmom, masses, CPPProcess::npari, nevt, hstMomenta, hstWeights ) );
    }
    else
    {
#ifdef MGONGPUCPP_GPUIMPL
      prsk.reset( new MasslessRamboSamplingKernelDevice( kEnergy, devRndmom, devMomenta, devWeights, gpublocks, gputhreads ) );
#else
      prsk.reset( new MasslessRamboSamplingKernelHost( kEnergy, hstRndmom, hstMomenta, hstWeights, nevt ) );
#endif
    }

    std::unique_ptr<double[]> genrtimes( new double[niter] );
    std::unique_ptr<double[]> rambtimes( new double[niter] );
    std::unique_ptr<double[]> wavetimes( new double[niter] );

    unsigned int nevtABN = 0;
    unsigned int nevtZERO = 0;
    double sumME = 0.;
    double sumMEsq = 0.;
    double minME = std::numeric_limits<double>::infinity();
    double maxME = -std::numeric_limits<double>::infinity();
    unsigned int nevtALL = 0;

    for( unsigned int iiter = 0; iiter < niter; ++iiter )
    {
      double genrtime = 0;
      timermap.start( "1a GenSeed " );
      prnk->seedGenerator( kSeed + iiter );
      genrtime += timermap.stop();
      timermap.start( "1b GenRnGen" );
      prnk->generateRnarray();
      genrtime += timermap.stop();
#ifdef MGONGPUCPP_GPUIMPL
      if( ramboType == RAMBO_MASSLESS )
      {
        timermap.start( "1c CpHTDrnd" );
        copyDeviceFromHost( devRndmom, hstRndmom );
        genrtime += timermap.stop();
      }
#endif

      double rambtime = 0;
      timermap.start( "2a RamboIni" );
      prsk->getMomentaInitial();
      rambtime += timermap.stop();
      timermap.start( "2b RamboFin" );
      prsk->getMomentaFinal();
      rambtime += timermap.stop();
#ifdef MGONGPUCPP_GPUIMPL
      // Massive host only (copy)
      if( ramboType != RAMBO_MASSLESS )
      {
        timermap.start( "2c CpHTDmom" );
        copyDeviceFromHost( devMomenta, hstMomenta );
        rambtime += timermap.stop();
      }
#endif

      timermap.start( "2d Aosoa2U " );
#ifdef MGONGPUCPP_GPUIMPL
      gpuLaunchKernel( aosoa_to_umami_kernel, gpublocks, gputhreads, devMomenta.data(), devUmamiMomenta.data(), (std::size_t)nevt );
      checkGpu( gpuPeekAtLastError() );
#else
      for( std::size_t ievt = 0; ievt < nevt; ++ievt )
        aosoa_to_umami_one( hstMomenta.data(), umamiMomenta.data(), ievt, nevt );
#endif
      rambtime += timermap.stop();

      double wavetime = 0;
      if( !run_umami( umami_handle, nevt, timermap, wavetime,
#ifdef MGONGPUCPP_GPUIMPL
                      devUmamiMomenta, devFlv, devUmamiMEs, hstUmamiMEs
#else
                      umamiMomenta, flvVec, umamiMEs
#endif
                      ) )
      {
        umami_free( umami_handle );
        return 3;
      }

#ifdef MGONGPUCPP_GPUIMPL
      if( verbose )
      {
        timermap.start( "3c CpDTHmom" );
        copyHostFromDevice( hstMomenta, devMomenta );
        wavetime += timermap.stop();
      }
      const double* mes = hstUmamiMEs.data();
#else
      const double* mes = umamiMEs.data();
#endif

      timermap.start( "4@ UpdtStat" );
      for( unsigned int ievt = 0; ievt < nevt; ++ievt )
      {
        double me = mes[ievt];
        ++nevtALL;
        if( !std::isfinite( me ) )
          ++nevtABN;
        else if( me == 0. )
          ++nevtZERO;
        sumME += me;
        sumMEsq += me * me;
        if( me < minME ) minME = me;
        if( me > maxME ) maxME = me;
      }

      genrtimes[iiter] = genrtime;
      rambtimes[iiter] = rambtime;
      wavetimes[iiter] = wavetime;

      if( verbose )
      {
        std::cout << std::string( SEP79, '*' ) << std::endl
                  << "Iteration #" << iiter + 1 << " of " << niter << std::endl;
        for( unsigned int ievt = 0; ievt < nevt; ++ievt )
        {
          std::cout << "Event #" << ievt + 1 << std::endl;
          print_momenta_table( std::cout, hstMomenta.data(), ievt );
          std::cout << " Matrix element = " << std::scientific << std::setprecision( 16 )
                    << mes[ievt] << " GeV^" << kMEGeVExponent << std::endl
                    << std::defaultfloat
                    << std::string( SEP79, '-' ) << std::endl;
        }
      }
    }

    double sumgtim = 0, sumrtim = 0, sumwtim = 0;
    double minwtim = wavetimes[0], maxwtim = wavetimes[0];
    for( unsigned int i = 0; i < niter; ++i )
    {
      sumgtim += genrtimes[i];
      sumrtim += rambtimes[i];
      sumwtim += wavetimes[i];
      minwtim = std::min( minwtim, wavetimes[i] );
      maxwtim = std::max( maxwtim, wavetimes[i] );
    }
    double meanwtim = sumwtim / niter;

    unsigned int nevtGood = nevtALL - nevtABN;
    double meanME = ( nevtGood > 0 ) ? sumME / nevtGood : 0.;
    double varME = ( nevtGood > 0 ) ? sumMEsq / nevtGood - meanME * meanME : 0.;
    double stdME = ( varME > 0 ) ? std::sqrt( varME ) : 0.;

    std::cout << std::string( SEP79, '*' ) << std::endl;
    print_run_header( std::cout );
    std::cout << "NumBlocksPerGrid            = " << gpublocks << std::endl
              << "NumThreadsPerBlock          = " << gputhreads << std::endl
              << "NumIterations               = " << niter << std::endl
              << "FlavorIndex                 = " << flavorID << " / " << CPPProcess::nmaxflavor << std::endl
              << std::string( SEP79, '-' ) << std::endl
              << "NaN/abnormal MEs            = " << nevtABN << std::endl
              << "Zero MEs                    = " << nevtZERO << std::endl
              << std::string( SEP79, '-' ) << std::endl
              << "NumberOfEntries             = " << niter << std::endl
              << std::scientific
              << "TotalTime[Rnd+Rmb+ME] (123) = ( " << sumgtim + sumrtim + sumwtim << " )  sec" << std::endl
              << "TotalTime[Rambo+ME]    (23) = ( " << sumrtim + sumwtim << " )  sec" << std::endl
              << "TotalTime[RndNumGen]    (1) = ( " << sumgtim << " )  sec" << std::endl
              << "TotalTime[Rambo]        (2) = ( " << sumrtim << " )  sec" << std::endl
              << "TotalTime[MatrixElems]  (3) = ( " << sumwtim << " )  sec" << std::endl
              << "MeanTimeInMatrixElems       = ( " << meanwtim << " )  sec" << std::endl
              << "[Min,Max]TimeInMatrixElems  = [ " << minwtim << " ,  " << maxwtim << " ]  sec" << std::endl
              << std::string( SEP79, '-' ) << std::endl
              << "TotalEventsComputed         = " << nevtALL << std::endl
              << "EvtsPerSec[Rnd+Rmb+ME](123) = ( " << nevtALL / ( sumgtim + sumrtim + sumwtim ) << " )  sec^-1" << std::endl
              << "EvtsPerSec[Rmb+ME]     (23) = ( " << nevtALL / ( sumrtim + sumwtim ) << " )  sec^-1" << std::endl
              << "EvtsPerSec[MatrixElems] (3) = ( " << nevtALL / sumwtim << " )  sec^-1" << std::endl
              << std::defaultfloat
              << std::string( SEP79, '*' ) << std::endl
              << "MeanMatrixElemValue         = ( " << meanME << " +- " << stdME / std::sqrt( (double)std::max( 1u, nevtGood ) )
              << " )  GeV^" << kMEGeVExponent << std::endl
              << "[Min,Max]MatrixElemValue    = [ " << minME << " ,  " << maxME << " ]  GeV^" << kMEGeVExponent << std::endl
              << std::string( SEP79, '*' ) << std::endl;
    timermap.dump();
    std::cout << std::string( SEP79, '*' ) << std::endl;

    umami_free( umami_handle );
    return 0;
  }
}

int main( int argc, char** argv )
{

  Mode mode = MODE_MATRIX;
  RamboType ramboType = RAMBO_MASSIVE; // default
  bool ramboTypeSet = false;
  bool verbose = false;
  unsigned int flavorID = 0;
  unsigned int gpublocks = 64;
  unsigned int gputhreads = 256;
  unsigned int niter = 1;
  unsigned int numvec[3] = { 0, 0, 0 };
  int nnum = 0;
  bool got_positional = false;

  // Optional leading subcommand (no leading dash).
  int firstArg = 1;
  if( firstArg < argc )
  {
    std::string a = argv[firstArg];
    if( a == "matrix" ) { mode = MODE_MATRIX; ++firstArg; }
    else if( a == "perf" ) { mode = MODE_PERF; ++firstArg; }
  }

  for( int argn = firstArg; argn < argc; ++argn )
  {
    std::string arg = argv[argn];
    if( arg == "--verbose" || arg == "-v" )
      verbose = true;
    else if( arg == "--performance" || arg == "-p" )
      mode = MODE_PERF; // legacy alias
    else if( ( arg == "--flavor" || arg == "-f" ) && argn + 1 < argc && is_number( argv[argn + 1] ) )
      flavorID = strtoul( argv[++argn], nullptr, 0 );
    else if( arg == "--rambo-massless"  )
    {
      std::string r = argv[++argn];
      ramboType = RAMBO_MASSLESS;
      ramboTypeSet = true;
    }
    else if( is_number( argv[argn] ) && nnum < 3 )
    {
      numvec[nnum++] = strtoul( argv[argn], nullptr, 0 );
      got_positional = true;
    }
    else
      return usage( argv[0] );
  }
//ENERGY CHANGE FOR THE MATRIX MODE
  if( mode == MODE_MATRIX ) kEnergy = 1000.;

  if( mode == MODE_MATRIX )
  {
    if( ramboType == RAMBO_MASSLESS )
    {
      std::cerr << "ERROR: matrix mode only supports the classic RAMBO (-r c)." << std::endl;
      return usage( argv[0] );
    }
    if( got_positional )
    {
      std::cerr << "WARNING: positional args are ignored in matrix mode "
                << "(dimensions are fixed at " << kMatrixBlocks << " " << kMatrixThreads << " 1)."
                << std::endl;
    }
    return run_matrix_mode( verbose );
  }

  // perf mode
  if( nnum == 3 )
  {
    gpublocks = numvec[0];
    gputhreads = numvec[1];
    niter = numvec[2];
  }
  else if( nnum == 1 )
  {
    niter = numvec[0];
  }
  else if( nnum != 0 )
  {
    return usage( argv[0] );
  }
  if( niter == 0 ) return usage( argv[0] );

  if( flavorID >= CPPProcess::nmaxflavor )
  {
    std::cerr << "ERROR: flavor index " << flavorID
              << " is out of range [0, " << CPPProcess::nmaxflavor << ")." << std::endl;
    return 1;
  }

  return run_perf_mode( verbose, gpublocks, gputhreads, niter, flavorID, ramboType );
}
