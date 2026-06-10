// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (May 2022) for the MG5aMC CUDACPP plugin.
// Further modified by: A. Valassi (2022-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MemoryAccessDenominators_H
#define MemoryAccessDenominators_H 1

#include "mgOnGpuConfig.h"

#include "MemoryAccessHelpers.h"
#include "MemoryAccessVectors.h"

// NB: namespaces mg5amcGpu and mg5amcCpu includes types which are defined in different ways for CPU and GPU builds (see #318 and #725)
#ifdef MGONGPUCPP_GPUIMPL
namespace mg5amcGpu
#else
namespace mg5amcCpu
#endif
{
  //----------------------------------------------------------------------------

  // A class describing the internal layout of memory buffers for the multichannel
  // denominators (the running sum over channels of the per-channel numerators). These
  // are amplitude-squared quantities, so they use the amplitude precision fptype_amp
  // (NB: this is unrelated to fptype_denom, which is the precision of the propagator
  // denominator p^2 - M^2 inside the vertex functions). Layout: plain ARRAY[nevt].
  class MemoryAccessDenominatorsBase //_ARRAYv1
  {
  private:

    friend class MemoryAccessHelper<MemoryAccessDenominatorsBase, fptype_amp>;
    friend class KernelAccessHelper<MemoryAccessDenominatorsBase, true, fptype_amp>;
    friend class KernelAccessHelper<MemoryAccessDenominatorsBase, false, fptype_amp>;

    // Locate an event record (output) in a memory buffer (input) from the given event number (input)
    // [Signature (non-const) ===> fptype_amp* ieventAccessRecord( fptype_amp* buffer, const int ievt ) <===]
    static __host__ __device__ inline fptype_amp*
    ieventAccessRecord( fptype_amp* buffer,
                        const int ievt )
    {
      return &( buffer[ievt] ); // ARRAY[nevt]
    }

    // Locate a field (output) of an event record (input) from the given field indexes (input)
    // [Signature (non-const) ===> fptype_amp& decodeRecord( fptype_amp* buffer ) <===]
    static __host__ __device__ inline fptype_amp&
    decodeRecord( fptype_amp* buffer )
    {
      constexpr int ievt = 0;
      return buffer[ievt]; // ARRAY[nevt]
    }
  };

  //----------------------------------------------------------------------------

  // A class providing access to memory buffers for a given event, based on explicit event numbers
  class MemoryAccessDenominators : public MemoryAccessDenominatorsBase
  {
  public:

    // [Signature (non-const) ===> fptype_amp* ieventAccessRecord( fptype_amp* buffer, const int ievt ) <===]
    static constexpr auto ieventAccessRecord = MemoryAccessHelper<MemoryAccessDenominatorsBase, fptype_amp>::ieventAccessRecord;

    // [Signature (const) ===> const fptype_amp* ieventAccessRecordConst( const fptype_amp* buffer, const int ievt ) <===]
    static constexpr auto ieventAccessRecordConst = MemoryAccessHelper<MemoryAccessDenominatorsBase, fptype_amp>::ieventAccessRecordConst;
  };

  //----------------------------------------------------------------------------

  // A class providing access to memory buffers for a given event, based on implicit kernel rules
  template<bool onDevice>
  class KernelAccessDenominators
  {
  public:

    // Expose selected functions from MemoryAccessDenominators
    static constexpr auto ieventAccessRecord = MemoryAccessDenominators::ieventAccessRecord;

    // [Signature (non-const, SCALAR) ===> fptype_amp& kernelAccess_s( fptype_amp* buffer ) <===]
    static constexpr auto kernelAccess_s =
      KernelAccessHelper<MemoryAccessDenominatorsBase, onDevice, fptype_amp>::template kernelAccessField<>; // requires cuda 11.4

    // [Signature (non-const, SCALAR OR VECTOR) ===> fptype_amp_sv& kernelAccess( fptype_amp* buffer ) <===]
#ifndef MGONGPU_CPPSIMD
    static __host__ __device__ inline fptype_amp_sv&
    kernelAccess( fptype_amp* buffer )
    {
      return kernelAccess_s( buffer );
    }
#else
    static inline fptype_amp_sv&
    kernelAccess( fptype_amp* buffer )
    {
      // NB: contiguous aligned arrays (bulk SIMD reinterpret). The precision experiment runs in
      // cppnone/CUDA; the SIMD path is not the target of the fptype_amp stage type.
      return reinterpret_cast<fptype_amp_sv&>( kernelAccess_s( buffer ) );
    }
#endif

    // [Signature (const, SCALAR OR VECTOR) ===> const fptype_amp_sv& kernelAccessConst( const fptype_amp* buffer ) <===]
    static constexpr auto kernelAccessConst_s =
      KernelAccessHelper<MemoryAccessDenominatorsBase, onDevice, fptype_amp>::template kernelAccessFieldConst<>; // requires cuda 11.4

#ifndef MGONGPU_CPPSIMD
    static __host__ __device__ inline const fptype_amp_sv&
    kernelAccessConst( const fptype_amp* buffer )
    {
      return kernelAccessConst_s( buffer );
    }
#else
    static inline const fptype_amp_sv&
    kernelAccessConst( const fptype_amp* buffer )
    {
      return reinterpret_cast<const fptype_amp_sv&>( kernelAccessConst_s( buffer ) );
    }
#endif
  };

  //----------------------------------------------------------------------------

  typedef KernelAccessDenominators<false> HostAccessDenominators;
  typedef KernelAccessDenominators<true> DeviceAccessDenominators;

  //----------------------------------------------------------------------------

} // end namespace mg5amcGpu/mg5amcCpu

#endif // MemoryAccessDenominators_H
