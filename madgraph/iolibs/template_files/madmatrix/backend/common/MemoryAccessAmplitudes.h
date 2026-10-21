// Copyright (C) 2020-2026 CERN and UCLouvain.
// Licensed under the GNU Lesser General Public License (version 3 or later).
// Created originally by: A. Valassi (Jan 2022) for the MadGraph7 CUDACPP plugin.
// Further modified by: A. Valassi (2022-2024).
// Integrated with the MadGraph7 project in Feb 2026.

#ifndef MemoryAccessAmplitudes_H
#define MemoryAccessAmplitudes_H 1

#include "MadMatrixConfig.h"

#include "MadMatrixCxtypes.h"

#include "MemoryAccessHelpers.h"

//One namespace. Split ber backend.
namespace madmatrix
{
  //----------------------------------------------------------------------------

  // A class providing access to memory buffers for a given event, based on implicit kernel rules
  // Its methods use the KernelAccessHelper template - note the use of the template keyword in template function instantiations
  template<bool onDevice>
  class KernelAccessAmplitudes
  {
  public:

    static __host__ __device__ inline cxtype_amp_sv*
    kernelAccess( fptype_amp* buffer )
    {
      return reinterpret_cast<cxtype_amp_sv*>( buffer );
    }

    static __host__ __device__ inline const cxtype_amp_sv*
    kernelAccessConst( const fptype_amp* buffer )
    {
      return reinterpret_cast<const cxtype_amp_sv*>( buffer );
    }
  };

  //----------------------------------------------------------------------------

  typedef KernelAccessAmplitudes<false> HostAccessAmplitudes;
  typedef KernelAccessAmplitudes<true> DeviceAccessAmplitudes;

  //----------------------------------------------------------------------------

} // end namespace madmatrix

#endif // MemoryAccessAmplitudes_H
