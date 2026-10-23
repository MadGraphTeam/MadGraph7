// Serial host emulation of the HIP runtime, for the madmatrix GPU templates
// (backend/gpu) on a machine without a GPU toolchain: bin/hipcc compiles them as
// host C++ with this header, every kernel launch runs its grid x block threads one
// after the other on the host, and device memory is host memory. It checks the
// logic of the GPU code (indexing, crossing tables, good helicities, selection)
// against the CPU backends; it cannot replace a real nvcc/hipcc build and run.
// The GPU code has no __syncthreads and no shared memory, so a serial run of the
// threads is a valid schedule; atomicAdd is then a plain addition.
// Compiled with clang -x cuda (__CUDA__), the same header only declares the
// device attributes: a device-side semantic check of the sources.
#ifndef HOST_HIP_RUNTIME_H
#define HOST_HIP_RUNTIME_H 1

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef __CUDA__
#define __device__ __attribute__( ( device ) )
#define __global__ __attribute__( ( global ) )
#define __host__ __attribute__( ( host ) )
#define __constant__ __attribute__( ( constant ) )
#define __shared__ __attribute__( ( shared ) )
#define __forceinline__ __attribute__( ( always_inline ) ) inline
#include <__clang_cuda_builtin_vars.h>
#define HOST_HIP_HD __attribute__( ( host ) ) __attribute__( ( device ) )
#else
#define __device__
#define __global__
#define __host__
#define __constant__
#define __shared__
#define __forceinline__ inline
#define HOST_HIP_HD
#endif

struct dim3
{
  unsigned int x, y, z;
  HOST_HIP_HD constexpr dim3( unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1 )
    : x( x_ ), y( y_ ), z( z_ ) {}
};

#ifdef __CUDA__
extern "C" __host__ int cudaConfigureCall( dim3 grid, dim3 block, size_t shared = 0, void* stream = 0 );
#else
inline dim3 threadIdx( 0, 0, 0 ), blockIdx( 0, 0, 0 ), blockDim( 1, 1, 1 ), gridDim( 1, 1, 1 );
#endif

typedef enum
{
  hipSuccess = 0,
  hipErrorInvalidValue = 1
} hipError_t;
typedef enum
{
  hipMemcpyHostToHost = 0,
  hipMemcpyHostToDevice = 1,
  hipMemcpyDeviceToHost = 2,
  hipMemcpyDeviceToDevice = 3,
  hipMemcpyDefault = 4
} hipMemcpyKind;
typedef void* hipStream_t;
#define hipStreamNonBlocking 1

inline const char* hipGetErrorString( hipError_t e ) { return e == hipSuccess ? "hipSuccess" : "hipError"; }
inline hipError_t hipPeekAtLastError() { return hipSuccess; }
inline hipError_t hipGetLastError() { return hipSuccess; }
inline hipError_t hipSetDevice( int ) { return hipSuccess; }
inline hipError_t hipDeviceSynchronize() { return hipSuccess; }
inline hipError_t hipDeviceReset() { return hipSuccess; }

// A fresh allocation is filled with 0xff bytes (NaN doubles, -1 ints), so that a
// read of memory the GPU code never wrote shows up instead of reading zeros
inline void* host_hip_alloc( size_t n )
{
  void* p = nullptr;
  if( posix_memalign( &p, 256, n ? n : 1 ) != 0 ) return nullptr;
  memset( p, 0xff, n );
  return p;
}
template<typename T>
inline hipError_t hipMalloc( T** p, size_t n ) { *p = (T*)host_hip_alloc( n ); return *p ? hipSuccess : hipErrorInvalidValue; }
template<typename T>
inline hipError_t hipHostMalloc( T** p, size_t n, unsigned int = 0 ) { return hipMalloc( p, n ); }
template<typename T>
inline hipError_t hipMallocAsync( T** p, size_t n, hipStream_t ) { return hipMalloc( p, n ); }
inline hipError_t hipFree( void* p ) { free( p ); return hipSuccess; }
inline hipError_t hipHostFree( void* p ) { free( p ); return hipSuccess; }
inline hipError_t hipFreeAsync( void* p, hipStream_t ) { free( p ); return hipSuccess; }
inline hipError_t hipMemcpy( void* d, const void* s, size_t n, hipMemcpyKind ) { memmove( d, s, n ); return hipSuccess; }
inline hipError_t hipMemcpyAsync( void* d, const void* s, size_t n, hipMemcpyKind, hipStream_t = nullptr ) { memmove( d, s, n ); return hipSuccess; }
inline hipError_t hipMemset( void* d, int v, size_t n ) { memset( d, v, n ); return hipSuccess; }
inline hipError_t hipMemsetAsync( void* d, int v, size_t n, hipStream_t = nullptr ) { memset( d, v, n ); return hipSuccess; }
template<typename T>
inline hipError_t hipMemcpyToSymbol( T& sym, const void* s, size_t n, size_t off = 0, hipMemcpyKind = hipMemcpyHostToDevice )
{
  if( off + n > sizeof( T ) )
  {
    fprintf( stderr, "host hip: hipMemcpyToSymbol of %zu bytes at %zu into a %zu-byte symbol\n", n, off, sizeof( T ) );
    abort();
  }
  memcpy( (char*)&sym + off, s, n );
  return hipSuccess;
}
template<typename T>
inline hipError_t hipMemcpyFromSymbol( void* d, const T& sym, size_t n, size_t off = 0, hipMemcpyKind = hipMemcpyDeviceToHost )
{
  if( off + n > sizeof( T ) )
  {
    fprintf( stderr, "host hip: hipMemcpyFromSymbol of %zu bytes at %zu from a %zu-byte symbol\n", n, off, sizeof( T ) );
    abort();
  }
  memcpy( d, (const char*)&sym + off, n );
  return hipSuccess;
}
template<typename P, typename T>
inline hipError_t hipGetSymbolAddress( P** p, T& sym ) { *p = (P*)&sym; return hipSuccess; }
inline hipError_t hipStreamCreate( hipStream_t* s ) { *s = (hipStream_t)0x1; return hipSuccess; }
inline hipError_t hipStreamCreateWithFlags( hipStream_t* s, unsigned int ) { *s = (hipStream_t)0x2; return hipSuccess; }
inline hipError_t hipStreamDestroy( hipStream_t ) { return hipSuccess; }
inline hipError_t hipStreamSynchronize( hipStream_t ) { return hipSuccess; }

template<typename T>
HOST_HIP_HD inline T atomicAdd( T* a, T v )
{
  T old = *a;
  *a += v;
  return old;
}

#ifndef __CUDA__
namespace host_hip
{
  // bin/hipcc rewrites kernel<<<grid, block, shared, stream>>>( args ) into
  // host_hip::Launch( grid, block, shared, stream )( kernel )( args )
  struct Launch
  {
    dim3 grid, block;
    Launch( dim3 g, dim3 b, size_t = 0, hipStream_t = nullptr ) : grid( g ), block( b ) {}
    template<typename K>
    auto operator()( K kernel ) const
    {
      const dim3 g = grid, b = block;
      return [=]( auto&&... args ) {
        if( g.x * g.y * g.z == 0 || b.x * b.y * b.z == 0 )
        {
          // a real GPU fails the launch (invalid configuration)
          fprintf( stderr, "host hip: invalid launch configuration\n" );
          abort();
        }
        gridDim = g;
        blockDim = b;
        for( unsigned bz = 0; bz < g.z; bz++ )
          for( unsigned by = 0; by < g.y; by++ )
            for( unsigned bx = 0; bx < g.x; bx++ )
              for( unsigned tz = 0; tz < b.z; tz++ )
                for( unsigned ty = 0; ty < b.y; ty++ )
                  for( unsigned tx = 0; tx < b.x; tx++ )
                  {
                    blockIdx = dim3( bx, by, bz );
                    threadIdx = dim3( tx, ty, tz );
                    kernel( args... );
                  }
      };
    }
  };
}
#endif

#endif // HOST_HIP_RUNTIME_H
