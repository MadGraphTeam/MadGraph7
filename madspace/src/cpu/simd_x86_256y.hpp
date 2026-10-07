// AVX512 instructions (F, VL, DQ) with 256-bit vectors, like the madmatrix avx512y mode
#include <cstddef>
#include <immintrin.h>
#include <sleef.h>

constexpr int simd_vec_size = 4;

struct FVec {
    FVec() = default;
    FVec(__m256d _v) : v(_v) {};
    FVec(double _v) : v(_mm256_set1_pd(_v)) {};
    explicit FVec(__m256i _v) : v(_mm256_cvtepi64_pd(_v)) {};
    operator __m256d() { return v; }
    FVec operator+=(FVec _v) {
        v = _mm256_add_pd(v, _v);
        return v;
    }
    __m256d v;
};

struct IVec {
    IVec() = default;
    IVec(__m256i _v) : v(_v) {};
    IVec(int _v) : v(_mm256_set1_epi64x(_v)) {};
    explicit IVec(__m256d _v) : v(_mm256_cvttpd_epi64(_v)) {};
    operator __m256i() { return v; }
    IVec operator+=(IVec _v) {
        v = _mm256_add_epi64(v, _v);
        return v;
    }
    __m256i v;
};

struct BVec {
    BVec() = default;
    BVec(bool _v) : v(_v ? 0x0F : 0) {};
    BVec(__mmask8 _v) : v(_v) {};
    operator __mmask8() { return v; }
    __mmask8 v;
};

inline __m256i stride_seq(std::size_t stride) {
    return _mm256_mullo_epi64(
        _mm256_set1_epi64x(stride), _mm256_set_epi64x(3, 2, 1, 0)
    );
}

inline __m256i
mem_indices(std::size_t batch_stride, std::size_t index_stride, IVec indices) {
    return _mm256_add_epi64(
        _mm256_mullo_epi64(indices, _mm256_set1_epi64x(index_stride)),
        stride_seq(batch_stride)
    );
}

inline FVec vgather(
    double* base_ptr, std::size_t batch_stride, std::size_t index_stride, IVec indices
) {
    return _mm256_i64gather_pd(
        base_ptr, mem_indices(batch_stride, index_stride, indices), 8
    );
}

inline IVec vgather(
    int* base_ptr, std::size_t batch_stride, std::size_t index_stride, IVec indices
) {
    return _mm256_cvtepi32_epi64(_mm256_i64gather_epi32(
        base_ptr, mem_indices(batch_stride, index_stride, indices), 4
    ));
}

inline FVec vload(double* base_ptr, std::size_t stride) {
    return _mm256_i64gather_pd(base_ptr, stride_seq(stride), 8);
}

inline IVec vload(int* base_ptr, std::size_t stride) {
    return _mm256_cvtepi32_epi64(
        _mm256_i64gather_epi32(base_ptr, stride_seq(stride), 4)
    );
}

inline void vscatter(
    double* base_ptr,
    std::size_t batch_stride,
    std::size_t index_stride,
    IVec indices,
    FVec values
) {
    _mm256_i64scatter_pd(
        base_ptr, mem_indices(batch_stride, index_stride, indices), values, 8
    );
}

inline void vscatter(
    int* base_ptr,
    std::size_t batch_stride,
    std::size_t index_stride,
    IVec indices,
    IVec values
) {
    _mm256_i64scatter_epi32(
        base_ptr,
        mem_indices(batch_stride, index_stride, indices),
        _mm256_cvtepi64_epi32(values),
        4
    );
}

inline void vstore(double* base_ptr, std::size_t stride, FVec values) {
    _mm256_i64scatter_pd(base_ptr, stride_seq(stride), values, 8);
}

inline void vstore(int* base_ptr, std::size_t stride, IVec values) {
    _mm256_i64scatter_epi32(
        base_ptr, stride_seq(stride), _mm256_cvtepi64_epi32(values), 4
    );
}

inline FVec where(BVec arg1, FVec arg2, FVec arg3) {
    return _mm256_mask_blend_pd(arg1, arg3, arg2);
}
inline IVec where(BVec arg1, IVec arg2, IVec arg3) {
    return _mm256_mask_blend_epi64(arg1, arg3, arg2);
}
inline std::size_t single_index(IVec arg) {
    return _mm_cvtsi128_si64(_mm256_castsi256_si128(arg));
}
inline FVec min(FVec arg1, FVec arg2) { return _mm256_min_pd(arg1, arg2); }
inline FVec max(FVec arg1, FVec arg2) { return _mm256_max_pd(arg1, arg2); }

inline BVec operator==(FVec arg1, FVec arg2) {
    return _mm256_cmp_pd_mask(arg1, arg2, _CMP_EQ_OQ);
}
inline BVec operator!=(FVec arg1, FVec arg2) {
    return _mm256_cmp_pd_mask(arg1, arg2, _CMP_NEQ_UQ);
}
inline BVec operator>(FVec arg1, FVec arg2) {
    return _mm256_cmp_pd_mask(arg1, arg2, _CMP_GT_OQ);
}
inline BVec operator<(FVec arg1, FVec arg2) {
    return _mm256_cmp_pd_mask(arg1, arg2, _CMP_LT_OQ);
}
inline BVec operator>=(FVec arg1, FVec arg2) {
    return _mm256_cmp_pd_mask(arg1, arg2, _CMP_GE_OQ);
}
inline BVec operator<=(FVec arg1, FVec arg2) {
    return _mm256_cmp_pd_mask(arg1, arg2, _CMP_LE_OQ);
}

inline BVec operator&(BVec arg1, BVec arg2) {
    return static_cast<__mmask8>(arg1.v & arg2.v & 0x0F);
}
inline BVec operator|(BVec arg1, BVec arg2) {
    return static_cast<__mmask8>((arg1.v | arg2.v) & 0x0F);
}
inline BVec operator!(BVec arg1) { return static_cast<__mmask8>(~arg1.v & 0x0F); }

inline BVec operator==(IVec arg1, IVec arg2) {
    return _mm256_cmpeq_epi64_mask(arg1, arg2);
}
inline BVec operator!=(IVec arg1, IVec arg2) {
    return _mm256_cmpneq_epi64_mask(arg1, arg2);
}
inline BVec operator>(IVec arg1, IVec arg2) {
    return _mm256_cmpgt_epi64_mask(arg1, arg2);
}
inline BVec operator>=(IVec arg1, IVec arg2) {
    return _mm256_cmpge_epi64_mask(arg1, arg2);
}
inline BVec operator<(IVec arg1, IVec arg2) {
    return _mm256_cmplt_epi64_mask(arg1, arg2);
}
inline BVec operator<=(IVec arg1, IVec arg2) {
    return _mm256_cmple_epi64_mask(arg1, arg2);
}

inline FVec operator-(FVec arg1) { return _mm256_sub_pd(_mm256_set1_pd(0.), arg1); }
inline FVec operator+(FVec arg1, FVec arg2) { return _mm256_add_pd(arg1, arg2); }
inline FVec operator-(FVec arg1, FVec arg2) { return _mm256_sub_pd(arg1, arg2); }
inline FVec operator*(FVec arg1, FVec arg2) { return _mm256_mul_pd(arg1, arg2); }
inline FVec operator/(FVec arg1, FVec arg2) { return _mm256_div_pd(arg1, arg2); }
inline IVec operator-(IVec arg1) {
    return _mm256_sub_epi64(_mm256_setzero_si256(), arg1);
}
inline IVec operator+(IVec arg1, IVec arg2) { return _mm256_add_epi64(arg1, arg2); }
inline IVec operator-(IVec arg1, IVec arg2) { return _mm256_sub_epi64(arg1, arg2); }

inline BVec isnan(FVec arg) { return arg != arg; }

inline FVec sqrt(FVec arg1) { return Sleef_sqrtd4_u05avx2(arg1); }
inline FVec sin(FVec arg1) { return Sleef_sind4_u10avx2(arg1); }
inline FVec cos(FVec arg1) { return Sleef_cosd4_u10avx2(arg1); }
inline FVec sinh(FVec arg1) { return Sleef_sinhd4_u10avx2(arg1); }
inline FVec asinh(FVec arg1) { return Sleef_asinhd4_u10avx2(arg1); }
inline FVec cosh(FVec arg1) { return Sleef_coshd4_u10avx2(arg1); }
inline FVec tanh(FVec arg1) { return Sleef_tanhd4_u10avx2(arg1); }
inline FVec atan2(FVec arg1, FVec arg2) { return Sleef_atan2d4_u10avx2(arg1, arg2); }
inline FVec pow(FVec arg1, FVec arg2) { return Sleef_powd4_u10avx2(arg1, arg2); }
inline FVec fabs(FVec arg1) { return Sleef_fabsd4_avx2(arg1); }
inline FVec log(FVec arg1) { return Sleef_logd4_u10avx2(arg1); }
inline FVec tan(FVec arg1) { return Sleef_tand4_u10avx2(arg1); }
inline FVec atan(FVec arg1) { return Sleef_atand4_u10avx2(arg1); }
inline FVec acos(FVec arg1) { return Sleef_acosd4_u10avx2(arg1); }
inline FVec atanh(FVec arg1) { return Sleef_atanhd4_u10avx2(arg1); }
inline FVec exp(FVec arg1) { return Sleef_expd4_u10avx2(arg1); }
inline FVec log1p(FVec arg1) { return Sleef_log1pd4_u10avx2(arg1); }
inline FVec expm1(FVec arg1) { return Sleef_expm1d4_u10avx2(arg1); }
inline FVec erf(FVec arg1) { return Sleef_erfd4_u10avx2(arg1); }
inline FVec fma(FVec arg1, FVec arg2, FVec arg3) {
    return _mm256_fmadd_pd(arg1, arg2, arg3);
}
