#include "RTNeural.h"

#if RTNEURAL_USE_XSIMD
// double-check SIMD width...
namespace
{
#if RTNEURAL_AVX2_ENABLED
static_assert(xsimd::simd_type<float>::size == 8, "SIMD float width needs to equal 8!");
static_assert(xsimd::simd_type<double>::size == 4, "SIMD double width needs to equal 4!");
#else
static_assert(xsimd::simd_type<float>::size == 4, "SIMD float width needs to equal 4!");
static_assert(xsimd::simd_type<double>::size == 2, "SIMD double width needs to equal 2!");
#endif // RTNEURAL_AVX2_ENABLED
}
#endif // RTNEURAL_USE_XSIMD

// forward declare some template classes
template class RTNeural::Model<float>;
template class RTNeural::Model<double>;
template class RTNeural::Layer<float>;
template class RTNeural::Layer<double>;
