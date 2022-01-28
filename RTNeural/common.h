#pragma once

namespace RTNeural
{

/** Divides two numbers and rounds up if there is a remainder. */
template <typename T>
constexpr T ceil_div(T num, T den)
{
    return (num + den - 1) / den;
}

/** Pade approximation of std::tanh() */
template <typename T>
static inline T tanh_approx(T x) noexcept
{
    constexpr auto clamp = (T)5.7;
    x = x > clamp ? clamp : (x < -clamp ? -clamp : x); // clamp to range [-clamp, clamp]

    auto x2 = x * x;
    auto numerator = x * ((T)2027025 + x2 * ((T)270270 + x2 * ((T)6930 + (T)36 * x2)));
    auto denominator = (T)2027025 + x2 * ((T)945945 + x2 * ((T)51975 + x2 * ((T)630 + x2)));
    return numerator / denominator;
}

/** Pade approximation of sigmoid */
template <typename T>
static inline T sigmoid_approx(T x) noexcept
{
    constexpr auto clamp = (T)7.0;
    x = x > clamp ? clamp : (x < -clamp ? -clamp : x); // clamp to range [-clamp, clamp]

    auto numerator = (T) 0.5 + x * ((T)0.24992827750481075 + x * ((T)0.05251880704605764 + x * ((T)0.005480993464771597 + x * (T)0.000236800130780413)));
    auto denominator = (T) 1.0 + x * ((T)-1.8312068766514214e-14 + x * ((T)0.10503761409212416 + x * ((T)-1.6911015553389271e-15 + x * (T)0.00047360026156094913)));
    return numerator / denominator;
}

} // namespace RTNeural

#if RTNEURAL_USE_EIGEN
#include <Eigen/Dense>

namespace RTNeural
{
#if RTNEURAL_DEFAULT_ALIGNMENT == 32
constexpr auto RTNeuralEigenAlignment = Eigen::Aligned32;
#else
constexpr auto RTNeuralEigenAlignment = Eigen::Aligned16;
#endif

template <typename T>
static inline void
sigmoid(Eigen::Matrix<T, Eigen::Dynamic, 1>& vector) noexcept
{
    vector = (T)1 / (((T)-1 * vector.array()).array().exp() + (T)1);
}

template <typename T>
static inline void
softmax(Eigen::Matrix<T, Eigen::Dynamic, 1>& vector) noexcept
{
    vector = vector.array().exp();
    vector = vector / vector.sum();
}

template <typename T, typename MatType>
static inline auto fast_tanh(const MatType& in)
{
    constexpr auto clamp = (T)5.7;
    auto xc = in.cwiseMin(clamp).cwiseMax(-clamp); // clamp to range [-clamp, clamp]

    auto x2 = xc.array().square();
    auto numerator = xc.array().cwiseProduct(((T)2027025 + x2.cwiseProduct((T)270270 + x2.cwiseProduct((T)6930 + (T)36 * x2.array()).array()).array()));
    auto denominator = (T)2027025 + x2.cwiseProduct((T)945945 + x2.cwiseProduct((T)51975 + x2.cwiseProduct((T)630 + x2.array()).array()).array()).array();
    return numerator.cwiseProduct(denominator.inverse());
}

template <typename T, typename MatType>
static inline auto fast_sigmoid(const MatType& in)
{
    constexpr auto clamp = (T)7.0;
    auto xc = in.cwiseMin(clamp).cwiseMax(-clamp); // clamp to range [-clamp, clamp]

    auto numerator = (T) 0.5 + xc.array().cwiseProduct((T)0.24992827750481075 + xc.array().cwiseProduct((T)0.05251880704605764 + xc.array().cwiseProduct((T)0.005480993464771597 + (T)0.000236800130780413 * xc.array()).array()).array()).array();
    auto denominator = (T) 1.0 + xc.array().cwiseProduct((T)-1.8312068766514214e-14 + xc.array().cwiseProduct((T)0.10503761409212416 + xc.array().cwiseProduct((T)-1.6911015553389271e-15 + (T)0.00047360026156094913 * xc.array()).array()).array()).array();
    return numerator.cwiseProduct(denominator.inverse());
}

} // namespace RTNeural

#elif RTNEURAL_USE_XSIMD
#include <xsimd/xsimd.hpp>

namespace RTNeural
{

template <typename T>
static inline xsimd::simd_type<T> set_value(xsimd::simd_type<T> x, int idx, T value)
{
    union UnionType
    {
        xsimd::simd_type<T> v;
        T s[xsimd::simd_type<T>::size];
    };
    UnionType u { x };

    u.s[idx] = value;
    return u.v;
}

template <typename T>
static inline T get_value(xsimd::simd_type<T> x, int idx)
{
    union UnionType
    {
        xsimd::simd_type<T> v;
        T s[xsimd::simd_type<T>::size];
    };
    UnionType u { x };

    return u.s[idx];
}

template <typename T>
static inline T vMult(const T* arg1, const T* arg2, T* prod,
    int dim) noexcept
{
    xsimd::transform(arg1, &arg1[dim], arg2, prod,
        [](auto const& a, auto const& b) { return a * b; });

    return xsimd::reduce(prod, &prod[dim], (T)0);
}

template <typename T>
static inline void vAdd(const T* in1, const T* in2, T* out,
    int dim) noexcept
{
    xsimd::transform(in1, &in1[dim], in2, out,
        [](auto const& a, auto const& b) { return a + b; });
}

template <typename T>
static inline void vSub(const T* in1, const T* in2, T* out,
    int dim) noexcept
{
    xsimd::transform(in1, &in1[dim], in2, out,
        [](auto const& a, auto const& b) { return a - b; });
}

template <typename T>
static inline void vProd(const T* in1, const T* in2, T* out,
    int dim) noexcept
{
    xsimd::transform(in1, &in1[dim], in2, out,
        [](auto const& a, auto const& b) { return a * b; });
}

template <typename T>
static inline void vCopy(const T* in, T* out, int dim) noexcept
{
    using b_type = xsimd::simd_type<T>;
    constexpr auto inc = (int)b_type::size;

    // size for which the vectorization is possible
    auto vec_size = dim - dim % inc;
    for(int i = 0; i < vec_size; i += inc)
    {
        b_type vec = xsimd::load_aligned(&in[i]);
        xsimd::store_aligned(&out[i], vec);
    }

    // Remaining part that cannot be vectorize
    for(auto i = vec_size; i < dim; ++i)
        out[i] = in[i];
}

template <typename T>
static inline void sigmoid(const T* in, T* out, int dim) noexcept
{
    using b_type = xsimd::simd_type<T>;
    constexpr auto inc = (int)b_type::size;

    // size for which the vectorization is possible
    auto vec_size = dim - dim % inc;
    for(int i = 0; i < vec_size; i += inc)
    {
        b_type x_vec = xsimd::load_aligned(&in[i]);
        b_type y_vec = 1.0 / (1.0 + xsimd::exp(-x_vec));
        xsimd::store_aligned(&out[i], y_vec);
    }

    // Remaining part that cannot be vectorize
    for(auto i = vec_size; i < dim; ++i)
        out[i] = 1.0 / (1.0 + std::exp(-in[i]));
}

template <typename T>
static inline void softmax(const T* in, T* out, int dim) noexcept
{
    using b_type = xsimd::simd_type<T>;
    constexpr auto inc = (int)b_type::size;

    T exp_sum = 0;

    // size for which the vectorization is possible
    auto vec_size = dim - dim % inc;
    for(int i = 0; i < vec_size; i += inc)
    {
        b_type x_vec = xsimd::load_aligned(&in[i]);
        b_type y_vec = xsimd::exp(x_vec);
        exp_sum += xsimd::hadd(y_vec);
        xsimd::store_aligned(&out[i], y_vec);
    }

    // Remaining part that cannot be vectorize
    for(auto i = vec_size; i < dim; ++i)
    {
        out[i] = std::exp(in[i]);
        exp_sum += out[i];
    }

    for(int i = 0; i < vec_size; i += inc)
    {
        b_type x_vec = xsimd::load_aligned(&out[i]);
        b_type y_vec = x_vec / exp_sum;
        xsimd::store_aligned(&out[i], y_vec);
    }

    // Remaining part that cannot be vectorize
    for(auto i = vec_size; i < dim; ++i)
    {
        out[i] /= exp_sum;
    }
}

template <typename T>
static inline void tanh(const T* in, T* out, int dim) noexcept
{
    using b_type = xsimd::simd_type<T>;
    constexpr auto inc = (int)b_type::size;

    // size for which the vectorization is possible
    auto vec_size = dim - dim % inc;
    for(int i = 0; i < vec_size; i += inc)
    {
        b_type x_vec = xsimd::load_aligned(&in[i]);
        b_type y_vec = xsimd::tanh(x_vec);
        xsimd::store_aligned(&out[i], y_vec);
    }

    // Remaining part that cannot be vectorize
    for(auto i = vec_size; i < dim; ++i)
        out[i] = std::tanh(in[i]);
}

template <typename T>
static inline xsimd::simd_type<T> fast_tanh(const xsimd::simd_type<T>& x) noexcept
{
    using b_type = xsimd::simd_type<T>;

    static const b_type clamp_hi((T)5.7);
    static const b_type clamp_lo((T)-5.7);
    auto xc = xsimd::clip(x, clamp_lo, clamp_hi); // clamp to range [-clamp, clamp]

    static const b_type v2027025((T)2027025);
    static const b_type v270270((T)270270);
    static const b_type v6930((T)6930);
    static const b_type v36((T)36);
    static const b_type v945945((T)945945);
    static const b_type v51975((T)51975);
    static const b_type v630((T)630);

    auto x2 = xc * xc;
    auto numerator = xc * (v2027025 + x2 * (v270270 + x2 * (v6930 + v36 * x2)));
    auto denominator = v2027025 + x2 * (v945945 + x2 * (v51975 + x2 * (v630 + x2)));
    return numerator / denominator;
}

template <typename T>
static inline void fast_tanh(const T* in, T* out, int dim) noexcept
{
    using b_type = xsimd::simd_type<T>;
    constexpr auto inc = (int)b_type::size;

    // size for which the vectorization is possible
    auto vec_size = dim - dim % inc;
    for(int i = 0; i < vec_size; i += inc)
    {
        b_type x_vec = xsimd::load_aligned(&in[i]);
        b_type y_vec = fast_tanh<T>(x_vec);
        xsimd::store_aligned(&out[i], y_vec);
    }

    // Remaining part that cannot be vectorize
    for(auto i = vec_size; i < dim; ++i)
        out[i] = tanh_approx(in[i]);
}

template <typename T>
static inline xsimd::simd_type<T> fast_sigmoid(const xsimd::simd_type<T>& x) noexcept
{
    using b_type = xsimd::simd_type<T>;

    static const b_type clamp_hi((T)7.0);
    static const b_type clamp_lo((T)-7.0);
    auto xc = xsimd::clip(x, clamp_lo, clamp_hi); // clamp to range [-clamp, clamp]

    static const b_type n0((T)0.5);
    static const b_type n1((T)0.24992827750481075);
    static const b_type n2((T)0.05251880704605764);
    static const b_type n3((T)0.005480993464771597);
    static const b_type n4((T)0.000236800130780413);
    static const b_type d0((T)1.0);
    static const b_type d1((T)-1.8312068766514214e-14);
    static const b_type d2((T)0.10503761409212416);
    static const b_type d3((T)-1.6911015553389271e-15);
    static const b_type d4((T)0.00047360026156094913);

    auto numerator = n0 + xc * (n1 + xc * (n2 + xc * (n3 + xc * n4)));
    auto denominator = d0 + xc * (d1 + xc * (d2 + xc * (d3 + xc * d4)));
    return numerator / denominator;
}

template <typename T>
static inline void fast_sigmoid(const T* in, T* out, int dim) noexcept
{
    using b_type = xsimd::simd_type<T>;
    constexpr auto inc = (int)b_type::size;

    // size for which the vectorization is possible
    auto vec_size = dim - dim % inc;
    for(int i = 0; i < vec_size; i += inc)
    {
        b_type x_vec = xsimd::load_aligned(&in[i]);
        b_type y_vec = fast_sigmoid<T>(x_vec);
        xsimd::store_aligned(&out[i], y_vec);
    }

    // Remaining part that cannot be vectorize
    for(auto i = vec_size; i < dim; ++i)
        out[i] = sigmoid_approx(in[i]);
}

} // namespace RTNeural

#elif RTNEURAL_USE_ACCELERATE
#include <Accelerate/Accelerate.h>

namespace RTNeural
{

static inline void sigmoid(const float* in, float* out, int dim) noexcept
{
    constexpr float one = 1.0f;
    constexpr float neg_one = -1.0f;
    const auto dim_int = static_cast<int>(dim);

    vDSP_vsmul(in, 1, &neg_one, out, 1, dim);
    vvexpf(out, out, &dim_int);
    vDSP_vsadd(out, 1, &one, out, 1, dim);
    vvrecf(out, out, &dim_int);
}

static inline void sigmoid(const double* in, double* out, int dim) noexcept
{
    constexpr double one = 1.0;
    constexpr double neg_one = -1.0;
    const auto dim_int = static_cast<int>(dim);

    vDSP_vsmulD(in, 1, &neg_one, out, 1, dim);
    vvexp(out, out, &dim_int);
    vDSP_vsaddD(out, 1, &one, out, 1, dim);
    vvrec(out, out, &dim_int);
}

static inline void softmax(const float* in, float* out, int dim) noexcept
{
    const auto dim_int = static_cast<int>(dim);
    float exp_sum;

    vvexpf(out, in, &dim_int);
    vDSP_sve(out, 1, &exp_sum, dim);
    vDSP_vsdiv(out, 1, &exp_sum, out, 1, dim);
}

static inline void softmax(const double* in, double* out, int dim) noexcept
{
    const auto dim_int = static_cast<int>(dim);
    double exp_sum;

    vvexp(out, in, &dim_int);
    vDSP_sveD(out, 1, &exp_sum, dim);
    vDSP_vsdivD(out, 1, &exp_sum, out, 1, dim);
}

} // namespace RTNeural

#else // STL backend
#include <algorithm>
#include <cmath>
#include <numeric>

namespace RTNeural
{

template <typename T>
static inline T vMult(const T* arg1, const T* arg2, int dim) noexcept
{
    return std::inner_product(arg1, arg1 + dim, arg2, (T)0);
}

template <typename T>
static inline T sigmoid(T value) noexcept
{
    return (T)1 / ((T)1 + std::exp(-value));
}

template <typename T>
static inline void softmax(const T* input, T* out, int size) noexcept
{
    T exp_sum = 0;
    for(int i = 0; i < size; ++i)
    {
        out[i] = std::exp(input[i]);
        exp_sum += out[i];
    }

    for(int i = 0; i < size; ++i)
    {
        out[i] /= exp_sum;
    }
}

} // namespace RTNeural

#endif
