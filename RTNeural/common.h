#pragma once

#if defined(USE_EIGEN)
#include <Eigen/Dense>

namespace RTNeural {

template <typename T>
static inline void
sigmoid(Eigen::Matrix<T, Eigen::Dynamic, 1> &vector) noexcept {
  vector = (T)1 / (((T)-1 * vector.array()).array().exp() + (T)1);
}

} // namespace RTNeural

#elif defined(USE_XSIMD)
#include <xsimd/xsimd.hpp>

namespace RTNeural {

template <typename T>
static inline T vMult(const T *arg1, const T *arg2, T *prod,
                      size_t dim) noexcept {
  xsimd::transform(arg1, &arg1[dim], arg2, prod,
                   [](auto const &a, auto const &b) { return a * b; });

  return xsimd::reduce(prod, &prod[dim], (T)0);
}

template <typename T>
static inline void vAdd(const T *in1, const T *in2, T *out,
                        size_t dim) noexcept {
  xsimd::transform(in1, &in1[dim], in2, out,
                   [](auto const &a, auto const &b) { return a + b; });
}

template <typename T>
static inline void vSub(const T *in1, const T *in2, T *out,
                        size_t dim) noexcept {
  xsimd::transform(in1, &in1[dim], in2, out,
                   [](auto const &a, auto const &b) { return a - b; });
}

template <typename T>
static inline void vProd(const T *in1, const T *in2, T *out,
                         size_t dim) noexcept {
  xsimd::transform(in1, &in1[dim], in2, out,
                   [](auto const &a, auto const &b) { return a * b; });
}

template <typename T>
static inline void vCopy(const T *in, T *out, size_t dim) noexcept {
  using b_type = xsimd::simd_type<T>;
  auto inc = b_type::size;

  // size for which the vectorization is possible
  auto vec_size = dim - dim % inc;
  for (size_t i = 0; i < vec_size; i += inc) {
    b_type vec = xsimd::load_aligned(&in[i]);
    xsimd::store_aligned(&out[i], vec);
  }

  // Remaining part that cannot be vectorize
  for (auto i = vec_size; i < dim; ++i)
    out[i] = in[i];
}

template <typename T>
static inline void sigmoid(const T *in, T *out, size_t dim) noexcept {
  using b_type = xsimd::simd_type<T>;
  auto inc = b_type::size;

  // size for which the vectorization is possible
  auto vec_size = dim - dim % inc;
  for (size_t i = 0; i < vec_size; i += inc) {
    b_type x_vec = xsimd::load_aligned(&in[i]);
    b_type y_vec = 1.0 / (1.0 + xsimd::exp(-x_vec));
    xsimd::store_aligned(&out[i], y_vec);
  }

  // Remaining part that cannot be vectorize
  for (auto i = vec_size; i < dim; ++i)
    out[i] = 1.0 / (1.0 + std::exp(-in[i]));
}

template <typename T>
static inline void tanh(const T *in, T *out, size_t dim) noexcept {
  using b_type = xsimd::simd_type<T>;
  auto inc = b_type::size;

  // size for which the vectorization is possible
  auto vec_size = dim - dim % inc;
  for (size_t i = 0; i < vec_size; i += inc) {
    b_type x_vec = xsimd::load_aligned(&in[i]);
    b_type y_vec = xsimd::tanh(x_vec);
    xsimd::store_aligned(&out[i], y_vec);
  }

  // Remaining part that cannot be vectorize
  for (auto i = vec_size; i < dim; ++i)
    out[i] = std::tanh(in[i]);
}

} // namespace RTNeural

#else // STL backend
#include <algorithm>
#include <cmath>
#include <numeric>

namespace RTNeural {

template <typename T>
static inline T vMult(const T *arg1, const T *arg2, size_t dim) noexcept {
  return std::inner_product(arg1, arg1 + dim, arg2, (T)0);
}

template <typename T> static inline T sigmoid(T value) noexcept {
  return (T)1 / ((T)1 + std::exp(-value));
}

} // namespace RTNeural

#endif
