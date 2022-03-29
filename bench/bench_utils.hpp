#pragma once

#include <random>
#include <vector>

#if RTNEURAL_USE_XSIMD
#include <xsimd/xsimd.hpp>
using vec_type = std::vector<double, xsimd::aligned_allocator<double>>;
#elif RTNEURAL_USE_EIGEN
#include <Eigen/Dense>
using vec_type = std::vector<double, Eigen::aligned_allocator<double>>;
#else
using vec_type = std::vector<double>;
#endif

std::vector<vec_type> generate_signal(size_t n_samples,
    size_t in_size)
{
    std::vector<vec_type> signal(n_samples);
    for(auto& x : signal)
        x.resize(in_size, 0.0);

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    for(size_t i = 0; i < n_samples; ++i)
        for(size_t k = 0; k < in_size; ++k)
            signal[i][k] = distribution(generator);

    return std::move(signal);
}
