#pragma once

#include <random>
#include <vector>

std::vector<std::vector<double>> generate_signal(size_t n_samples,
    size_t in_size)
{
    std::vector<std::vector<double>> signal(n_samples);
    for(auto& x : signal)
        x.resize(in_size, 0.0);

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    for(size_t i = 0; i < n_samples; ++i)
        for(size_t k = 0; k < in_size; ++k)
            signal[i][k] = distribution(generator);

    return std::move(signal);
}
