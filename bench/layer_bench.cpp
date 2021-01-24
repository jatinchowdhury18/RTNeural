#include <chrono>
#include <iostream>
#include "layer_creator.hpp"
#include <RTNeural.h>

void help()
{
    std::cout << "RTNeural layer benchmarks:" << std::endl;
    std::cout << "Usage: rtneural_layer_bench <layer_type> <length> <in_size> <out_size>" << std::endl;
    std::cout << "    Note that for activation layers the out_size argument is ignored." << std::endl;
}

std::vector<std::vector<double>> generate_signal(size_t n_samples, size_t in_size)
{
    std::vector<std::vector<double>> signal (n_samples);
    for(auto& x : signal)
        x.resize(in_size, 0.0);

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution (-1.0, 1.0);

    for(size_t i = 0; i < n_samples; ++i)
        for(size_t k = 0; k < in_size; ++k)
            signal[i][k] = distribution(generator);

    return std::move(signal);
}

int main(int argc, char* argv[])
{
    if(argc < 4 || argc > 5)
    {
        help();
        return 1;
    }

    std::string layer_type = argv[1];
    if(layer_type == "--help")
    {
        help();
        return 1;
    }

    // parse args
    const auto length_seconds = std::atof(argv[2]);
    const auto in_size = std::atol(argv[3]);
    const auto out_size = argc == 5 ? std::atol(argv[4]) : in_size;
    std::cout << "Benchmarking " << layer_type
        << " layer, with input size " << in_size
        << " and output size " << out_size
        << ", with signal length " << length_seconds
        << " seconds" << std::endl;

    // create layer
    auto layer = create_layer(layer_type, in_size, out_size);
    if(layer == nullptr)
        return 1;

    // generate audio
    constexpr double sample_rate = 48000.0;
    const auto n_samples = static_cast<size_t>(sample_rate * length_seconds);
    const auto signal = generate_signal(n_samples, in_size);
    std::vector<double> output (out_size);

    // run benchmark
    using clock_t = std::chrono::high_resolution_clock;
	using second_t = std::chrono::duration<double>;

    auto start = clock_t::now();
    for(size_t i = 0; i < n_samples; ++i)
        layer->forward(signal[i].data(), output.data());
    auto duration = std::chrono::duration_cast<second_t>(clock_t::now() - start).count();

    std::cout << "Processed " << length_seconds << " seconds of signal in "
        << duration << " seconds" << std::endl;
    std::cout << length_seconds / duration << "x real-time" << std::endl;

    return 0;
}
