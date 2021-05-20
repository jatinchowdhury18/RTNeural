#include "bench_utils.hpp"
#include <RTNeural.h>
#include <chrono>

template <typename ModelType>
double runBench(ModelType& model, double length_seconds)
{
    // generate audio
    constexpr double sample_rate = 48000.0;
    const auto n_samples = static_cast<size_t>(sample_rate * length_seconds);
    const auto signal = generate_signal(n_samples, 1);
    auto y = 0.0;

    // run benchmark
    using clock_t = std::chrono::high_resolution_clock;
    using second_t = std::chrono::duration<double>;

    auto start = clock_t::now();
    for(size_t i = 0; i < n_samples; ++i)
        model.forward(signal[i].data());
    auto duration = std::chrono::duration_cast<second_t>(clock_t::now() - start).count();

    std::cout << "Processed " << length_seconds << " seconds of signal in "
              << duration << " seconds" << std::endl;
    std::cout << length_seconds / duration << "x real-time" << std::endl;

    return duration;
}

int main(int argc, char* argv[])
{
    const std::string model_file = "models/full_model.json";
    constexpr double bench_time = 100.0;
    double nonTemplatedDur = 0.0;

    // non-templated model
    {
        std::cout << "Measuring non-templated model..." << std::endl;
        std::ifstream jsonStream(model_file, std::ifstream::binary);
        auto model = RTNeural::json_parser::parseJson<double>(jsonStream);
        nonTemplatedDur = runBench(*model.get(), bench_time);
    }

#if MODELT_AVAILABLE
    // templated model
    double templatedDur = 0.0;
    {
        std::cout << "Measuring templated model..." << std::endl;
        RTNeural::ModelT<double, 1, 1,
            RTNeural::DenseT<double, 1, 8>,
            RTNeural::TanhActivationT<double, 8>,
            RTNeural::Conv1DT<double, 8, 4, 3, 2>,
            RTNeural::TanhActivationT<double, 4>,
            RTNeural::GRULayerT<double, 4, 8>,
            RTNeural::DenseT<double, 8, 1>>
            modelT;

        std::ifstream jsonStream(model_file, std::ifstream::binary);
        modelT.parseJson(jsonStream);
        templatedDur = runBench(modelT, bench_time);
    }

    std::cout << "Templated model is " << nonTemplatedDur / templatedDur << "x faster!" << std::endl;
#endif
}
