#pragma once

#include <random>
#include <RTNeural.h>

template <typename T>
int fastTanhTest(T limit)
{
    using namespace RTNeural;
    constexpr int layerSize = 8;
    constexpr int nIter = 100;
    constexpr T range = (T) 10;

    auto testTanh = [=] (auto forwardFunc, const T (&test_outs)[layerSize])
    {
        std::default_random_engine generator;
        std::uniform_real_distribution<T> distribution(-range, range);

        constexpr int layerSize = 8; // MSVC can't capture this in the lambda
        T test_ins alignas(RTNEURAL_DEFAULT_ALIGNMENT) [layerSize];
        T actual_outs[layerSize];

        auto maxError = (T) 0;
        auto maxErrorInput = (T) 0;
        for(int i = 0; i < nIter; ++i)
        {
            // generate random input and reference
            for(int n = 0; n < layerSize; ++n)
            {
                test_ins[n] = distribution(generator);
                actual_outs[n] = std::tanh(test_ins[n]);
            }

            // compute layer output
            forwardFunc(test_ins);
            // layer.forward(test_ins, test_outs);

            // check for errors
            for(int n = 0; n < layerSize; ++n)
            {
                auto error = std::abs(actual_outs[n] - test_outs[n]);
                if(error > maxError)
                {
                    maxError = error;
                    maxErrorInput = test_ins[n];
                }
            }
        }

        std::cout << "    Maximum error: " << maxError << ", at input value: " << maxErrorInput << std::endl;
        if(maxError > limit)
        {
            std::cout << "    FAIL: Error is too high!" << std::endl;
            return 1;
        }

        return 0;
    };

    int result = 0;
    auto dtype = std::is_same<T, float>::value ? "float" : "double";

    T test_outs alignas(RTNEURAL_DEFAULT_ALIGNMENT) [layerSize];
    FastTanh<T> fastTanh { layerSize };
    std::cout << "Testing FastTanh for data type " << dtype << std::endl;
    result |= testTanh([&fastTanh, &test_outs] (const T (&test_ins)[layerSize])
        {
            fastTanh.forward(test_ins, test_outs);
        }, test_outs);

    FastTanhT<T, layerSize> fastTanhT;
    std::cout << "Testing FastTanhT for data type " << dtype << std::endl;
#if RTNEURAL_USE_XSIMD
    result |= testTanh([&fastTanhT, &test_outs] (const T (&test_ins)[layerSize])
        {
            constexpr int layerSize = 8; // MSVC can't capture this in the lambda
            using b_type = xsimd::simd_type<T>;
            constexpr auto b_size = (int)b_type::size;
            constexpr auto v_size = layerSize / b_size;

            b_type test_ins_v[v_size];
            for(int i = 0; i < v_size; ++i)
                test_ins_v[i] = xsimd::load_aligned(test_ins + i * b_size);

            fastTanhT.forward(test_ins_v);

            for(int i = 0; i < v_size; ++i)
                xsimd::store_aligned(test_outs + i * b_size, fastTanhT.outs[i]);
        }, test_outs);
#elif RTNEURAL_USE_EIGEN
        result |= testTanh([&fastTanhT, &test_outs] (const T (&test_ins)[layerSize])
        {
            constexpr int layerSize = 8; // MSVC can't capture this in the lambda
            using MatType = Eigen::Matrix<T, layerSize, 1>;
            Eigen::Map<const MatType> test_ins_v (test_ins);

            fastTanhT.forward(test_ins_v);

            for(int i = 0; i < layerSize; ++i)
                test_outs[i] = fastTanhT.outs(i);
        }, test_outs);
#else
    result |= testTanh([&fastTanhT] (const T (&test_ins)[layerSize])
        {
            fastTanhT.forward(test_ins);
        }, fastTanhT.outs);
#endif

    return result;
}

int approximationTests()
{
    int result = 0;
    result |= fastTanhTest<float>(5.1e-5f);
    result |= fastTanhTest<double>(5.1e-5);

    return result;
}
