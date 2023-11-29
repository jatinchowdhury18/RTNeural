#include <gmock/gmock.h>

#include <RTNeural/activation/activation.h>

using namespace testing;

TEST(ActivationTest, tanhActivationNameIsReportedCorrectly)
{
    EXPECT_THAT(RTNeural::TanhActivation<float>(1).getName(), Eq("tanh"));
}

TEST(ActivationTest, tanhActivationPassMatchesSTL)
{
    auto const input = std::vector<float> { -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 3.0f };
    auto tanh = RTNeural::TanhActivation<float>(input.size());
    auto output = std::vector<float>(input.size());
    tanh.forward(input.data(), output.data());

    auto const expected = std::vector<float> {
        std::tanh(-2.0f),
        std::tanh(-1.0f),
        std::tanh(0.0f),
        std::tanh(1.0f),
        std::tanh(2.0f),
        std::tanh(3.0f),
    };

    EXPECT_THAT(output, Pointwise(FloatNear(1e-6f), expected));
}

TEST(ActivationTest, reluActivationNameIsReportedCorrectly)
{
    EXPECT_THAT(RTNeural::ReLuActivation<float>(1).getName(), Eq("relu"));
}

TEST(ActivationTest, reluActivationPassClipsNegativeVaulesToZero)
{
    auto const input = std::vector<float> {
        -1e5f,
        -1.0f,
        std::nextafter(0.0f, -1.0f),
        0.0f,
        std::nextafter(0.0f, 1.0f),
        1.0f,
        1e5f,
    };
    auto relu = RTNeural::ReLuActivation<float>(input.size());
    auto output = std::vector<float>(input.size());
    relu.forward(input.data(), output.data());

    auto const expected = std::vector<float> {
        0.0f,
        0.0f,
        0.0f,
        0.0f,
        std::nextafter(0.0f, 1.0f),
        1.0f,
        1e5f,
    };

    EXPECT_THAT(output, Pointwise(FloatNear(1e-6f), expected));
}
