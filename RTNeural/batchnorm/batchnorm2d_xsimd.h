#ifndef RTNEURAL_BATCHNORM2D_XSIMD_H
#define RTNEURAL_BATCHNORM2D_XSIMD_H

#include "../Layer.h"
#include <xsimd/xsimd.hpp>

namespace RTNeural
{
/** Dynamic batch normalization layer. */
template <typename T>
class BatchNorm2DLayer final : public Layer<T>
{
public:
    BatchNorm2DLayer(int num_filters, int num_features);

    /** Returns the name of this layer. */
    std::string getName() const noexcept override { return "batchnorm2d"; }

    /** Performs forward propagation for this layer. */
    inline void forward(const T* input, T* out) noexcept override
    {
        for(int i = 0; i < num_features; i++)
        {
            const auto* inCol = input + i * num_filters;
            auto* outCol = out + i * num_filters;
            xsimd::transform(inCol, inCol + num_filters, running_mean.begin(), outCol,
                [](auto const& a, auto const& b)
                { return a - b; });
            xsimd::transform(outCol, outCol + num_filters, multiplier.begin(), outCol,
                [](auto const& a, auto const& b)
                { return a * b; });
            xsimd::transform(outCol, outCol + num_filters, beta.begin(), outCol,
                [](auto const& a, auto const& b)
                { return a + b; });
        }
    }

    /** Sets the layer "gamma" values. */
    void setGamma(const std::vector<T>& gammaVals);

    /** Sets the layer "beta" values. */
    void setBeta(const std::vector<T>& betaVals);

    /** Sets the layer's trained running mean. */
    void setRunningMean(const std::vector<T>& runningMean);

    /** Set's the layer's trained running variance. */
    void setRunningVariance(const std::vector<T>& runningVar);

    /** Set's the layer "epsilon" value. */
    void setEpsilon(T epsilon);

private:
    void updateMultiplier();

    const int num_filters;
    const int num_features;

    using vec_type = std::vector<T, xsimd::aligned_allocator<T>>;

    vec_type gamma;
    vec_type beta;

    vec_type running_mean;
    vec_type running_var;

    vec_type multiplier;

    T epsilon = (T)0;
};

/** Static batch normalization layer. */
template <typename T, int num_filters_t, int num_features_t, bool affine = true>
class BatchNorm2DT
{
public:
    static constexpr auto in_size = num_filters_t * num_features_t;
    static constexpr auto out_size = num_filters_t * num_features_t;
    static constexpr auto num_filters = num_filters_t;
    static constexpr auto num_features = num_features_t;
    static constexpr bool is_affine = affine;

    BatchNorm2DT();

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "batchnorm2d"; }

    /** Returns false since batch-norm is not an activation layer. */
    constexpr bool isActivation() const noexcept { return false; }

    /** Resets the layer state. */
    void reset() { }

    /** Performs forward propagation for this layer. */
    template <bool isAffine = affine>
    inline typename std::enable_if<isAffine, void>::type
    forward(const T (&ins)[in_size]) noexcept
    {
        // pad along "num_filters"

        for(int i = 0; i < num_features; i++)
        {
            for(int j = 0; j < num_filters; ++j)
            {
                outs[i * num_filters + j] = (ins[i * num_filters + j] - running_mean[j]) * multiplier[j] + beta[j];
            }
        }
    }

    /** Performs forward propagation for this layer. */
    template <bool isAffine = affine>
    inline typename std::enable_if<!isAffine, void>::type
    forward(const T (&ins)[in_size]) noexcept
    {
        for(int i = 0; i < num_features; i++)
        {
            for(int j = 0; j < num_filters; ++j)
            {
                outs[i * num_filters + j] = (ins[i * num_filters + j] - running_mean[j]) * multiplier[j];
            }
        }
    }

    /** Sets the layer "gamma" values. */
    template <bool isAffine = affine>
    typename std::enable_if<isAffine, void>::type setGamma(const std::vector<T>& gammaVals);

    /** Sets the layer "gamma" values. */
    template <bool isAffine = affine>
    typename std::enable_if<!isAffine, void>::type setGamma(const std::vector<T>&) { }

    /** Sets the layer "beta" values. */
    template <bool isAffine = affine>
    typename std::enable_if<isAffine, void>::type setBeta(const std::vector<T>& betaVals);

    /** Sets the layer "beta" values. */
    template <bool isAffine = affine>
    typename std::enable_if<!isAffine, void>::type setBeta(const std::vector<T>&) { }

    /** Sets the layer's trained running mean. */
    void setRunningMean(const std::vector<T>& runningMean);

    /** Set's the layer's trained running variance. */
    void setRunningVariance(const std::vector<T>& runningVar);

    /** Set's the layer "epsilon" value. */
    void setEpsilon(T epsilon);

    T outs alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];

private:
    void updateMultiplier();

    alignas(RTNEURAL_DEFAULT_ALIGNMENT) T gamma[num_filters_t];
    alignas(RTNEURAL_DEFAULT_ALIGNMENT) T beta[num_filters_t];

    alignas(RTNEURAL_DEFAULT_ALIGNMENT) T running_mean[num_filters_t];
    alignas(RTNEURAL_DEFAULT_ALIGNMENT) T running_var[num_filters_t];

    alignas(RTNEURAL_DEFAULT_ALIGNMENT) T multiplier[num_filters_t];

    T epsilon = (T)0;
};
}

#endif // RTNEURAL_BATCHNORM2D_XSIMD_H
